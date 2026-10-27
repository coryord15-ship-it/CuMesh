// fill_worker.cpp -- the seam-compatible R1 fill worker shell.
//
// SEAM (spec section 2.1, the contract that must not move):
//   argv      : [worker, in_path, out_path,
//                repr(float(max_hole_perimeter))]
//   in .npz   : keys "verts" float32 (K,3), "faces" int64 (M,3),
//               chunk-local indexing
//   out .npz  : same keys/dtypes; input prefix bit-exact, fills
//               appended (spec section 2.2)
//   exit      : 0 on success ONLY. Any failure exits nonzero and
//               leaves NO output file behind (the output is written
//               to a temp file in the same directory and renamed
//               into place only once complete), so the parent's
//               ladder (spec section 2.3) sees a clean failure.
//   stateless : nothing is kept between invocations; safe to kill
//               at any point (a kill leaves at most an orphan temp
//               file inside the parent's own per-attempt temp dir,
//               which the parent always removes).
//
// The fill itself is fill_logic (host build) or the HIP path in
// fill_hip.hip.cpp (built with -DR1_USE_HIP, see BUILD-ON-PC.md);
// this file is only argv + .npz plumbing around it.
//
// r35.7 BATCH MODE (the optimization campaign's V1 variant; the
// single-shot seam above is unchanged): job 045 measured the
// per-chunk subprocess toll at ~0.47 s of process spawn + HIP
// runtime init against ~0.017 s of fill compute -- 386,195 chunks
// projected to ~41 h. Batch mode starts the process ONCE:
//   argv      : [worker, --batch, manifest_path]
//   manifest  : one item per line, TAB-separated:
//               in_path <TAB> out_path <TAB> threshold-repr
//   progress  : stdout, flushed per line --
//               "R1-BATCH item <i> start",
//               "R1-BATCH item <i> ok elapsed=<s>
//                appended_faces=<n>", finally
//               "R1-BATCH done items=<n>".
//   exit      : 0 when every item succeeded; 10 IMMEDIATELY on
//               the first item failure (the failing item's
//               R1-FAIL stage line is on stderr as usual) -- a
//               failed item ends the process so the parent
//               re-batches the remainder into a FRESH HIP
//               context and sends the failed item through its
//               single-shot ladder; 2 on manifest errors.
// Every item runs the identical run_one body the single-shot
// seam has always executed; batch mode changes process
// lifetime, never fill math.
//
// .NPZ I/O: an .npz is a ZIP of .npy entries. Reading handles both
// stored (method 0, what numpy.savez writes) and deflated
// (method 8, numpy.savez_compressed) entries, via zlib. Writing
// emits stored entries with a minimal v1.0 .npy header -- plain
// numpy.load reads them (verified end-to-end against the real R1
// harness; see CPU-SMOKE-TESTS.md).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <zlib.h>

#include "fill_logic.h"
#ifdef R1_USE_HIP
#include "fill_hip.h"
#endif

namespace {

// ------------------------------------------------------------------
// Little-endian scalar reads from a byte buffer (the ZIP and NPY
// formats are little-endian; explicit assembly keeps this correct
// regardless of host alignment rules).
// ------------------------------------------------------------------
uint16_t rd16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
uint32_t rd32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

// Read a whole file into memory. Worker inputs are one chunk
// (spec budgets: a few MB); a size sanity cap keeps a corrupt or
// hostile path from making the worker try to swallow a disk.
bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    const long kMaxBytes = 1L << 30;  // 1 GiB, far above any chunk
    if (size < 0 || size > kMaxBytes) { std::fclose(f); return false; }
    out.resize(static_cast<size_t>(size));
    bool ok = size == 0 ||
              std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}

// One ZIP entry located in the archive: where its (possibly
// compressed) data starts, its method, and its true size.
struct ZipEntry {
    bool found = false;
    uint16_t method = 0;
    uint32_t comp_size = 0;
    uint32_t raw_size = 0;
    size_t data_offset = 0;
};

// Find an entry by name via the central directory (authoritative:
// local headers alone cannot be trusted for sizes when data
// descriptors are used), then resolve its data offset through its
// local header (name/extra lengths can differ between the two).
ZipEntry find_zip_entry(const std::vector<uint8_t>& zip,
                        const std::string& name) {
    ZipEntry miss;
    if (zip.size() < 22) return miss;
    // End Of Central Directory: scan the tail (comment <= 65535).
    size_t eocd = SIZE_MAX;
    size_t tail = zip.size() < 65557 ? 0 : zip.size() - 65557;
    for (size_t i = zip.size() - 22 + 1; i-- > tail;) {
        if (rd32(&zip[i]) == 0x06054b50u) { eocd = i; break; }
    }
    if (eocd == SIZE_MAX) return miss;
    uint32_t cd_count = rd16(&zip[eocd + 10]);
    size_t cd = rd32(&zip[eocd + 16]);
    for (uint32_t n = 0; n < cd_count; ++n) {
        if (cd + 46 > zip.size() || rd32(&zip[cd]) != 0x02014b50u)
            return miss;
        uint16_t method = rd16(&zip[cd + 10]);
        uint32_t comp = rd32(&zip[cd + 20]);
        uint32_t raw = rd32(&zip[cd + 24]);
        uint16_t nlen = rd16(&zip[cd + 28]);
        uint16_t xlen = rd16(&zip[cd + 30]);
        uint16_t clen = rd16(&zip[cd + 32]);
        uint32_t lho = rd32(&zip[cd + 42]);
        if (cd + 46 + nlen > zip.size()) return miss;
        std::string ename(reinterpret_cast<const char*>(&zip[cd + 46]),
                          nlen);
        if (ename == name) {
            if (lho + 30 > zip.size() ||
                rd32(&zip[lho]) != 0x04034b50u)
                return miss;
            uint16_t ln = rd16(&zip[lho + 26]);
            uint16_t lx = rd16(&zip[lho + 28]);
            ZipEntry e;
            e.found = true;
            e.method = method;
            e.comp_size = comp;
            e.raw_size = raw;
            e.data_offset = static_cast<size_t>(lho) + 30 + ln + lx;
            if (e.data_offset + comp > zip.size()) return miss;
            return e;
        }
        cd += 46 + nlen + xlen + clen;
    }
    return miss;
}

// Extract one entry's uncompressed bytes. Stored entries copy
// out; deflated entries go through zlib's raw-inflate (ZIP uses
// raw deflate streams, hence -MAX_WBITS).
bool extract_entry(const std::vector<uint8_t>& zip, const ZipEntry& e,
                   std::vector<uint8_t>& out) {
    out.assign(zip.begin() + static_cast<long>(e.data_offset),
               zip.begin() + static_cast<long>(e.data_offset + e.comp_size));
    if (e.method == 0) return out.size() == e.raw_size;
    if (e.method != 8) return false;
    std::vector<uint8_t> raw(e.raw_size);
    z_stream zs{};
    zs.next_in = out.data();
    zs.avail_in = static_cast<uInt>(out.size());
    zs.next_out = raw.data();
    zs.avail_out = static_cast<uInt>(raw.size());
    if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) return false;
    int rc = inflate(&zs, Z_FINISH);
    inflateEnd(&zs);
    if (rc != Z_STREAM_END || zs.total_out != e.raw_size) return false;
    out = std::move(raw);
    return true;
}

// ------------------------------------------------------------------
// Minimal .npy parsing. Accepts exactly the seam's dtypes:
// verts descr "<f4", faces descr "<i8", C order, shape (N, 3).
// Anything else is a loud failure, not a reinterpretation.
// ------------------------------------------------------------------
struct NpyArray {
    std::string descr;         // "<f4" or "<i8"
    int64_t rows = 0;          // shape[0]; shape is (rows, 3)
    size_t data_offset = 0;    // byte offset of the payload
};

bool parse_npy(const std::vector<uint8_t>& buf, NpyArray& arr) {
    if (buf.size() < 10 || std::memcmp(buf.data(), "\x93NUMPY", 6) != 0)
        return false;
    uint8_t major = buf[6];
    size_t header_len, dict_at;
    if (major == 1) {
        header_len = rd16(&buf[8]);
        dict_at = 10;
    } else if (major == 2 || major == 3) {
        header_len = rd32(&buf[8]);
        dict_at = 12;
    } else {
        return false;
    }
    if (dict_at + header_len > buf.size()) return false;
    std::string dict(reinterpret_cast<const char*>(&buf[dict_at]),
                     header_len);
    // Extract the three fields the seam cares about. The header is
    // a Python dict literal; substring extraction of descr /
    // fortran_order / shape is sufficient for files numpy writes,
    // and every extraction is verified below.
    auto field = [&](const char* key) -> std::string {
        size_t p = dict.find(key);
        if (p == std::string::npos) return "";
        p = dict.find(':', p);
        if (p == std::string::npos) return "";
        ++p;
        size_t e = dict.find_first_of(",}", p);
        std::string v = dict.substr(p, e - p);
        size_t a = v.find_first_not_of(" \t'\"");
        size_t b = v.find_last_not_of(" \t'\"");
        return a == std::string::npos ? "" : v.substr(a, b - a + 1);
    };
    arr.descr = field("descr");
    if (field("fortran_order") != "False") return false;
    std::string shape = field("shape");
    // shape looks like "(123, 3)" (field() stopped at the comma
    // inside... no: the tuple's first comma is inside parens; parse
    // the raw text instead for the two integers.)
    size_t sp = dict.find("shape");
    if (sp == std::string::npos) return false;
    const char* s = dict.c_str() + sp;
    int64_t d0 = -1, d1 = -1;
    if (std::sscanf(s, "shape': (%lld, %lld", (long long*)&d0,
                    (long long*)&d1) != 2 &&
        std::sscanf(s, "shape\": (%lld, %lld", (long long*)&d0,
                    (long long*)&d1) != 2) {
        // numpy writes 'shape': (N, 3), -- the scan above covers
        // it; a one-element fallback is not part of the seam.
        return false;
    }
    if (d1 != 3 || d0 < 0) return false;
    arr.rows = d0;
    arr.data_offset = dict_at + header_len;
    size_t elem = arr.descr == "<f4" ? 4 : arr.descr == "<i8" ? 8 : 0;
    if (elem == 0) return false;
    if (arr.data_offset + static_cast<size_t>(d0) * 3 * elem > buf.size())
        return false;
    return true;
}

// Load one named .npy entry from the .npz and materialise it as
// typed rows. `want_descr` pins the dtype ("<f4" / "<i8").
template <typename T>
bool load_array(const std::vector<uint8_t>& zip, const char* entry,
                const char* want_descr, std::vector<T>& out,
                int64_t& rows) {
    ZipEntry e = find_zip_entry(zip, entry);
    if (!e.found) return false;
    std::vector<uint8_t> npy;
    if (!extract_entry(zip, e, npy)) return false;
    NpyArray arr;
    if (!parse_npy(npy, arr)) return false;
    if (arr.descr != want_descr) return false;
    rows = arr.rows;
    out.resize(static_cast<size_t>(rows) * 3);
    if (rows > 0)
        std::memcpy(out.data(), &npy[arr.data_offset],
                    out.size() * sizeof(T));
    return true;
}

// ------------------------------------------------------------------
// Minimal .npy / .npz writing: stored entries only (numpy.load
// reads them; verified against the real harness). The .npy header
// follows numpy's v1.0 layout: dict padded with spaces to a
// 64-byte-aligned total, newline-terminated.
// ------------------------------------------------------------------
std::vector<uint8_t> make_npy(const char* descr, int64_t rows,
                              const void* data, size_t data_bytes) {
    std::string dict = "{'descr': '" + std::string(descr) +
                       "', 'fortran_order': False, 'shape': (" +
                       std::to_string(rows) + ", 3), }";
    size_t total = 10 + 2 + dict.size() + 1;  // magic+ver+len+dict+\n
    size_t pad = (64 - (total % 64)) % 64;
    dict.append(pad, ' ');
    dict.push_back('\n');
    std::vector<uint8_t> out;
    out.reserve(10 + 2 + dict.size() + data_bytes);
    const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    out.insert(out.end(), magic, magic + 6);
    out.push_back(1);  // version 1.0
    out.push_back(0);
    uint16_t hlen = static_cast<uint16_t>(dict.size());
    out.push_back(static_cast<uint8_t>(hlen & 0xFF));
    out.push_back(static_cast<uint8_t>(hlen >> 8));
    out.insert(out.end(), dict.begin(), dict.end());
    const uint8_t* d = static_cast<const uint8_t*>(data);
    out.insert(out.end(), d, d + data_bytes);
    return out;
}

// Append one stored ZIP entry (local header + data) to `zip` and
// remember what the central directory will need.
struct CentralRecord {
    std::string name;
    uint32_t crc = 0;
    uint32_t size = 0;
    uint32_t local_offset = 0;
};

void put16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(static_cast<uint8_t>(x & 0xFF));
    v.push_back(static_cast<uint8_t>(x >> 8));
}
void put32(std::vector<uint8_t>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i)
        v.push_back(static_cast<uint8_t>((x >> (8 * i)) & 0xFF));
}

bool append_stored_entry(std::vector<uint8_t>& zip, const char* name,
                         const std::vector<uint8_t>& payload,
                         std::vector<CentralRecord>& centrals) {
    CentralRecord rec;
    rec.name = name;
    rec.size = static_cast<uint32_t>(payload.size());
    rec.crc = static_cast<uint32_t>(
        crc32(crc32(0L, Z_NULL, 0), payload.data(),
              static_cast<uInt>(payload.size())));
    rec.local_offset = static_cast<uint32_t>(zip.size());
    put32(zip, 0x04034b50u);           // local file header
    put16(zip, 20);                    // version needed
    put16(zip, 0);                     // flags
    put16(zip, 0);                     // method 0 = stored
    put16(zip, 0);                     // mod time
    put16(zip, 0x21);                  // mod date (1980-01-01)
    put32(zip, rec.crc);
    put32(zip, rec.size);
    put32(zip, rec.size);
    put16(zip, static_cast<uint16_t>(rec.name.size()));
    put16(zip, 0);                     // extra length
    zip.insert(zip.end(), rec.name.begin(), rec.name.end());
    zip.insert(zip.end(), payload.begin(), payload.end());
    centrals.push_back(rec);
    return true;
}

// Serialise a complete .npz (verts then faces, numpy's key order
// for savez with these kwargs) into one byte buffer.
std::vector<uint8_t> make_npz(const std::vector<float>& verts,
                              int64_t vrows,
                              const std::vector<int64_t>& faces,
                              int64_t frows) {
    std::vector<uint8_t> zip;
    std::vector<CentralRecord> centrals;
    append_stored_entry(zip, "verts.npy",
                        make_npy("<f4", vrows, verts.data(),
                                 verts.size() * sizeof(float)),
                        centrals);
    append_stored_entry(zip, "faces.npy",
                        make_npy("<i8", frows, faces.data(),
                                 faces.size() * sizeof(int64_t)),
                        centrals);
    size_t cd_at = zip.size();
    for (const CentralRecord& rec : centrals) {
        put32(zip, 0x02014b50u);       // central directory entry
        put16(zip, 20);                // version made by
        put16(zip, 20);                // version needed
        put16(zip, 0);                 // flags
        put16(zip, 0);                 // method stored
        put16(zip, 0);
        put16(zip, 0x21);
        put32(zip, rec.crc);
        put32(zip, rec.size);
        put32(zip, rec.size);
        put16(zip, static_cast<uint16_t>(rec.name.size()));
        put16(zip, 0); put16(zip, 0); put16(zip, 0);  // extra/comment/disk
        put16(zip, 0);                 // internal attrs
        put32(zip, 0);                 // external attrs
        put32(zip, rec.local_offset);
        zip.insert(zip.end(), rec.name.begin(), rec.name.end());
    }
    // Capture the central directory's size BEFORE writing the
    // EOCD record (writing it grows zip and would corrupt the
    // value -- this exact bug was caught by numpy's BadZipFile).
    uint32_t cd_size = static_cast<uint32_t>(zip.size() - cd_at);
    put32(zip, 0x06054b50u);           // end of central directory
    put16(zip, 0); put16(zip, 0);
    put16(zip, static_cast<uint16_t>(centrals.size()));
    put16(zip, static_cast<uint16_t>(centrals.size()));
    put32(zip, cd_size);
    put32(zip, static_cast<uint32_t>(cd_at));
    put16(zip, 0);                     // comment length
    return zip;
}

// Write bytes to `path` atomically-enough for the seam: write to
// path + ".tmp" first, rename over path only when complete. On
// any failure the temp file is removed and false returned; the
// caller then exits nonzero having left no (partial) output.
bool write_file_atomic(const std::string& path,
                       const std::vector<uint8_t>& bytes) {
    std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    bool ok = bytes.empty() ||
              std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    if (std::fclose(f) != 0) ok = false;
    if (!ok) { std::remove(tmp.c_str()); return false; }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

}  // namespace

// Parse a threshold string as a C float, matching the
// incumbent's binding (Python float -> C++ float, Phase 0
// Item 3). strtof also takes the "inf"/"nan" spellings
// repr(float) can emit; NaN / non-positive thresholds are silent
// no-fills downstream, exactly as the incumbent behaves
// (Phase 0 Item 7). False on a string strtof cannot consume
// whole.
bool parse_threshold(const char* text, float& out) {
    char* end = nullptr;
    out = std::strtof(text, &end);
    return !(end == text || *end != '\0');
}

// Run ONE seam item: read in_path, fill it, write out_path.
// This is the body main() has executed since the worker's first
// version, extracted unchanged for r35.7 so batch mode runs the
// identical code per item. Returns the seam's exit codes (0
// success ONLY; 1 any fill/IO failure, with the R1-FAIL stage
// line printed). On success, when appended_faces_out is given,
// it receives (output face rows - input face rows) -- batch
// mode's progress accounting; the single-shot seam ignores it.
//
// Attempt-5 instrumentation: every failure print below carries
// an "R1-FAIL stage=<name>:" tag, so the worker's captured
// stderr names the failing stage (the calibration harness
// records stderr_tail; before this, several exits printed a
// reason but nothing stage-shaped, and the HIP path's error
// text only travelled back through result.error).
int run_one(const std::string& in_path, const std::string& out_path,
            float max_hole_perimeter, int64_t* appended_faces_out) {

    std::vector<uint8_t> zip;
    if (!read_file(in_path, zip)) {
        std::fprintf(stderr, "R1-FAIL stage=read-input: cannot read "
                             "input: %s\n", in_path.c_str());
        return 1;
    }
    std::vector<float> verts;
    std::vector<int64_t> faces;
    int64_t nv = 0, nf = 0;
    if (!load_array(zip, "verts.npy", "<f4", verts, nv) ||
        !load_array(zip, "faces.npy", "<i8", faces, nf)) {
        std::fprintf(stderr, "R1-FAIL stage=parse-npz: input .npz "
                             "does not match the seam "
                             "(verts <f4 (K,3), faces <i8 (M,3))\n");
        return 1;
    }

#ifdef R1_USE_HIP
    // HIP path (fill_hip.hip.cpp): mirrors fill_logic on device.
    r1fill::FillResult result;
    result.ok = r1_hip_fill(verts.data(), nv, faces.data(), nf,
                            max_hole_perimeter, result.appended_verts,
                            result.appended_faces, result.error) == 0;
#else
    r1fill::FillResult result = r1fill::compute_fill(
        nv > 0 ? verts.data() : nullptr, nv,
        nf > 0 ? faces.data() : nullptr, nf, max_hole_perimeter);
#endif
    if (!result.ok) {
        // result.error names the failure (the brain's FillResult
        // .error, or the HIP path's stage/hip string); the HIP
        // path has ALSO already printed it via its fail() helper.
        std::fprintf(stderr, "R1-FAIL stage=fill: %s\n",
                     result.error.c_str());
        return 1;  // no output file written -- exit discipline
    }

    std::vector<float> out_verts;
    std::vector<int64_t> out_faces;
    if (!r1fill::assemble_output(nv > 0 ? verts.data() : nullptr, nv,
                                 nf > 0 ? faces.data() : nullptr, nf,
                                 result, out_verts, out_faces)) {
        std::fprintf(stderr, "R1-FAIL stage=assemble: output "
                             "assembly failed\n");
        return 1;
    }
    // Self-check before writing (spec section 4.3's discipline,
    // applied host-side): rows out must cover rows in, prefix
    // intact -- fail loud instead of emitting a wounded array.
    if (static_cast<int64_t>(out_verts.size()) / 3 < nv ||
        static_cast<int64_t>(out_faces.size()) / 3 < nf ||
        (nv > 0 && std::memcmp(out_verts.data(), verts.data(),
                               verts.size() * sizeof(float)) != 0) ||
        (nf > 0 && std::memcmp(out_faces.data(), faces.data(),
                               faces.size() * sizeof(int64_t)) != 0)) {
        std::fprintf(stderr, "R1-FAIL stage=self-check: output "
                             "self-check failed\n");
        return 1;
    }

    std::vector<uint8_t> out_zip =
        make_npz(out_verts, static_cast<int64_t>(out_verts.size()) / 3,
                 out_faces, static_cast<int64_t>(out_faces.size()) / 3);
    if (!write_file_atomic(out_path, out_zip)) {
        std::fprintf(stderr, "R1-FAIL stage=write-output: cannot "
                             "write output: %s\n",
                     out_path.c_str());
        return 1;
    }
    if (appended_faces_out)
        *appended_faces_out =
            static_cast<int64_t>(out_faces.size()) / 3 - nf;
    return 0;
}

// r35.7 batch mode: fill every item in the manifest in THIS
// process (see the file header for the contract). Progress
// lines go to stdout and are flushed per line so the parent's
// reader sees chunks land as they happen; failure diagnostics
// stay on stderr via run_one's R1-FAIL prints. The first failed
// item ends the process with exit 10 -- the parent re-batches
// the remainder into a fresh process (fresh HIP context) and
// ladders the failed item single-shot. Manifest format errors
// exit 2 before any item runs past the bad line.
int run_batch(const std::string& manifest_path) {
    std::vector<uint8_t> raw;
    if (!read_file(manifest_path, raw)) {
        std::fprintf(stderr, "R1-FAIL stage=batch-manifest: "
                             "cannot read manifest: %s\n",
                     manifest_path.c_str());
        return 2;
    }
    std::string text(raw.begin(), raw.end());
    size_t pos = 0;
    size_t item = 0;
    while (pos <= text.size()) {
        size_t nl = text.find('\n', pos);
        std::string line = text.substr(
            pos, nl == std::string::npos ? std::string::npos
                                         : nl - pos);
        pos = (nl == std::string::npos) ? text.size() + 1 : nl + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        size_t t1 = line.find('\t');
        size_t t2 = t1 == std::string::npos
                        ? std::string::npos
                        : line.find('\t', t1 + 1);
        if (t1 == std::string::npos || t2 == std::string::npos) {
            std::fprintf(stderr, "R1-FAIL stage=batch-manifest: "
                                 "line %zu is not three "
                                 "TAB-separated fields\n",
                         item);
            return 2;
        }
        float threshold = 0.0f;
        if (!parse_threshold(line.substr(t2 + 1).c_str(),
                             threshold)) {
            std::fprintf(stderr, "R1-FAIL stage=batch-manifest: "
                                 "bad max_hole_perimeter on "
                                 "line %zu: %s\n",
                         item, line.substr(t2 + 1).c_str());
            return 2;
        }
        std::printf("R1-BATCH item %zu start\n", item);
        std::fflush(stdout);
        std::chrono::steady_clock::time_point t0 =
            std::chrono::steady_clock::now();
        int64_t appended = 0;
        int rc = run_one(line.substr(0, t1),
                         line.substr(t1 + 1, t2 - t1 - 1),
                         threshold, &appended);
        if (rc != 0) {
            std::printf("R1-BATCH item %zu fail\n", item);
            std::fflush(stdout);
            return 10;
        }
        double elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
        std::printf("R1-BATCH item %zu ok elapsed=%.3f "
                    "appended_faces=%lld\n",
                    item, elapsed, (long long)appended);
        std::fflush(stdout);
        ++item;
    }
    std::printf("R1-BATCH done items=%zu\n", item);
    std::fflush(stdout);
    return 0;
}

int main(int argc, char** argv) {
    // r35.7 batch mode: [worker, --batch, manifest_path].
    if (argc == 3 && std::strcmp(argv[1], "--batch") == 0)
        return run_batch(argv[2]);
    // Seam argv: [worker, in_path, out_path, max_hole_perimeter].
    if (argc != 4) {
        std::fprintf(stderr,
                     "R1-FAIL stage=argv: usage: %s in.npz out.npz "
                     "max_hole_perimeter | %s --batch "
                     "manifest.txt\n",
                     argc > 0 ? argv[0] : "fill_worker",
                     argc > 0 ? argv[0] : "fill_worker");
        return 2;
    }
    float max_hole_perimeter = 0.0f;
    if (!parse_threshold(argv[3], max_hole_perimeter)) {
        std::fprintf(stderr, "R1-FAIL stage=argv: bad "
                             "max_hole_perimeter: %s\n", argv[3]);
        return 2;
    }
    return run_one(argv[1], argv[2], max_hole_perimeter, nullptr);
}
