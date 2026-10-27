// =====================================================================
// fill_hip.hip.cpp -- R1 hole fill, HIP device path.
//
//   *** DRAFT -- UNCOMPILED, UNPROVEN until built on the target rig ***
//
//   This file has never been compiled: the build VM has no ROCm
//   toolchain and no gfx1201 device. It is a careful draft written
//   against ROCm 7.2.x / hipcc / --offload-arch=gfx1201 (see
//   BUILD-ON-PC.md for the exact build + probe steps). Nothing may
//   trust it until it has been built on Cory's rig and passed the
//   harness there. The CPU path (fill_logic.cpp) is the proven
//   reference for every behaviour this file mirrors.
//
// DESIGN (spec section 4, Phase 0 findings):
//   * Boundary compaction first: the face array is touched once by
//     an edge-census kernel; everything after works on the
//     compacted boundary set only.
//   * Wave32-first: block size 256 (8 RDNA wavefronts of 32); no
//     kernel uses warp shuffles or any hardcoded warp width, so
//     nothing assumes wave32 OR wave64 -- the compiler specialises
//     for gfx1201.
//   * The graph stage (manifold-boundary components + loop test)
//     runs on the HOST through the shared brain, fill_logic: spec
//     section 4.2's determinism bias -- tracing is done where it
//     is simplest to make deterministic, and the parent already
//     does the same union-find on the host. The device recomputes
//     the per-loop MATH (perimeter, centroid, fans) from the
//     host-supplied loop descriptors, and a host/device perimeter
//     disagreement fails loud rather than emitting a fill the
//     brain did not approve.
//   * ALL host<->device transfers go through ONE helper,
//     chunked_transfer(), capped at 65,536 rows per hipMemcpy --
//     the 2^20-row wound of spec section 3.1.4 is avoided by
//     construction, not by an element-size coincidence. This file
//     contains exactly one hipMemcpy call site: inside that helper.
//   * Explicit 64-bit counts everywhere; grids are computed in
//     64-bit and range-checked before launch; counts outside the
//     designed envelope fail loud (worker error -> nonzero exit ->
//     the parent's ladder), never truncate, never wrap.
//   * NO grid-0 launch is possible: every kernel launch goes
//     through launch1d(), which refuses count <= 0, and every call
//     site is additionally guarded by an explicit count > 0 test.
//     The incumbent's code-9 shape (B > 0 boundary edges, zero
//     manifold pairs) therefore cannot construct an invalid launch
//     here: the pair/hook step does not exist as a launch at all
//     (it is host union-find), and such a chunk yields NO FILL by
//     construction (Phase 0 Item 8).
// =====================================================================

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "fill_hip.h"
#include "fill_logic.h"

namespace {

// ---------------------------------------------------------------------
// Constants. kMaxTransferRows is the chunked-transfer cap (spec
// section 4.3: at most 65,536 rows per copy, the shape of the
// parent's own chunked_readout). kBlock is the launch block size.
// kEnvelope* are the designed-envelope bounds: chunk budgets are
// 131,072 faces / 262,144 verts (spec section 4.2) and the stock
// path's are 524,288 (spec section 2.6); the envelope sits far
// above both, and anything past it is a bug to fail on, not an
// input to truncate for.
// ---------------------------------------------------------------------
constexpr int64_t kMaxTransferRows = 65536;
constexpr int kBlock = 256;
constexpr int64_t kEnvelopeMaxVerts = int64_t{1} << 26;
constexpr int64_t kEnvelopeMaxFaces = int64_t{1} << 26;

// Fail helper: record the message, print it to stderr (attempt 5
// instrumentation: the message must be visible in the worker's
// captured stderr at the point of failure, not only returned
// through `err` for the caller to print), return a nonzero code.
int fail(std::string& err, const std::string& msg) {
    err = msg;
    std::fprintf(stderr, "R1-FAIL %s\n", msg.c_str());
    std::fflush(stderr);
    return 1;
}

// Every HIP API call is checked; an error is a worker failure
// (nonzero exit -> ladder), never a half-written output. The
// message names the pipeline stage (the `stage` local in
// r1_hip_fill, updated at each stage boundary) and the HIP error
// string, per the attempt-5 instrumentation format:
//   R1-FAIL stage=<name> hip=<hipGetErrorString> call=<expr>
#define R1_HIP_CHECK(call)                                              \
    do {                                                                \
        hipError_t rc_ = (call);                                        \
        if (rc_ != hipSuccess) {                                        \
            return fail(error, std::string("stage=") + stage +          \
                                   " hip=" + hipGetErrorString(rc_) +   \
                                   " call=" #call);                     \
        }                                                               \
    } while (0)

// RAII device buffer: frees on every exit path so a failed attempt
// leaves no device state behind (the worker is safe to kill and
// safe to re-run; spec section 2.3).
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    explicit DeviceBuffer(int64_t count) {
        if (count > 0)
            (void)hipMalloc(&ptr, static_cast<size_t>(count) * sizeof(T));
    }
    ~DeviceBuffer() {
        if (ptr) (void)hipFree(ptr);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// ---------------------------------------------------------------------
// THE chunked-transfer helper -- the only door between host and
// device in this file (spec section 4.3). Copies `rows` rows of
// `row_bytes` each, in blocks of at most kMaxTransferRows rows.
// Counts are 64-bit end to end; a row count of 0 is a no-op.
// ---------------------------------------------------------------------
hipError_t chunked_transfer(void* dst, const void* src, int64_t rows,
                            size_t row_bytes, hipMemcpyKind kind) {
    int64_t done = 0;
    while (done < rows) {
        int64_t block = rows - done;
        if (block > kMaxTransferRows) block = kMaxTransferRows;
        hipError_t rc = hipMemcpy(
            static_cast<char*>(dst) + done * static_cast<int64_t>(row_bytes),
            static_cast<const char*>(src) +
                done * static_cast<int64_t>(row_bytes),
            static_cast<size_t>(block) * row_bytes, kind);
        if (rc != hipSuccess) return rc;
        done += block;
    }
    return hipSuccess;
}

// ---------------------------------------------------------------------
// launch1d -- the only kernel-launch shape in this file. Refuses
// count <= 0 (a grid-0 launch is a bug here, not an input case:
// call sites guard with count > 0 and this is the loud backstop),
// computes the grid in 64-bit, and range-checks it against the
// device grid limit before launching.
// ---------------------------------------------------------------------
template <typename Kernel, typename... Args>
hipError_t launch1d(Kernel kernel, int64_t count, Args... args) {
    if (count <= 0) return hipErrorInvalidConfiguration;
    uint64_t grid = (static_cast<uint64_t>(count) + kBlock - 1) / kBlock;
    if (grid > 0x7FFFFFFFull) return hipErrorInvalidConfiguration;
    kernel<<<static_cast<unsigned int>(grid), kBlock, 0, 0>>>(args...);
    return hipGetLastError();
}

// ---------------------------------------------------------------------
// Device kernels. All are per-element over a compacted array; none
// reads the full face array except expand_edges_kernel (the census
// touch, once).
// ---------------------------------------------------------------------

// Per face, emit its 3 undirected edges as packed (min<<32)|max
// keys -- direction in the incident face is discarded here and
// never recovered (Phase 0 Item 2, stage 1).
__global__ void expand_edges_kernel(const int64_t* faces,
                                    int64_t num_faces,
                                    uint64_t* keys_out) {
    int64_t f = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;
    uint32_t v[3];
    for (int k = 0; k < 3; ++k)
        v[k] = static_cast<uint32_t>(faces[f * 3 + k]);
    for (int k = 0; k < 3; ++k) {
        uint32_t a = v[k], b = v[(k + 1) % 3];
        uint32_t lo = a < b ? a : b;
        uint32_t hi = a < b ? b : a;
        keys_out[f * 3 + k] =
            (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
    }
}

// flag[i] = 1 iff key[i] starts a new run in the sorted key array.
__global__ void run_flags_kernel(const uint64_t* sorted_keys, int64_t n,
                                 int32_t* flags_out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags_out[i] = (i == 0 || sorted_keys[i] != sorted_keys[i - 1]) ? 1 : 0;
}

// At each run start, scatter the unique edge key to its scan
// position. Output order is ascending key order because run
// starts are scattered to their scan positions. Counts are NOT
// written here (a run's length is not locally visible); they are
// computed by count_runs_kernel below from run_index alone.
__global__ void scatter_unique_kernel(const uint64_t* sorted_keys,
                                      const int32_t* flags,
                                      const int32_t* run_index,
                                      int64_t n,
                                      uint64_t* uniq_keys_out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n || flags[i] == 0) return;
    uniq_keys_out[run_index[i]] = sorted_keys[i];
}

// At each run start, scatter the position to its run index:
// starts[r] = the sorted-key position where run r begins. Same
// scatter shape as scatter_unique_kernel (flags + exclusive
// scan); at a run start the exclusive scan value IS the run
// index, so the scatter is exact.
__global__ void scatter_run_starts_kernel(const int32_t* flags,
                                          const int32_t* run_index,
                                          int64_t n,
                                          int32_t* starts_out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n || flags[i] == 0) return;
    starts_out[run_index[i]] = static_cast<int32_t>(i);
}

// Per unique edge, its incident-face count: the distance from
// its run start to the next run start (or to n for the last
// run). ATTEMPT 6 FIX: the previous version of this kernel
// binary-searched the exclusive scan as if scan[i] were the
// run index of position i. It is not: at a run's LATER
// positions the exclusive scan reads r+1, not r, so
// lower_bound(scan, r) lands one past run r-1's start instead
// of at run r's start, and every count came out shifted by
// one run (count[0] forced to 1; the last run's count never
// read). The boundary total then differed from the host
// census by exactly +1 whenever the largest unique edge was
// not itself a boundary edge -- job 043's named
// "device/host boundary count disagreement" (probe-1 chunks
// 1/4/11/12/24, both runs, deterministic; a host replica of
// the device algorithm as written reproduces the +1 delta
// on exactly those chunks and equality on the other 20).
// Differencing scattered run starts is exact by
// construction: starts are positions, counts are
// consecutive differences, and the counts sum to n.
__global__ void count_runs_kernel(const int32_t* starts, int64_t n,
                                  int32_t num_runs,
                                  int32_t* uniq_counts_out) {
    int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= num_runs) return;
    int64_t end = (r + 1 < num_runs)
                      ? static_cast<int64_t>(starts[r + 1])
                      : n;
    uniq_counts_out[r] =
        static_cast<int32_t>(end - static_cast<int64_t>(starts[r]));
}

// flag[i] = 1 iff unique edge i is a boundary edge (count == 1).
__global__ void boundary_flags_kernel(const int32_t* uniq_counts,
                                      int64_t num_unique,
                                      int32_t* flags_out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_unique) return;
    flags_out[i] = (uniq_counts[i] == 1) ? 1 : 0;
}

// Scatter boundary edge keys to their scan positions: the
// compacted boundary list, ascending by (min, max) -- the order
// every later stage (and fill_logic) assumes.
__global__ void scatter_boundary_kernel(const uint64_t* uniq_keys,
                                        const int32_t* flags,
                                        const int32_t* scan,
                                        int64_t num_unique,
                                        uint64_t* boundary_keys_out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_unique || flags[i] == 0) return;
    boundary_keys_out[scan[i]] = uniq_keys[i];
}

// Per qualifying loop (one thread per loop): the float32 perimeter
// as a sequential sum of edge lengths in loop_boundaries order --
// the same pinned order fill_logic uses, so the device mask must
// agree with the host brain's decision exactly; the host checks
// that agreement and fails loud on any divergence.
__global__ void loop_perimeters_kernel(const float* verts,
                                       const uint32_t* loop_edge_lo,
                                       const uint32_t* loop_edge_hi,
                                       const int64_t* loop_offsets,
                                       int64_t num_loops,
                                       float max_hole_perimeter,
                                       float* perimeters_out,
                                       int32_t* mask_out) {
    int64_t loop = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (loop >= num_loops) return;  // grid rounds up past num_loops
    float perimeter = 0.0f;
    for (int64_t e = loop_offsets[loop]; e < loop_offsets[loop + 1]; ++e) {
        const float* a = verts + static_cast<int64_t>(loop_edge_lo[e]) * 3;
        const float* b = verts + static_cast<int64_t>(loop_edge_hi[e]) * 3;
        float dx = b[0] - a[0];
        float dy = b[1] - a[1];
        float dz = b[2] - a[2];
        perimeter += sqrtf(dx * dx + dy * dy + dz * dz);
    }
    perimeters_out[loop] = perimeter;
    mask_out[loop] = (perimeter < max_hole_perimeter) ? 1 : 0;
}

// Per qualifying loop (one thread per loop): the centroid as the
// float32 mean of the loop's EDGE MIDPOINTS, summed sequentially
// in loop_boundaries order, divided by float(edge count) -- the
// fork incumbent's pinned arithmetic (Phase 0 Item 4).
__global__ void loop_centroids_kernel(const float* verts,
                                      const uint32_t* loop_edge_lo,
                                      const uint32_t* loop_edge_hi,
                                      const int64_t* loop_offsets,
                                      int64_t num_loops,
                                      float* centroids_out) {
    int64_t loop = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (loop >= num_loops) return;  // grid rounds up past num_loops
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    for (int64_t e = loop_offsets[loop]; e < loop_offsets[loop + 1]; ++e) {
        const float* a = verts + static_cast<int64_t>(loop_edge_lo[e]) * 3;
        const float* b = verts + static_cast<int64_t>(loop_edge_hi[e]) * 3;
        sx += (a[0] + b[0]) * 0.5f;
        sy += (a[1] + b[1]) * 0.5f;
        sz += (a[2] + b[2]) * 0.5f;
    }
    float n = static_cast<float>(loop_offsets[loop + 1] - loop_offsets[loop]);
    centroids_out[loop * 3 + 0] = sx / n;
    centroids_out[loop * 3 + 1] = sy / n;
    centroids_out[loop * 3 + 2] = sz / n;
}

// Per qualifying loop edge (one thread per edge): the fan face
// (max endpoint, min endpoint, centroid) -- pure index order
// (Phase 0 Item 5). centroid_global = num_verts + loop id.
__global__ void emit_fans_kernel(const uint32_t* loop_edge_lo,
                                 const uint32_t* loop_edge_hi,
                                 const int32_t* edge_loop,
                                 int64_t num_edges, int64_t num_verts,
                                 int64_t* fans_out) {
    int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;  // grid rounds up past num_edges
    fans_out[e * 3 + 0] = static_cast<int64_t>(loop_edge_hi[e]);  // max
    fans_out[e * 3 + 1] = static_cast<int64_t>(loop_edge_lo[e]);  // min
    fans_out[e * 3 + 2] = num_verts + edge_loop[e];               // centroid
}

}  // namespace

// ---------------------------------------------------------------------
// r1_hip_fill -- see fill_hip.h for the contract. Pipeline:
//   upload (chunked) -> edge census + radix sort (hipcub) ->
//   run-length encode -> boundary compaction -> boundary keys back
//   to host (chunked) -> host brain (fill_logic) for components /
//   loop test / qualification -> loop descriptors up (chunked) ->
//   device perimeter (cross-checked) + centroids + fans -> results
//   back (chunked). Every launch is count-guarded; every count is
//   64-bit; every failure returns nonzero with `error` set.
// ---------------------------------------------------------------------
int r1_hip_fill(const float* verts, int64_t num_verts,
                const int64_t* faces, int64_t num_faces,
                float max_hole_perimeter,
                std::vector<float>& appended_verts,
                std::vector<int64_t>& appended_faces,
                std::string& error) {
    appended_verts.clear();
    appended_faces.clear();
    // Attempt-5 instrumentation: the current pipeline stage, named
    // in every R1_HIP_CHECK failure message (see the macro above).
    const char* stage = "entry";
    if (num_verts < 0 || num_faces < 0 ||
        num_verts > kEnvelopeMaxVerts || num_faces > kEnvelopeMaxFaces)
        return fail(error, "counts outside the designed envelope");
    if (num_faces == 0) return 0;  // no faces: no boundary, no fill

    // Host brain first for index validation (the one deliberate
    // difference from the incumbent; Phase 0 Item 7): the full
    // plan is computed now and its STRUCTURE (which loops qualify,
    // their edge lists -- recoverable from its fans, which are
    // loop-major in loop_boundaries order) drives the device math
    // below. If the device census disagrees with the plan's
    // boundary count, we fail loud rather than improvise.
    r1fill::FillResult plan = r1fill::compute_fill(
        verts, num_verts, faces, num_faces, max_hole_perimeter);
    if (!plan.ok) return fail(error, plan.error);

    // Zero-append short-circuit (attempt 3, extended by
    // attempt 4): a chunk whose host plan appends NOTHING has
    // no fill, and returns the brain's proven answer here,
    // before ANY device call. Attempt 3 (job 040) proved the
    // device census chain below mishandles the zero-boundary
    // shape: closed_cube and duplicate_faces (both B == 0)
    // exited 1 while every B > 0 fixture passed, so B == 0
    // was short-circuited. Job 041 then proved the bad family
    // is broader than B == 0: the battery passed 5/5, but
    // calibration probe 2's crafted code-9 shape (B = 4,
    // components = 4, loops = 0, qualifying = 0 -- boundary
    // exists, but every boundary vertex is non-manifold, so
    // no loop ever forms) ALSO exited 1 in the census chain,
    // where the spec requires a clean no-fill exit 0. The
    // shared trait of every proven-bad shape is not a zero in
    // any single census count -- it is a plan that appends
    // nothing (num_qualifying_loops == 0, which subsumes
    // B == 0, the loops == 0 route, and the perimeter route
    // where loops exist but none is under threshold). Such a
    // chunk's correct output is the input, unchanged; the
    // device has nothing to compute and nothing to check
    // (the cross-checks below exist to validate a fill the
    // device is about to produce). The wrapper returns the
    // brain's proven answer by construction (appended blocks
    // were cleared on entry; the caller assembles the input
    // through unchanged). Below this point the plan appends
    // at least one fan: num_qualifying >= 1, hence B >= 3
    // (a qualifying loop has at least 3 boundary edges),
    // num_keys >= 3, num_unique >= 1 (the num_unique <= 0
    // guard below stays as the loud backstop), num_qedges
    // >= 3 -- every hipcub call, launch, allocation and
    // readback receives a count >= 1, and the loop stage's
    // own num_qualifying == 0 exit remains as the backstop.
    if (plan.num_qualifying_loops == 0) return 0;  // no fill: no device

    // ---- Upload (through the chunked helper only) ------------------
    stage = "upload";
    DeviceBuffer<float> d_verts(num_verts * 3);
    DeviceBuffer<int64_t> d_faces(num_faces * 3);
    if (!d_verts.ptr || !d_faces.ptr)
        return fail(error, "hipMalloc failed (verts/faces)");
    R1_HIP_CHECK(chunked_transfer(d_verts.ptr, verts, num_verts,
                                  3 * sizeof(float),
                                  hipMemcpyHostToDevice));
    R1_HIP_CHECK(chunked_transfer(d_faces.ptr, faces, num_faces,
                                  3 * sizeof(int64_t),
                                  hipMemcpyHostToDevice));

    // ---- Edge census: expand, sort, run-length encode --------------
    stage = "edge-census";
    const int64_t num_keys = num_faces * 3;
    DeviceBuffer<uint64_t> d_keys(num_keys);
    DeviceBuffer<uint64_t> d_keys_sorted(num_keys);
    DeviceBuffer<int32_t> d_flags(num_keys);
    DeviceBuffer<int32_t> d_scan(num_keys);
    if (!d_keys.ptr || !d_keys_sorted.ptr || !d_flags.ptr || !d_scan.ptr)
        return fail(error, "hipMalloc failed (edge census)");
    if (num_faces > 0) {  // guarded launch: count > 0
        R1_HIP_CHECK(launch1d(expand_edges_kernel, num_faces,
                              d_faces.ptr, num_faces, d_keys.ptr));
    }
    // hipcub radix sort of the packed keys (ascending (min, max)).
    {
        size_t temp_bytes = 0;
        R1_HIP_CHECK(hipcub::DeviceRadixSort::SortKeys(
            nullptr, temp_bytes, d_keys.ptr, d_keys_sorted.ptr,
            static_cast<int>(num_keys)));
        DeviceBuffer<uint8_t> d_temp(static_cast<int64_t>(temp_bytes));
        if (temp_bytes > 0 && !d_temp.ptr)
            return fail(error, "hipMalloc failed (sort scratch)");
        R1_HIP_CHECK(hipcub::DeviceRadixSort::SortKeys(
            d_temp.ptr, temp_bytes, d_keys.ptr, d_keys_sorted.ptr,
            static_cast<int>(num_keys)));
    }
    if (num_keys > 0) {
        R1_HIP_CHECK(launch1d(run_flags_kernel, num_keys,
                              d_keys_sorted.ptr, num_keys, d_flags.ptr));
    }
    {
        size_t temp_bytes = 0;
        R1_HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(
            nullptr, temp_bytes, d_flags.ptr, d_scan.ptr,
            static_cast<int>(num_keys)));
        DeviceBuffer<uint8_t> d_temp(static_cast<int64_t>(temp_bytes));
        if (temp_bytes > 0 && !d_temp.ptr)
            return fail(error, "hipMalloc failed (scan scratch)");
        R1_HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(
            d_temp.ptr, temp_bytes, d_flags.ptr, d_scan.ptr,
            static_cast<int>(num_keys)));
    }
    // Total unique edges = scan[last] + flag[last], read back
    // through the chunked helper (single rows).
    int32_t scan_last = 0, flag_last = 0;
    R1_HIP_CHECK(chunked_transfer(&scan_last, d_scan.ptr + (num_keys - 1),
                                  1, sizeof(int32_t),
                                  hipMemcpyDeviceToHost));
    R1_HIP_CHECK(chunked_transfer(&flag_last, d_flags.ptr + (num_keys - 1),
                                  1, sizeof(int32_t),
                                  hipMemcpyDeviceToHost));
    const int64_t num_unique = static_cast<int64_t>(scan_last) + flag_last;
    if (num_unique <= 0)
        return fail(error, "edge census produced no unique edges");

    DeviceBuffer<uint64_t> d_uniq_keys(num_unique);
    DeviceBuffer<int32_t> d_uniq_counts(num_unique);
    DeviceBuffer<int32_t> d_starts(num_unique);
    if (!d_uniq_keys.ptr || !d_uniq_counts.ptr || !d_starts.ptr)
        return fail(error, "hipMalloc failed (unique edges)");
    if (num_keys > 0) {
        R1_HIP_CHECK(launch1d(scatter_unique_kernel, num_keys,
                              d_keys_sorted.ptr, d_flags.ptr, d_scan.ptr,
                              num_keys, d_uniq_keys.ptr));
        R1_HIP_CHECK(launch1d(scatter_run_starts_kernel, num_keys,
                              d_flags.ptr, d_scan.ptr, num_keys,
                              d_starts.ptr));
        R1_HIP_CHECK(launch1d(count_runs_kernel, num_unique,
                              d_starts.ptr, num_keys,
                              static_cast<int32_t>(num_unique),
                              d_uniq_counts.ptr));
    }

    // ---- Boundary compaction ---------------------------------------
    stage = "boundary-compaction";
    DeviceBuffer<int32_t> d_bflags(num_unique);
    DeviceBuffer<int32_t> d_bscan(num_unique);
    if (!d_bflags.ptr || !d_bscan.ptr)
        return fail(error, "hipMalloc failed (boundary flags)");
    if (num_unique > 0) {
        R1_HIP_CHECK(launch1d(boundary_flags_kernel, num_unique,
                              d_uniq_counts.ptr, num_unique,
                              d_bflags.ptr));
    }
    {
        size_t temp_bytes = 0;
        R1_HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(
            nullptr, temp_bytes, d_bflags.ptr, d_bscan.ptr,
            static_cast<int>(num_unique)));
        DeviceBuffer<uint8_t> d_temp(static_cast<int64_t>(temp_bytes));
        if (temp_bytes > 0 && !d_temp.ptr)
            return fail(error, "hipMalloc failed (scan scratch)");
        R1_HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(
            d_temp.ptr, temp_bytes, d_bflags.ptr, d_bscan.ptr,
            static_cast<int>(num_unique)));
    }
    int32_t bscan_last = 0, bflag_last = 0;
    R1_HIP_CHECK(chunked_transfer(&bscan_last, d_bscan.ptr + (num_unique - 1),
                                  1, sizeof(int32_t),
                                  hipMemcpyDeviceToHost));
    R1_HIP_CHECK(chunked_transfer(&bflag_last, d_bflags.ptr + (num_unique - 1),
                                  1, sizeof(int32_t),
                                  hipMemcpyDeviceToHost));
    const int64_t num_boundary = static_cast<int64_t>(bscan_last) + bflag_last;

    // Cross-check the device census against the host brain. A
    // disagreement is a build/stack bug: fail loud, fill nothing.
    if (num_boundary != plan.num_boundary_edges)
        return fail(error, "device/host boundary count disagreement");

    if (num_boundary > 0) {  // guarded launch + guarded readback
        DeviceBuffer<uint64_t> d_boundary(num_boundary);
        if (!d_boundary.ptr)
            return fail(error, "hipMalloc failed (boundary keys)");
        R1_HIP_CHECK(launch1d(scatter_boundary_kernel, num_unique,
                              d_uniq_keys.ptr, d_bflags.ptr,
                              d_bscan.ptr, num_unique, d_boundary.ptr));
        std::vector<uint64_t> boundary_keys(
            static_cast<size_t>(num_boundary));
        R1_HIP_CHECK(chunked_transfer(boundary_keys.data(),
                                      d_boundary.ptr, num_boundary,
                                      sizeof(uint64_t),
                                      hipMemcpyDeviceToHost));
        // The keys themselves are not re-consumed host-side in
        // this draft (the brain already holds the same list); the
        // readback exists so the compaction is proven end-to-end
        // at first build, and so a future revision can move the
        // component stage without changing the transfer shape.
        (void)boundary_keys;
    }

    // ---- Loop descriptors from the brain's plan ---------------------
    // plan.appended_faces is loop-major: consecutive fans sharing a
    // centroid index (num_verts + loop rank) form one loop, their
    // (max, min) pairs are the loop's edges in loop_boundaries
    // order. Recover flat edge arrays + offsets from it.
    const int64_t num_qualifying = plan.num_qualifying_loops;
    if (num_qualifying == 0) return 0;  // nothing qualifies: no fill
    std::vector<uint32_t> edge_lo;
    std::vector<uint32_t> edge_hi;
    std::vector<int32_t> edge_loop;
    std::vector<int64_t> loop_offsets;
    loop_offsets.push_back(0);
    {
        int64_t cur_loop = -1;
        // appended_faces is a flat (M,3) block: i steps per face.
        for (size_t i = 0; i < plan.appended_faces.size(); i += 3) {
            int64_t mx = plan.appended_faces[i];
            int64_t mn = plan.appended_faces[i + 1];
            int64_t ci = plan.appended_faces[i + 2];
            int64_t loop = ci - num_verts;
            if (loop != cur_loop) {
                if (cur_loop >= 0)
                    loop_offsets.push_back(
                        static_cast<int64_t>(edge_lo.size()));
                cur_loop = loop;
            }
            edge_hi.push_back(static_cast<uint32_t>(mx));
            edge_lo.push_back(static_cast<uint32_t>(mn));
            edge_loop.push_back(static_cast<int32_t>(loop));
        }
        loop_offsets.push_back(static_cast<int64_t>(edge_lo.size()));
    }
    const int64_t num_qedges = static_cast<int64_t>(edge_lo.size());
    if (static_cast<int64_t>(loop_offsets.size()) != num_qualifying + 1 ||
        num_qedges == 0)
        return fail(error, "loop descriptor recovery failed");

    // ---- Device math: perimeter (check), centroids, fans ------------
    stage = "loop-math";
    DeviceBuffer<uint32_t> d_edge_lo(num_qedges);
    DeviceBuffer<uint32_t> d_edge_hi(num_qedges);
    DeviceBuffer<int32_t> d_edge_loop(num_qedges);
    DeviceBuffer<int64_t> d_offsets(num_qualifying + 1);
    DeviceBuffer<float> d_perims(num_qualifying);
    DeviceBuffer<int32_t> d_mask(num_qualifying);
    DeviceBuffer<float> d_centroids(num_qualifying * 3);
    DeviceBuffer<int64_t> d_fans(num_qedges * 3);
    if (!d_edge_lo.ptr || !d_edge_hi.ptr || !d_edge_loop.ptr ||
        !d_offsets.ptr || !d_perims.ptr || !d_mask.ptr ||
        !d_centroids.ptr || !d_fans.ptr)
        return fail(error, "hipMalloc failed (loop math)");
    R1_HIP_CHECK(chunked_transfer(d_edge_lo.ptr, edge_lo.data(),
                                  num_qedges, sizeof(uint32_t),
                                  hipMemcpyHostToDevice));
    R1_HIP_CHECK(chunked_transfer(d_edge_hi.ptr, edge_hi.data(),
                                  num_qedges, sizeof(uint32_t),
                                  hipMemcpyHostToDevice));
    R1_HIP_CHECK(chunked_transfer(d_edge_loop.ptr, edge_loop.data(),
                                  num_qedges, sizeof(int32_t),
                                  hipMemcpyHostToDevice));
    R1_HIP_CHECK(chunked_transfer(d_offsets.ptr, loop_offsets.data(),
                                  num_qualifying + 1, sizeof(int64_t),
                                  hipMemcpyHostToDevice));

    if (num_qualifying > 0) {  // guarded launches: counts > 0
        R1_HIP_CHECK(launch1d(loop_perimeters_kernel, num_qualifying,
                              d_verts.ptr, d_edge_lo.ptr, d_edge_hi.ptr,
                              d_offsets.ptr, num_qualifying,
                              max_hole_perimeter,
                              d_perims.ptr, d_mask.ptr));
        R1_HIP_CHECK(launch1d(loop_centroids_kernel, num_qualifying,
                              d_verts.ptr, d_edge_lo.ptr, d_edge_hi.ptr,
                              d_offsets.ptr, num_qualifying,
                              d_centroids.ptr));
    }
    if (num_qedges > 0) {
        R1_HIP_CHECK(launch1d(emit_fans_kernel, num_qedges,
                              d_edge_lo.ptr, d_edge_hi.ptr,
                              d_edge_loop.ptr, num_qedges, num_verts,
                              d_fans.ptr));
    }

    // The device perimeter mask must confirm EVERY loop the brain
    // qualified (same sequential float32 sum, same threshold): a
    // single disagreement fails the attempt loudly.
    std::vector<int32_t> mask(static_cast<size_t>(num_qualifying));
    R1_HIP_CHECK(chunked_transfer(mask.data(), d_mask.ptr,
                                  num_qualifying, sizeof(int32_t),
                                  hipMemcpyDeviceToHost));
    for (int64_t i = 0; i < num_qualifying; ++i)
        if (mask[static_cast<size_t>(i)] != 1)
            return fail(error, "device/host perimeter disagreement");

    // ---- Readback (chunked helper only) ------------------------------
    stage = "readback";
    appended_verts.resize(static_cast<size_t>(num_qualifying) * 3);
    appended_faces.resize(static_cast<size_t>(num_qedges) * 3);
    R1_HIP_CHECK(chunked_transfer(appended_verts.data(), d_centroids.ptr,
                                  num_qualifying, 3 * sizeof(float),
                                  hipMemcpyDeviceToHost));
    R1_HIP_CHECK(chunked_transfer(appended_faces.data(), d_fans.ptr,
                                  num_qedges, 3 * sizeof(int64_t),
                                  hipMemcpyDeviceToHost));
    R1_HIP_CHECK(hipDeviceSynchronize());
    return 0;
}
