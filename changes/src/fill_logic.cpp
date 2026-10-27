// fill_logic.cpp -- implementation of fill_logic.h.
//
// Structure, mirroring Phase 0's four stages (Item 2):
//   1. Edge census     : every face emits its 3 undirected edges as
//                        packed (min<<32)|max keys; sort + run-length
//                        encode gives unique edges in ascending
//                        (min, max) order with incident-face counts.
//   2. Boundary select : edges with count exactly 1, kept in
//                        ascending edge order; their positions in
//                        this list are "boundary positions" 0..B-1.
//   3. Components      : union-find over boundary positions, united
//                        ONLY at manifold boundary vertices (see
//                        vertex_is_manifold below).
//   4. Loop test+fill  : per component, the two-sided continuation
//                        test; per loop, the float32 perimeter test,
//                        the midpoint centroid, the index-order fan.
//
// Everything is deterministic: no hash-table iteration order, no
// atomics, no data-dependent ordering reaches the output.

#include "fill_logic.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace r1fill {
namespace {

// ---------------------------------------------------------------------
// Unique undirected edge with its incident-face count.
// Endpoints are stored unpacked (lo <= hi) after the packed-key sort;
// the packing exists only to give the sort its (min, then max) order,
// exactly the incumbent's radix order (Phase 0 Item 6, step 1).
// ---------------------------------------------------------------------
struct Edge {
    uint32_t lo;
    uint32_t hi;
    int64_t count;  // incident faces; boundary iff exactly 1
};

// Pack (lo, hi) into one 64-bit key so a plain integer sort orders
// edges by lo first, then hi.
uint64_t pack_edge(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(lo) << 32) |
           static_cast<uint64_t>(hi);
}

// ---------------------------------------------------------------------
// Union-find over boundary positions with the incumbent's root rule:
// the higher root always unions into the lower (Phase 0 Item 6,
// step 3), so a component's root ends as its smallest boundary
// position and component order falls out of a plain ascending scan.
// ---------------------------------------------------------------------
struct UnionFind {
    std::vector<int64_t> parent;  // parent[i] == i iff i is a root

    explicit UnionFind(int64_t n) : parent(static_cast<size_t>(n)) {
        for (int64_t i = 0; i < n; ++i) parent[static_cast<size_t>(i)] = i;
    }

    // Root of x, with path compression (iterative; no recursion).
    int64_t find(int64_t x) {
        int64_t root = x;
        while (parent[static_cast<size_t>(root)] != root)
            root = parent[static_cast<size_t>(root)];
        while (parent[static_cast<size_t>(x)] != root) {
            int64_t next = parent[static_cast<size_t>(x)];
            parent[static_cast<size_t>(x)] = root;
            x = next;
        }
        return root;
    }

    // Union by min-root: the smaller root survives.
    void unite(int64_t a, int64_t b) {
        int64_t ra = find(a), rb = find(b);
        if (ra == rb) return;
        if (rb < ra) std::swap(ra, rb);
        parent[static_cast<size_t>(rb)] = ra;
    }
};

// Float32 Euclidean length of one edge, computed the incumbent's
// way: Vec3f differences, sqrtf(x*x + y*y + z*z), all float32
// (Phase 0 Item 3). Endpoint order is irrelevant to the length.
float edge_length(const float* verts, const Edge& e) {
    const float* a = verts + static_cast<int64_t>(e.lo) * 3;
    const float* b = verts + static_cast<int64_t>(e.hi) * 3;
    float dx = b[0] - a[0];
    float dy = b[1] - a[1];
    float dz = b[2] - a[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Fail helper: build a not-ok FillResult carrying the reason.
FillResult fail(const char* msg) {
    FillResult r;
    r.ok = false;
    r.error = msg;
    return r;
}

}  // namespace

FillResult compute_fill(const float* verts, int64_t num_verts,
                        const int64_t* faces, int64_t num_faces,
                        float max_hole_perimeter) {
    // ---- Contract checks (loud, never silent truncation) ----------
    if (num_verts < 0 || num_faces < 0)
        return fail("negative vertex/face count");
    if (num_verts > 0 && verts == nullptr)
        return fail("verts is null but num_verts > 0");
    if (num_faces > 0 && faces == nullptr)
        return fail("faces is null but num_faces > 0");
    // Edge packing stores each endpoint in 32 bits (Phase 0 Item 7).
    if (num_verts > static_cast<int64_t>(
                        std::numeric_limits<uint32_t>::max()))
        return fail("num_verts exceeds the 32-bit edge packing");
    if (num_faces > std::numeric_limits<int64_t>::max() / 3)
        return fail("num_faces overflows the directed-edge count");

    FillResult result;
    result.ok = true;

    // Empty chunk: no faces => no edges => no boundary => no fill.
    // (The pipeline's own chain never routes an empty chunk to the
    // native fill; our worker answers it cleanly instead -- a
    // recorded, deliberate extension, Phase 0 Item 7.)
    if (num_faces == 0) return result;

    // Index validation -- the ONE deliberate difference from the
    // incumbent (Phase 0 Item 7, last row): out-of-range indices
    // fail loud here instead of indexing unchecked there.
    for (int64_t i = 0; i < num_faces * 3; ++i) {
        int64_t idx = faces[i];
        if (idx < 0 || idx >= num_verts)
            return fail("face index out of range");
    }

    // ---- Stage 1: edge census --------------------------------------
    // Every face (a,b,c) emits (a,b), (b,c), (c,a) as packed
    // undirected keys; direction in the incident face is discarded
    // here and never recovered (Phase 0 Item 2, stage 1). A face
    // like (a,a,b) therefore emits (a,a) once and (a,b) twice, and
    // (a,a,a) emits (a,a) three times -- the counts below are what
    // make duplicate faces erase their boundary (Item 7).
    std::vector<uint64_t> keys;
    keys.reserve(static_cast<size_t>(num_faces * 3));
    for (int64_t f = 0; f < num_faces; ++f) {
        uint32_t v[3];
        for (int k = 0; k < 3; ++k)
            v[k] = static_cast<uint32_t>(faces[f * 3 + k]);
        for (int k = 0; k < 3; ++k) {
            uint32_t a = v[k], b = v[(k + 1) % 3];
            keys.push_back(pack_edge(std::min(a, b), std::max(a, b)));
        }
    }
    std::sort(keys.begin(), keys.end());

    // Run-length encode the sorted keys into unique edges, still in
    // ascending (min, max) order, with incident-face counts.
    std::vector<Edge> edges;
    edges.reserve(keys.size());
    for (size_t i = 0; i < keys.size();) {
        size_t j = i + 1;
        while (j < keys.size() && keys[j] == keys[i]) ++j;
        Edge e;
        e.lo = static_cast<uint32_t>(keys[i] >> 32);
        e.hi = static_cast<uint32_t>(keys[i] & 0xFFFFFFFFu);
        e.count = static_cast<int64_t>(j - i);
        edges.push_back(e);
        i = j;
    }

    // ---- Stage 2: boundary selection --------------------------------
    // Boundary = count exactly 1 (Phase 0 Item 2, stage 2). The
    // selection scans edge indices ascending, so boundary positions
    // are in ascending edge order by construction.
    std::vector<int64_t> edge_to_bpos(edges.size(), -1);
    std::vector<int64_t> boundary_edges;  // bpos -> edge index
    for (size_t ei = 0; ei < edges.size(); ++ei) {
        if (edges[ei].count == 1) {
            edge_to_bpos[ei] = static_cast<int64_t>(boundary_edges.size());
            boundary_edges.push_back(static_cast<int64_t>(ei));
        }
    }
    const int64_t B = static_cast<int64_t>(boundary_edges.size());
    result.num_boundary_edges = B;
    if (B == 0) return result;  // closed (or boundary-erased) mesh

    // ---- Vertex manifold data ---------------------------------------
    // For every vertex we need, over ALL its incident edges:
    //   (a) its boundary-entry list -- boundary positions incident to
    //       it, in ascending boundary-position order; a self-edge
    //       (a,a) contributes TWO entries (one per endpoint), which
    //       is what makes it occupy both of a's boundary slots in
    //       the incumbent (Phase 0 Item 7, the (a,a,b) row);
    //   (b) whether any incident edge has count > 2.
    // A vertex is a MANIFOLD boundary vertex iff it has at most 2
    // boundary entries and (b) is false (Item 2, stage 3). Built as
    // CSR (offsets + flat entries) so the build is one pass over the
    // boundary list and per-vertex lists come out ascending.
    std::vector<int64_t> bnd_count(static_cast<size_t>(num_verts), 0);
    std::vector<char> touches_nonmanifold(
        static_cast<size_t>(num_verts), 0);
    for (const Edge& e : edges) {
        if (e.count > 2) {
            touches_nonmanifold[e.lo] = 1;
            touches_nonmanifold[e.hi] = 1;
        }
    }
    for (int64_t bp = 0; bp < B; ++bp) {
        const Edge& e = edges[static_cast<size_t>(boundary_edges[static_cast<size_t>(bp)])];
        if (e.lo == e.hi) {
            bnd_count[e.lo] += 2;  // self-edge: both endpoint slots
        } else {
            bnd_count[e.lo] += 1;
            bnd_count[e.hi] += 1;
        }
    }
    std::vector<int64_t> bnd_offsets(static_cast<size_t>(num_verts) + 1, 0);
    for (int64_t v = 0; v < num_verts; ++v)
        bnd_offsets[static_cast<size_t>(v) + 1] =
            bnd_offsets[static_cast<size_t>(v)] + bnd_count[static_cast<size_t>(v)];
    std::vector<int64_t> bnd_entries(
        static_cast<size_t>(bnd_offsets[static_cast<size_t>(num_verts)]), -1);
    {
        std::vector<int64_t> cursor = bnd_offsets;  // per-vertex write head
        for (int64_t bp = 0; bp < B; ++bp) {
            const Edge& e = edges[static_cast<size_t>(boundary_edges[static_cast<size_t>(bp)])];
            bnd_entries[static_cast<size_t>(cursor[e.lo]++)] = bp;
            if (e.hi != e.lo)
                bnd_entries[static_cast<size_t>(cursor[e.hi]++)] = bp;
            // For a self-edge the second slot is the same position
            // again: cursor[e.lo] has already advanced once above.
            // (e.hi == e.lo would double-write with the branch
            // above, so the count-2 accounting is: the write above
            // filled slot 1; fill slot 2 explicitly here.)
            if (e.hi == e.lo)
                bnd_entries[static_cast<size_t>(cursor[e.lo]++)] = bp;
        }
    }

    // vertex_is_manifold_boundary(v): the gate for component links.
    auto is_manifold_vertex = [&](int64_t v) -> bool {
        return bnd_count[static_cast<size_t>(v)] <= 2 &&
               !touches_nonmanifold[static_cast<size_t>(v)];
    };

    // ---- Stage 3: components through manifold vertices only --------
    // For each manifold boundary vertex with exactly 2 boundary
    // entries, union its two boundary positions. A vertex with 1
    // entry links nothing; with >= 3 entries (or a non-manifold
    // incident edge) it is not manifold and components are CUT
    // there -- this single rule produces the figure-eight lobe
    // split, the T-junction arcs, and the B>0 / M=0 no-fill shape
    // (Phase 0 Items 2 and 8) without any special case.
    UnionFind uf(B);
    for (int64_t v = 0; v < num_verts; ++v) {
        if (bnd_count[static_cast<size_t>(v)] == 2 && is_manifold_vertex(v)) {
            int64_t first = bnd_entries[static_cast<size_t>(bnd_offsets[static_cast<size_t>(v)])];
            int64_t second = bnd_entries[static_cast<size_t>(bnd_offsets[static_cast<size_t>(v)]) + 1];
            uf.unite(first, second);  // self-edge: unite(p,p), a no-op
        }
    }

    // Group boundary positions by root. Roots are component minima
    // (union-by-min), so scanning positions ascending discovers
    // components in ascending smallest-boundary-edge order -- the
    // loop order of Phase 0 Item 6, steps 3-4.
    std::vector<int64_t> group_of_root(static_cast<size_t>(B), -1);
    std::vector<std::vector<int64_t>> components;  // ascending bpos each
    for (int64_t bp = 0; bp < B; ++bp) {
        int64_t root = uf.find(bp);
        int64_t g = group_of_root[static_cast<size_t>(root)];
        if (g < 0) {
            g = static_cast<int64_t>(components.size());
            group_of_root[static_cast<size_t>(root)] = g;
            components.emplace_back();
        }
        components[static_cast<size_t>(g)].push_back(bp);
    }
    result.num_components = static_cast<int64_t>(components.size());

    // Component id per boundary position, for the loop test's
    // same-component continuation check.
    std::vector<int64_t> comp_of_bpos(static_cast<size_t>(B), -1);
    for (size_t g = 0; g < components.size(); ++g)
        for (int64_t bp : components[g]) comp_of_bpos[static_cast<size_t>(bp)] = static_cast<int64_t>(g);

    // ---- Stage 4: loop test, perimeter, centroid, fan ---------------
    // Loop test (Phase 0 Item 2, stage 4): a component is a loop
    // iff EVERY edge in it, at BOTH endpoints, finds another
    // boundary position (!= itself) in the SAME component incident
    // to that endpoint. One failing edge poisons the whole
    // component. The check scans the endpoint's boundary-entry
    // list; a self-edge's two entries are both itself, so a lone
    // self-edge component fails exactly as the incumbent's does.
    auto has_continuation = [&](int64_t bp, int64_t vertex) -> bool {
        for (int64_t i = bnd_offsets[static_cast<size_t>(vertex)];
             i < bnd_offsets[static_cast<size_t>(vertex) + 1]; ++i) {
            int64_t other = bnd_entries[static_cast<size_t>(i)];
            if (other != bp &&
                comp_of_bpos[static_cast<size_t>(other)] ==
                    comp_of_bpos[static_cast<size_t>(bp)])
                return true;
        }
        return false;
    };

    int64_t qualifying_rank = 0;  // rank among qualifying loops
    for (const std::vector<int64_t>& comp : components) {
        bool is_loop = true;
        for (int64_t bp : comp) {
            const Edge& e = edges[static_cast<size_t>(boundary_edges[static_cast<size_t>(bp)])];
            if (!has_continuation(bp, e.lo) ||
                !has_continuation(bp, e.hi)) {
                is_loop = false;
                break;
            }
        }
        if (!is_loop) continue;  // arcs, branches, tails: silently skipped
        result.num_loops += 1;

        // Perimeter: float32 sum of edge lengths in loop_boundaries
        // order (the component's ascending boundary positions).
        // Phase 0 Item 3: the incumbent's order here is CUB-internal
        // and not source-pinned, so this deterministic sequential
        // order is a faithful reproduction up to last-bit threshold
        // flips. STRICT <: equality does not fill.
        float perimeter = 0.0f;
        for (int64_t bp : comp) {
            const Edge& e = edges[static_cast<size_t>(boundary_edges[static_cast<size_t>(bp)])];
            perimeter += edge_length(verts, e);
        }
        // NaN or non-positive thresholds fail this comparison for
        // every loop -- silent no-fill, as the incumbent (Item 7).
        if (!(perimeter < max_hole_perimeter)) continue;

        // Centroid: float32 mean of the loop's EDGE MIDPOINTS,
        // summed SEQUENTIALLY in loop_boundaries order (the fork's
        // pinned order, Phase 0 Item 4), divided by float(edge
        // count). NOT a vertex mean: a vertex is weighted by its
        // incidence in the loop's edges.
        float sx = 0.0f, sy = 0.0f, sz = 0.0f;
        for (int64_t bp : comp) {
            const Edge& e = edges[static_cast<size_t>(boundary_edges[static_cast<size_t>(bp)])];
            const float* a = verts + static_cast<int64_t>(e.lo) * 3;
            const float* b = verts + static_cast<int64_t>(e.hi) * 3;
            sx += (a[0] + b[0]) * 0.5f;
            sy += (a[1] + b[1]) * 0.5f;
            sz += (a[2] + b[2]) * 0.5f;
        }
        float n = static_cast<float>(comp.size());
        int64_t centroid_index = num_verts + qualifying_rank;
        result.appended_verts.push_back(sx / n);
        result.appended_verts.push_back(sy / n);
        result.appended_verts.push_back(sz / n);

        // Fans: one per loop edge, in loop_boundaries order, each
        // (max endpoint, min endpoint, centroid) -- pure index
        // order from the edge packing (Phase 0 Item 5). The edge's
        // stored lo/hi ARE min/max by construction.
        for (int64_t bp : comp) {
            const Edge& e = edges[static_cast<size_t>(boundary_edges[static_cast<size_t>(bp)])];
            result.appended_faces.push_back(static_cast<int64_t>(e.hi));
            result.appended_faces.push_back(static_cast<int64_t>(e.lo));
            result.appended_faces.push_back(centroid_index);
        }
        qualifying_rank += 1;
    }
    result.num_qualifying_loops = qualifying_rank;
    return result;
}

bool assemble_output(const float* verts, int64_t num_verts,
                     const int64_t* faces, int64_t num_faces,
                     const FillResult& result,
                     std::vector<float>& out_verts,
                     std::vector<int64_t>& out_faces) {
    if (!result.ok) return false;
    // Append-only assembly (spec section 2.2): the output IS the
    // input arrays, copied verbatim, with the appended block behind
    // them. Nothing is mutated, reordered, or removed.
    std::vector<float> v;
    v.reserve(static_cast<size_t>(num_verts * 3) +
              result.appended_verts.size());
    if (num_verts > 0)
        v.insert(v.end(), verts, verts + num_verts * 3);
    v.insert(v.end(), result.appended_verts.begin(),
             result.appended_verts.end());
    std::vector<int64_t> f;
    f.reserve(static_cast<size_t>(num_faces * 3) +
              result.appended_faces.size());
    if (num_faces > 0)
        f.insert(f.end(), faces, faces + num_faces * 3);
    f.insert(f.end(), result.appended_faces.begin(),
             result.appended_faces.end());
    out_verts = std::move(v);
    out_faces = std::move(f);
    return true;
}

}  // namespace r1fill
