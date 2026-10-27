// fill_logic.h -- R1 hole-fill LOGIC: the algorithm's brain, as a
// separable host unit.
//
// WHAT THIS IS
//   A pure C++17 / STL-only implementation of the incumbent CuMesh
//   hole-fill behaviour as established by the Phase 0 source reading
//   (../PHASE0-FINDINGS.md, which is the authority for every choice
//   below). No HIP, no CUDA, no torch includes: this header and its
//   .cpp compile with g++ alone, so the exact same logic is unit-
//   testable on any host (tests/test_fill_logic.cpp) and is the
//   reference the HIP wrapper (fill_hip.hip.cpp) mirrors on device.
//
// WHAT THE ALGORITHM DOES (Phase 0, Items 1-9, in one paragraph)
//   Boundary edges are undirected edges with exactly one incident
//   face. Boundary connected components are built ONLY through
//   manifold boundary vertices (a vertex incident to at most 2
//   boundary-edge entries and to no edge with >= 3 incident faces);
//   there is NO cycle tracing anywhere. A component is a loop iff
//   every one of its edges finds a same-component continuation at
//   BOTH endpoints. A loop is filled iff its perimeter -- the
//   float32 sum of its edges' Euclidean lengths, in raw mesh units
//   -- is STRICTLY less than max_hole_perimeter. A filled loop gains
//   exactly one centroid vertex (the float32 mean of its edges'
//   MIDPOINTS, summed sequentially in loop order) and one fan face
//   per loop edge, wound (max endpoint, min endpoint, centroid) --
//   pure index order, orientation-blind by construction; do not
//   "fix" it. Nothing else happens: no dedup, no degenerate-face
//   removal, no unreferenced-vertex removal, no reordering. The
//   output is the input copied verbatim plus one appended block
//   (spec section 2.2: append-only is a construction rule).
//
// DETERMINISM
//   Edges are ordered ascending by (min, max) vertex index; loops
//   are ordered by their smallest boundary edge; edges inside a
//   loop are ascending; the qualify filter preserves that order;
//   centroids are appended in loop order, then fans in loop order
//   (edges in loop_boundaries order). Same input => same appended
//   block, byte for byte.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace r1fill {

// Result of one fill computation over one chunk.
//
// The appended block is the ONLY output: the caller assembles the
// full output as (input arrays copied verbatim) + (this block),
// which is what makes the spec section 2.2 prefix invariant hold
// by construction rather than by check.
struct FillResult {
    // True when the computation completed (including the many
    // legitimate "nothing qualifies, append nothing" outcomes).
    // False only on a contract violation by the INPUT (out-of-range
    // face index, absurd counts) -- a deliberate, recorded
    // difference from the incumbent, which leaves such indices
    // unchecked (Phase 0 Item 7); the parent treats any worker
    // failure through its ladder, and parent-produced chunks are
    // in-range by construction, so the difference is unreachable
    // in the pipeline.
    bool ok = false;

    // Human-readable failure reason when ok == false; empty else.
    std::string error;

    // Appended centroid vertices, float32, row-major (Q, 3), in
    // qualifying-loop order. Centroid k belongs to the k-th
    // qualifying loop and has vertex index (num_input_verts + k).
    std::vector<float> appended_verts;

    // Appended fan faces, int64, row-major (E, 3), in
    // qualifying-loop order, edges in loop_boundaries order inside
    // each loop. Every face is (max endpoint, min endpoint,
    // centroid index) -- Phase 0 Item 5, reproduced exactly.
    std::vector<int64_t> appended_faces;

    // Diagnostics (not part of the seam; used by tests and logs).
    int64_t num_boundary_edges = 0;    // edges with exactly 1 face
    int64_t num_components = 0;        // boundary components found
    int64_t num_loops = 0;             // components passing the loop test
    int64_t num_qualifying_loops = 0;  // loops passing the perimeter test
};

// Compute the appended block for one chunk.
//
//   verts              : float32 positions, (num_verts, 3) row-major.
//                        May be nullptr only when num_verts == 0.
//   num_verts          : number of input vertices (>= 0). Must fit
//                        the 32-bit edge packing: <= 2^32 - 1.
//   faces              : int64 indices, (num_faces, 3) row-major,
//                        chunk-local. May be nullptr only when
//                        num_faces == 0.
//   num_faces          : number of input faces (>= 0).
//   max_hole_perimeter : fill threshold in raw mesh units, taken
//                        per invocation (never hardcoded). A loop
//                        fills iff perimeter < threshold (strict,
//                        float32). <= 0 or NaN therefore fills
//                        nothing, silently -- matching the
//                        incumbent (Phase 0 Item 7).
//
// All counts are explicit 64-bit; any count outside the designed
// envelope fails loud (ok == false) instead of truncating.
FillResult compute_fill(const float* verts, int64_t num_verts,
                        const int64_t* faces, int64_t num_faces,
                        float max_hole_perimeter);

// Assemble a full output mesh the append-only way: copies of the
// input arrays with the result's appended block behind them.
// Returns false (and leaves the outputs untouched) when the result
// is not ok. Used by the worker shell and by the tests; the prefix
// of the outputs is bit-identical to the inputs by construction
// (plain memcpy semantics through std::vector assignment).
bool assemble_output(const float* verts, int64_t num_verts,
                     const int64_t* faces, int64_t num_faces,
                     const FillResult& result,
                     std::vector<float>& out_verts,
                     std::vector<int64_t>& out_faces);

}  // namespace r1fill
