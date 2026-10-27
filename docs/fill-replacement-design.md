# R1 design: the fill contract, reproduced exactly

*Part of the CuMesh AMD bundle — start at `AMD-PORT.md`.*

R1 replaces CuMesh's `fill_holes` at the worker seam. The incumbent's
behavior was established by a complete source study before any kernel code
was written; this page is the contract that study produced. Where the
incumbent's behavior looks wrong, R1 reproduces it anyway — a replacement
that "improves" the semantics is a different mesh generator, and the
equivalence gate (see `proof-and-results.md`) is built to catch exactly
that.

## The seam

```
argv:  fill_worker  <in.npz>  <out.npz>  <repr(float(max_hole_perimeter))>
in:    verts float32 (K,3), faces int64 (M,3), chunk-local indexing
out:   same arrays, append-only: out = [input rows; appended rows]
exit:  0 success · nonzero any failure, and no output file is written
```

Chunk budgets in the calling planner: 131,072 faces / 262,144 verts per
chunk; chunk-local vertex ids therefore stay ≤ 2¹⁸, inside the incumbent's
32-bit edge-packing envelope.

## What the fill does (and only this)

A source-level audit of the incumbent's full call chain (TRELLIS.2
`Mesh.fill_holes` → Python wrapper → native `CuMesh::fill_holes`) found
**no cleanup, dedup, reordering, deletion, or orientation work anywhere on
the fill path** — no `remove_faces`, no degenerate-face removal, no
unreferenced-vertex removal, despite several of those living in the same
source file. The chain computes connectivity caches, extracts loop
components, appends centroids and fans, and reads the arrays back. R1
reproduces exactly this scope. Adding any "safety" cleanup would be a
behavior difference, not an improvement.

## The algorithm, as specified by the incumbent

1. **Edges.** Each face emits 3 undirected edges packed as `(min<<32)|max`.
   Sort + run-length encode → unique edges with incident-face counts.
   Direction in the incident face is discarded here and never recovered.
2. **Boundary.** Edges with count exactly 1, in ascending edge-index order.
3. **Manifold boundary vertices.** A vertex is one iff over all its incident
   edges it has at most 2 boundary edges and no edge with count > 2.
   Branching vertices (≥3 boundary edges) and vertices touching a
   non-manifold edge emit no adjacency pair — components are cut there.
4. **Components.** Union-find (hook-and-compress; the higher root always
   unions into the lower, so a component's root is its smallest boundary
   position) over boundary positions, linked by the manifold-vertex pairs.
5. **Loop test.** A component is a loop iff *every* edge in it has, at both
   endpoints, another boundary position of the same component. One failing
   edge fails the whole component — a loop with a tail is never partially
   filled; open arcs and T-junctions are silently skipped.
6. **Perimeter test.** Per loop: float32 sum of Euclidean edge lengths
   (`sqrtf(x*x + y*y + z*z)` in float32), compared with strict `<` against
   the float32 threshold in raw mesh units. Perimeter == threshold ⇒ not
   filled. No proxy (edge count, bbox, area) enters the test.
7. **Centroid.** Per qualifying loop: the arithmetic mean of the loop's
   **edge midpoints** — summed in float32, sequentially in the loop's
   edge order, divided by `float(edge_count)`. On a simple loop this equals
   the vertex mean in exact arithmetic; on non-simple loops vertices are
   weighted by incidence, because the sum runs over edges. R1 matches the
   incumbent fork's pinned sequential order (upstream's CUB segmented
   reduction order is not source-visible; the fork is the reference).
8. **Fans.** One face per loop edge, emitted as
   `(larger endpoint, smaller endpoint, centroid)` — the packed edge
   decoded low-word first. Winding is index order, not mesh orientation;
   the incumbent makes no attempt to unify it, and neither does R1.
9. **Append order.** Centroids in ascending qualifying-loop order, then
   fans in retained loop-edge order (loop-major, edges ascending within a
   loop). The output prefix is the input, bit-exact.

Divergence that is incumbent behavior, kept deliberately: the *parent's*
census groups boundary edges by any shared vertex and sums perimeters per
graph component in float64, while the kernel tests per manifold-component
in float32. On pinched (multi-lobe) components the two can disagree —
kernel fills a lobe the parent then drops. Both sides are reproduced as
found; the harness's vetting capture records the parent's verdict per
chunk so the interaction is observable, not hidden.

## Hostile inputs, per case (all reproduced)

| Case | Behavior |
|---|---|
| Duplicate faces | Not removed; shared edges gain count ≥ 2 and stop being boundary. A duplicated rim erases that rim from the boundary set. |
| Duplicate vertex positions | Topology is index-based only; positions are never compared. A rim split across duplicate indices is topologically open ⇒ not filled. |
| Face with a repeated vertex (a,a,b) | Emits a self-edge (a,a) (boundary, count 1) whose component fails the loop test ⇒ not filled, no error. |
| Face (a,a,a) | One edge with count 3 ⇒ non-manifold; no boundary; not filled. |
| Non-manifold edge (≥3 faces) | Excluded from boundary; cuts component links at its vertices. Faces retained untouched. |
| Isolated vertex | Invisible to every stage; preserved verbatim in the output. |
| Single isolated triangle | It *is* a loop (all vertices manifold boundary, degree 2) ⇒ filled if under threshold: +1 centroid, +3 fans. |
| Threshold ≤ 0 or NaN | Strict `<` qualifies nothing; silent no-fill, no error. +inf qualifies every loop. |
| Out-of-range face indices | The incumbent does not validate (undefined behavior). R1 validates and exits nonzero — a **recorded deliberate difference**, unreachable from the planner, which only emits in-range chunks. |

## The two defect classes R1 is built to avoid

**The 2²⁰ copy cliff.** The incumbent's `init` uploads with single-shot
2D copies whose height is the row count; on HIP that path zero-fills rows
≥ 2²⁰ (see `AMD-PORT.md`). R1's transfer is linear and chunked: one
`hipMemcpy` call site, ≤ 65,536 rows per call. Verify after any edit:
`grep -n "hipMemcpy(" fill_hip.hip.cpp` must print exactly one line.

**The zero-grid launch (error 9 class).** The incumbent launches a kernel
whose grid is sized by the manifold-boundary-vertex count M, unguarded.
M = 0 with B > 0 boundary edges (a chunk whose entire rim lives on
non-manifold skeleton) yields a zero grid dimension. R1 guards every
data-sized launch (`count > 0`), computes grids in 64-bit, and asserts the
count is inside the designed envelope before launching — fail loud to the
seam (worker exit → caller ladder), never an invalid launch. For the
M = 0 ∧ B > 0 shape itself, R1 returns the mathematically correct result
(no fill) with exit 0.

## The cross-check

After the device fill, the worker recounts the boundary total on device
and compares it against the host plan's count for the same chunk. Any
disagreement is a hard failure (no output file), not a warning. Rationale:
every earlier gate in the campaign — fixture battery, CPU tests — passed a
worker whose run-count computation was off by one (see
`failure-catalog.md`); the cross-check is the instrument that caught it,
so it ships at full strength.
