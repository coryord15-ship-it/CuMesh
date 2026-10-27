# CuMesh on AMD: a HIP hole-fill backend (R1)

**An independent HIP replacement for upstream CuMesh's `fill_holes` on
AMD GPUs (ROCm): it drops into the same worker seam, avoids the 2²⁰ HIP
upload cliff by construction, and is proven against the incumbent by
calibration and a 2,355-chunk corpus equivalence check (0 violations).**

Developed while porting TRELLIS.2 inference to an RX 9070 XT. This page
documents the hole-fill component only: what the upstream code does, where it
assumes CUDA, what actually breaks on ROCm/HIP, and the replacement that was
built and proven.

## Status at a glance

| Check | Result |
|---|---|
| Calibration — 25 real chunks | Zero failures (jobs 044, 046) |
| Corpus equivalence — 2,355 chunks vs the incumbent | 2,352 strict-equal · 3 outcome-equal · **0 violations** |
| Batch fill — 386,195 chunks (job 046, subject arm) | 1,024.1 s = 377.1 chunks/s |
| Real generations with R1 as the fill worker | Genuine rows equal to totals on every arm; 0 fallback chunks (jobs 044, 046, 047) |

## Tested environment

| Component | Version / value |
|---|---|
| GPU | RX 9070 XT (gfx1201, RDNA 4) |
| ROCm | 7.2.x — proof runs on 7.2.2 |
| hipcc | HIP 7.2.53211 |
| PyTorch | 2.11.0 |
| Host environment | WSL2 Ubuntu 24.04 |

Full toolchain detail: `building-on-rocm.md`.

## In this bundle

- `fill-replacement-design.md` — the fill contract, point by point, and
  the hostile-input table.
- `proof-and-results.md` — proof method, corpus, and full result numbers.
- `batch-worker.md` — the `--batch` worker mode and its measured rates.
- `building-on-rocm.md` — toolchain, exact build lines, load probe,
  slot-in.
- `failure-catalog.md` — the six attempts it took to get here, each
  defect convicted.
- `known-issues-and-planned-work.md` — open bugs and planned work.

**Failures first:** if you are here because something broke, start with
`failure-catalog.md` (what broke before, and how each defect was
convicted) and `known-issues-and-planned-work.md` (what is still open).

## What the upstream component does

CuMesh is a CUDA mesh-processing library used by TRELLIS.2's export path. The
operation that matters here is `CuMesh::fill_holes(max_hole_perimeter)`:

1. Build edge structure: every face emits its 3 edges as packed 64-bit keys
   `(min<<32)|max`; a radix sort + run-length encode yields unique edges with
   incident-face counts.
2. Boundary edges are exactly the edges with incident-face count **1**.
3. Boundary components are formed through *manifold boundary vertices* only
   (a vertex with at most 2 boundary edges and no incident edge of count > 2).
   A component in which every edge continues within the component at both
   endpoints is a **loop**.
4. Per loop, the perimeter (float32 sum of Euclidean edge lengths) is tested
   with a strict `<` against `max_hole_perimeter` (mesh units; stock default
   `3e-2`; no normalization of any kind).
5. Each qualifying loop gets **one centroid vertex** (the float32 mean of the
   loop's edge midpoints, accumulated sequentially in loop order, divided by
   the edge count) and **one fan face per loop edge**, emitted as
   `(larger endpoint, smaller endpoint, centroid)`.
6. Centroids and fans are **appended** to the vertex/face arrays. Nothing is
   deleted, deduplicated, reordered, or reoriented anywhere on the fill path.

That behavior — including its quirks (index-order fan winding, lobe-wise
perimeter tests on pinched loops, silent no-fill for open arcs) — is the
contract a replacement must reproduce. The full source-study record is
summarized in `fill-replacement-design.md`.

## Where CUDA is assumed

- The native implementation is CUDA C++ (`src/clean_up.cu`,
  `src/connectivity.cu`, `src/io.cu`) built on CUB primitives
  (`DeviceRadixSort`, `DeviceSelect`, `DeviceSegmentedReduce`, scans).
- `CuMesh::init` uploads vertices/faces with single-shot `cudaMemcpy2D`
  calls whose height is the row count.
- The Python wrapper asserts CUDA tensors on a CUDA device.

## What breaks on AMD/ROCm, specifically

1. **The 2²⁰ upload cliff.** On HIP, the 2D-copy upload path zero-fills
   every row at index ≥ 2²⁰ (1,048,576). Meshes read back through it carry a
   contiguous all-zero desert starting at exactly row 1,048,576. This is a
   known wound class upstream: CuMesh issue #42 (filed 2026-09-24) reports
   `CuMesh.init()` zero-filling rows ≥ 2^20 on HIP / RX 9070 XT /
   ROCm 7.1.1, and PR #35 (chunked linear copies in `io.cu`) addresses the
   copy mechanics but was still open at study time. TRELLIS.2-class meshes
   routinely exceed 2²⁰ rows, so this is not an edge case.
2. **Error 9 on a data-shaped input.** `get_boundary_connected_components`
   launches its hook kernel with a grid sized by M, the number of manifold
   boundary vertices, with no guard for M = 0. A chunk whose boundary has
   *no* manifold boundary vertex (B > 0 boundary edges, M = 0) constructs a
   zero grid dimension → HIP/CUDA error 9, "invalid configuration argument",
   which poisons the HIP context and segfaults the host process. Three
   chunks of one real 373,812-chunk plan hit this class.
3. **Nothing else to stand on.** There is no published HIP build of CuMesh
   to fall back to; the CUDA source does not compile under hipcc as-is
   (CUDA spellings throughout, e.g. `hipMemcpyKind_t` vs ROCm's
   `hipMemcpyKind`).

## The change: R1

R1 is an independent HIP implementation of the fill contract above, written
from the source study (behavior and math; the incumbent's code is not
reproduced). It drops into the same worker seam the pipeline already used:

```
fill_worker  in.npz  out.npz  <max_hole_perimeter>
# in.npz / out.npz keys: verts float32 (K,3), faces int64 (M,3)
# exit 0 on success; nonzero on any failure, with no output file written
```

Design points that matter for correctness on AMD:

- **Chunked linear transfers.** Exactly one `hipMemcpy` call site, moving
  at most 65,536 rows per call. The 2²⁰ 2D-copy cliff is avoided by
  construction, not by patching the incumbent's copy code.
- **No zero-sized launches.** Kernel grids are never computed from a count
  that can be 0 without an explicit guard. The M = 0 ∧ B > 0 chunk class
  returns the correct result (no fill — its components are all singletons
  and none passes the loop test) with exit 0, where the incumbent crashes.
- **A device/host cross-check at full strength.** The device path recounts
  the boundary total independently and compares it against the host plan on
  every chunk; a disagreement aborts the chunk instead of emitting a wrong
  fill. This check is what convicted R1's own last defect (see
  `failure-catalog.md`, attempts 5–6) — it is kept in the shipped code.
- **Determinism by construction.** Edges sorted by (min, max); loops
  ordered by smallest boundary edge; edges within a loop ascending;
  qualification filter applied in that order; centroids then fans appended
  in that order. Same chunk in ⇒ same appended block out.

## Building it (quick path)

Prove the CPU logic first, then build the HIP worker (full recipe,
load probe, and slot-in: `building-on-rocm.md`):

```bash
g++ -std=c++17 -O2 -Wall -Wextra -I. \
    tests/test_fill_logic.cpp fill_logic.cpp -o test_fill_logic
./test_fill_logic                     # expect: ALL TESTS PASS

/opt/rocm/bin/hipcc -std=c++17 -O2 --offload-arch=gfx1201 \
    -DR1_USE_HIP -I. \
    fill_worker.cpp fill_logic.cpp fill_hip.hip.cpp \
    -lz -o fill_worker
```

## Measured results (RX 9070 XT, gfx1201, ROCm 7.2.2, torch 2.11.0)

- **Calibration:** 25 real chunks (including the five that killed attempts
  041–043) — zero failures (job 044; re-run in full in job 046).
- **Corpus equivalence** against the incumbent over a 2,355-chunk sample:
  2,352 strict-equal, 3 outcome-equal (chunks where the incumbent itself
  fails), **0 violations** (jobs 044 and 046).
- **Real generations:** full TRELLIS.2 runs with R1 as the fill worker,
  genuine vertex/face rows equal to totals on every arm (no 2²⁰ wound),
  0 fallback chunks (jobs 044, 046, 047). Per-run fill throughput is in
  `batch-worker.md`: 386,195 chunks in 1,024.1 s (377.1 chunks/s) in job
  046's subject arm.

Proof method and full numbers: `proof-and-results.md`.

## Credits

- Upstream: **CuMesh** by JeffreyXiang (MIT) — the library whose
  `fill_holes` contract this replacement reproduces.
- The incumbent this port ran against is the community fork carrying
  PR #35 (chunked copies in `io.cu`); upstream issue #42 documents the
  same 2²⁰ wound class on HIP.
- R1 was written from a source study of the incumbent's behavior; no
  incumbent code is reproduced in it.

## License

MIT — the same license as upstream CuMesh.
