# CuMesh on AMD — the R1 hole-fill replacement (ROCm/HIP port)

**This fork:** [JeffreyXiang/CuMesh](https://github.com/JeffreyXiang/CuMesh)
is CUDA-only — its `fill_holes` does not run on AMD hardware. This fork
carries **R1**, an independent HIP implementation of the same fill
behavior, written from a source study of the original (behavior and
math, not a translation of its code) and proven on an
**RX 9070 XT (gfx1201) under ROCm 7.2**.

The AMD work lives on this branch (`amd-port`). The default upstream
code is untouched on the fork's upstream branch. Start with
[`MANIFEST.md`](MANIFEST.md) for every changed file and its proof, or
[`docs/AMD-PORT.md`](docs/AMD-PORT.md) for the full port write-up.

## What works — proven, with the test scores

| Check | Result |
|---|---|
| Calibration (25 real chunks) | **Zero failures** |
| Corpus equivalence vs the incumbent (2,355 chunks through the real seam) | **2,352 strict-equal, 3 outcome-equal, 0 violations** |
| Batch fill (job 046) | **386,195 chunks in 1,024.1 s = 377.1 chunks/s** |
| Real end-to-end generations | Jobs 044, 046, 047, 049 — the fill held in all of them |

The acceptance standard: the caller cannot tell the workers apart.
Full method, corpus construction, and harness calibration (including a
deliberately corrupt worker the harness catches) are in
[`docs/proof-and-results.md`](docs/proof-and-results.md).

## What's not working

- **The patched-CuMesh comparison wheel does not build** (found in
  job 046, Phase K2: "patched CuMesh BUILD FAILED — install
  untouched"). Open bug, recorded at wheel-build-error strength; no
  deeper diagnosis yet. The unpatched installed baseline is unaffected
  and the failure is non-cascade. Details:
  [`docs/known-issues-and-planned-work.md`](docs/known-issues-and-planned-work.md)

## What's left to do

- **Re-interface R1 to the upstream API** — R1 is proven at the worker
  seam (chunk `.npz` in, filled `.npz` out). An in-tree backend behind
  `CuMesh.fill_holes` is the owed packaging step; nothing here claims
  it exists yet.
- **Publish the equivalence corpus** — the 7,529 chunks (109.8 MB) that
  make the equivalence claim checkable are referenced by hash in
  `MANIFEST.md`, not yet carried in this repo.
- **Pin the upstream base snapshot** for CuMesh (currently unpinned;
  the studied incumbent is recorded in `MANIFEST.md`).

The six failed build attempts that led to R1 — and the defect each one
convicted — are kept in
[`docs/failure-catalog.md`](docs/failure-catalog.md): the fingerprints
are searchable on purpose.

## Building

Toolchain, exact build lines, and the load probe:
[`docs/building-on-rocm.md`](docs/building-on-rocm.md). The batch
worker design is in [`docs/batch-worker.md`](docs/batch-worker.md).

## Credits and license

Upstream: JeffreyXiang/CuMesh (MIT). This port's changes are MIT as
well. Developed as part of porting TRELLIS.2 to AMD — see the companion
fork, TRELLIS.2 `amd-port`, for the generation side.
