# Proof method and measured results

*Part of the CuMesh AMD bundle — start at `AMD-PORT.md`.*

The acceptance standard for the replacement is: **the caller cannot tell
the workers apart.** Byte-identical output against a CUDA run is not the
standard (different kernels, different precision paths); equivalence
against the incumbent *through the real seam, on real chunks, with the
parent's own vetting* is.

## The corpus

7,529 seam chunks extracted from real TRELLIS.2 run meshes, built offline
with the pipeline's own planner imported — not reimplemented — so the
chunks are exactly what production sends the worker:

- **Canvas arm: all 4,348 chunks** of the plan.
- **Subject arm: 3,181 chunks**, stratified from a 373,812-chunk plan
  (first 2,000 candidate-component chunks by id; top 200 by faces; top 200
  by boundary edges; top 200 cut-heavy; 500 random mid-size; 500
  fill-nothing; the 3 chunks that hit the incumbent's error-9 class in
  production).
- Planner reproduction against the reference run's log is **exact on every
  number**: canvas 4,348 chunks / 3,380 candidate components / 2,692
  atomic groups; subject 373,812 / 50,748 / 24,721.
- Manifests are deterministic: SHA-256 over the serialized plan is
  identical across independent rebuilds (canvas `cc44eb43d812e8ef…`,
  subject `c8697964be336929…`).
- All 7,529 files validated (keys, dtypes, shapes, index ranges, full
  face coverage of the extraction mesh): 0 bad chunks.

## The harness, calibrated in both directions

`run_harness.py` runs a worker through the real subprocess seam — real
`.npz` round-trip, the production argument shape — **twice per chunk**,
then checks: output shape growth, prefix bit-exactness (`np.array_equal`,
no tolerance), appended-index range, and bit-identical determinism across
the two runs. In `compare` mode it additionally diffs two workers'
appended vertex/face arrays and the parent's vetting verdicts
(kept faces / kept verts / dropped count); any difference is an
`EQUIVALENCE-VIOLATION`.

Calibration, all from recorded runs:

- A reference worker passes all 5 fixtures with the expected appends
  (small hole: +1 centroid at the exact expected position, +4 fans;
  big hole over threshold: 0 appends).
- A deliberately corrupt worker (one appended vertex displaced) **passes
  every single-worker check** and is caught only by compare mode:
  `EQUIVALENCE-VIOLATION` on `appended_verts` / `kept_new_verts`. This is
  why compare mode is the gate.
- Reference-vs-reference compares clean; a missing worker fails as
  `WORKER-FAILURE`. Each failure class the harness distinguishes
  (prefix / shape / index / nondeterminism / worker-failure /
  equivalence) was observed firing on a purpose-built probe worker.

One stated approximation: standalone-chunk vetting context is built from
the chunk itself (in-chunk component perimeters; the all-incident-faces
clause taken as true). For whole-mesh chunks the context is exact; for
compare mode the approximation is safe regardless, because both workers
are vetted against the identical context.

## Run results (RX 9070 XT, gfx1201, ROCm 7.2.2, hipcc HIP 7.2.53211)

**Calibration (jobs 044 and 046, in full both times).** 25 real chunks —
the set includes the five that killed attempts 041–043 — **zero
failures**. The code-9-shape probe chunk (boundary edges > 0, qualifying
loops = 0) returns clean no-fill, exit 0, output bit-identical to input.

**Corpus equivalence (jobs 044 and 046).** The 2,355-chunk sample through
compare mode against the incumbent: **2,352 strict-equal, 3
outcome-equal** (the incumbent itself fails those three; R1's outcome
matches the incumbent's fallback outcome), **0 violations**. Job 046's
re-run took 4,853 s.

**Generation (jobs 044, 046, 047).** Full TRELLIS.2 generations with R1
as the only fill worker:

- Genuine == totals on every arm — e.g. job 046 subject cascade:
  3,180,133 / 3,180,133 faces and 1,627,189 / 1,627,189 verts genuine,
  including all rows past the 2²⁰ cliff that wounds the incumbent's
  readout path.
- 0 fallback chunks on every arm of job 046; job 047 the same on its
  single arm.
- Job 044's first R1 generation: extraction 841,188 voxels → 18 chunks,
  1,683,908 faces, genuine rows = totals exactly.

**Batch mode (job 046).** See `batch-worker.md`: 386,195 chunks through
one worker process in 1,024.1 s (377.1 chunks/s) vs the per-chunk seam's
measured 2.64 chunks/s (~40.6 h projection for the same plan, job 045).
Job 047 re-ran the batch gate in full and reproduced 376.2 chunks/s
(1,026.5 s).

## What the proof does not claim

- Equivalence is against the incumbent **fork + PR #35** that the port
  actually ran, on the corpus above — not against every CuMesh version,
  and not a claim about mesh *quality* (normals, bounds, watertightness
  are downstream stages' business).
- The perimeter sum's internal order inside CUB is not source-visible in
  either incumbent copy; a loop whose float32 perimeter sits within
  rounding of the threshold can legitimately flip sides between two
  faithful implementations. No such flip appeared in 2,355 chunks; the
  tolerance question is recorded, not hidden.
