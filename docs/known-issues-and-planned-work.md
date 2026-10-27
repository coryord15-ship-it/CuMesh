# Known issues and planned work — CuMesh HIP hole-fill (R1)

Everything in this file is open, refuted, planned, or parked —
nothing here is presented as working that has not passed a run.
Items are stated at exactly the strength the campaign record
supports; numbers are the record's own.

**Publication status of this bundle:** R1's own proof gate has
passed (calibration + corpus equivalence + real generations, jobs
044/046/047 — see `proof-and-results.md`). The bundles post as a
set, and the set was gated on the TRELLIS.2 sectioned export
passing its bounds verification; that gate is satisfied (job 049,
Oct 27, 2026 — merged bounds verified, full span on every axis).
This bundle stays staged alongside it until the author confirms
publication.

## Open bugs

### 1. The patched-CuMesh wheel does not build (Phase K2 probe)

- **What:** The port maintains a patched build of the incumbent
  CuMesh as a comparison baseline (the K2 probe: baseline probe on
  the unpatched install, then a wheel build of the patched source).
  In job 046 that wheel build failed — the run's own line is
  "patched CuMesh BUILD FAILED - install untouched" (a cumesh wheel
  build error). The installed package was not modified, the probe's
  unpatched baseline ran normally, and the failure was non-cascade:
  the run still completed all phases with exit 0.
- **Status:** open bug. The cause is recorded only at "wheel build
  error" strength; no deeper diagnosis exists in the record.
- **Planned attempt:** none scheduled. The baseline the probe exists
  to compare against is the unpatched install, which is unaffected;
  the build failure is on record here so it is not rediscovered as
  a surprise.

## Planned work — API and packaging

### 2. R1 is proven at the worker seam; the upstream-API interface is still owed

- **What:** As staged, this bundle carries R1 in the form it was
  proven in: a standalone HIP backend behind a worker seam (chunk
  `.npz` in, filled `.npz` out), with its tests and harness. The
  filing package's own checklist carries one unfinished packaging
  step: "R1 source, re-interfaced to the upstream seam" — i.e. an
  in-tree CuMesh backend behind `CuMesh.fill_holes`, rather than a
  worker beside it.
- **Status:** planned (packaging, hours-class work). The gate the
  step waited on — calibration + equivalence + one real generation
  on gfx1201 — has since passed (jobs 044/046). What ships in this
  bundle is the worker-seam form; if the fork is meant to carry an
  in-tree backend, that interface work is still owed and no text in
  this bundle claims it exists.

### 3. The equivalence corpus is referenced, not yet published

- **What:** The 7,529-chunk seam corpus (109,796,038 bytes of
  `.npz`; deterministic manifest hashes in this bundle's
  `MANIFEST.md`) is what makes the equivalence claim checkable, but
  the corpus data itself stays in the local port records — it is
  referenced by this bundle, not duplicated into it.
- **Status:** open (publication decision).
- **Planned attempt:** decide the corpus's publication form —
  in-repo as-is, Git LFS, or an external host — before posting. The
  hashes are deterministic across rebuilds, so any of the three can
  be verified against the manifest.

### 4. The CuMesh base snapshot is not pinned to a commit

- **What:** The incumbent that was studied, replaced, and
  equivalence-tested is identified in the records only as the
  community fork (`cumesh-visualbruno`) plus PR #35 (the io.cu
  chunked-copy patch). No upstream commit hash for
  JeffreyXiang/CuMesh appears in any source read.
- **Status:** open (record gap).
- **Planned attempt:** pin the base commit before publication, so
  the equivalence claims have a named baseline instead of a fork
  name and a PR number.

## Refuted

### 5. The suspected AMD-library (hipcub) defect

- **What was believed:** during the attempt-043/044 debugging, the
  fill's wrong counts were suspected to be a bug in AMD's hipcub
  layer, and a ROCm bug report was reserved in the posting plan
  conditional on that conviction.
- **What the evidence showed:** job 044 convicted R1's own
  `count_runs_kernel` — it binary-searched the exclusive scan as if
  it were a run-index array, shifting every run count by one. No
  AMD-library defect was ever evidenced, and no ROCm bug report
  exists. After the fix, calibration passed with zero failures on
  all 25 chunks and the full corpus equivalence reproduced (2,355
  chunks: 2,352 strict-equal, 3 outcome-equal, 0 violations).
- **Status:** refuted. The episode is kept in
  `docs/failure-catalog.md` as part of the attempts arc; it is
  recorded here so the suspicion is never repeated as a finding.

## Parked

### 6. Batch-worker scaling variants (V2, V3) — triggers never fired

- **What:** Two follow-on designs for the fill stage were filed
  behind the batch worker (V1): V2, K parallel batch processes
  overlapping host I/O, to be built only if V1 missed; and V3, a
  device-side work queue (all chunks staged on-card in one bulk
  transfer, a device-side counter handing kernels the next chunk,
  one download at the end), to be built only if V1+V2 missed.
- **What the run showed:** V1 hit. Job 046 ran the full
  386,195-chunk plan through one worker process in 1,024.1 s
  (377.1 chunks/s) against the previous path's measured
  2.64 chunks/s (~0.473 s of process spawn + HIP init per chunk,
  a ~40.6 h projection for the same plan); job 047 reproduced the
  pace (376.2 chunks/s).
- **Status:** parked. Neither trigger fired, so neither variant is
  built. V3's design brief (`R1-V2-DESIGN-BRIEF.md` in the port
  records) is kept on file if a future fill profile ever misses
  the bar.
