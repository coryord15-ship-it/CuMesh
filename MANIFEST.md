# Bundle manifest — CuMesh (HIP hole-fill backend)

**Upstream repo:** JeffreyXiang/CuMesh (Jianfeng Xiang)
**License:** MIT (verified against the LICENSE file in the local upstream study copy)
**Fork shape (planned):** fork under the author's name; the AMD backend lands on a
clearly named branch. A pull request upstream is optional and later; this bundle
is the staging copy, not a submission.
**Base snapshot:** no upstream commit hash is recorded in the local port records
for CuMesh — the incumbent that was studied and replaced is the community fork
(`cumesh-visualbruno`) plus PR #35 (io.cu chunked-copy patch). See GAPS.md.

## Why this bundle exists

CuMesh's hole fill (`fill_holes`) is CUDA-built and does not run on AMD/ROCm.
This bundle contains **R1**: an independent HIP implementation of the same
fill behavior, written from a source study of the incumbent (behavior and
math), not a translation of its code. R1 speaks the worker seam the TRELLIS.2
port already used (chunk `.npz` in, filled `.npz` out), and was proven against
the incumbent by calibration + corpus equivalence on an RX 9070 XT (gfx1201).

## Changed / added files (as they would be committed)

| File | Purpose | Proven by |
|---|---|---|
| `changes/src/fill_hip.hip.cpp` (sha256 9dcc851a…) | HIP device fill path: edge census, boundary/loop structure, perimeter test, centroid fans, device/host boundary cross-check. One `hipMemcpy` call site, chunked at ≤65,536 rows. | Jobs 044, 046, 047 |
| `changes/src/fill_hip.h` (b10ae6fb…) | Host-side declaration of the HIP fill entry point and its contract. | Jobs 044+ |
| `changes/src/fill_logic.cpp` (9743e000…) | Host fill logic shared by the CPU and HIP workers — the plan/brain the device path reuses; byte-identical across jobs 044–047 (sha-locked). | CPU tests + jobs 044, 046, 047 |
| `changes/src/fill_logic.h` (0ded1854…) | Host fill logic declarations. | CPU tests |
| `changes/src/fill_worker.cpp` (502e6c83…) | Worker executable shell: single-shot seam (`in.npz out.npz threshold`) plus the `--batch` manifest mode added in r35.7. | Jobs 044 (single-shot), 046 (batch gate + run), 047 |
| `changes/tests/test_fill_logic.cpp` | CPU test suite for the host logic (`ALL TESTS PASS` is a run gate). | Every campaign run |
| `changes/harness/run_harness.py` | Standalone equivalence/invariant checker: runs a worker through the real seam twice per chunk; checks shape, prefix bit-exactness, index range, determinism; `compare` mode diffs two workers' appended blocks and parent vetting verdicts. | Calibrated in 8 recorded runs (see docs/proof-and-results.md) |
| `changes/harness/fixtures/` | Five fixture chunks (closed cube, small hole, big hole, duplicate faces, non-manifold edge) + reference/identity/corrupt fixture workers used to calibrate the harness in both directions. | Harness calibration runs 01–08 |
| `changes/corpus-tools/*.py` | Corpus construction/validation tools: build the 7,529-chunk seam corpus from real run meshes with the planner imported (not reimplemented), verify sources, validate every serialized chunk. | Corpus build record |

## Not copied into this bundle

The corpus data itself: 7,529 `.npz` chunk files, 109,796,038 bytes
(canvas manifest SHA-256 `cc44eb43d812e8ef…`, subject `c8697964be336929…`,
both deterministic across rebuilds). It lives in the local port records
(`pipeline-rethink-20261026/r1-build/corpus/`) and is referenced, not
duplicated, in this staging bundle. See GAPS.md.

## Docs in this bundle

- `docs/AMD-PORT.md` — what CuMesh's fill does, where CUDA is assumed, what
  breaks on HIP, and what R1 is.
- `docs/fill-replacement-design.md` — the fill contract R1 reproduces, point
  by point, and the two defect classes it is built to avoid.
- `docs/batch-worker.md` — the `--batch` mode: one worker process per chunk
  plan instead of one per chunk.
- `docs/proof-and-results.md` — corpus, harness calibration, and the measured
  run results (jobs 044/046/047).
- `docs/building-on-rocm.md` — toolchain, exact build lines, load probe.
- `docs/failure-catalog.md` — the six build attempts and the defect each one
  convicted; kept because the fingerprints are searchable.
- `docs/known-issues-and-planned-work.md` — open bugs, refuted items,
  planned work, and parked items for this bundle (the K2 patched-wheel
  build failure, the upstream-API re-interface, the corpus publication
  decision, the unpinned base snapshot).
