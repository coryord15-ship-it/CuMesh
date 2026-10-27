# Building and running R1 on ROCm

*Part of the CuMesh AMD bundle — start at `AMD-PORT.md`.*

Proven stack: **RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.x, WSL2 Ubuntu
24.04** (the campaign's proof runs used ROCm 7.2.2, hipcc HIP 7.2.53211).
If `hipcc` reports a different ROCm major/minor than the build was
developed against, treat that as a recorded event, not a detail to absorb
silently.

## Toolchain check

```bash
/opt/rocm/bin/hipcc --version        # expect HIP from ROCm 7.2.x
g++ --version                        # host worker + tests
python3 -c "import numpy; print(numpy.__version__)"
ls /usr/include/zlib.h               # worker links -lz
```

## 1. CPU worker first (proven logic, no GPU)

```bash
g++ -std=c++17 -O2 -Wall -Wextra -I. \
    tests/test_fill_logic.cpp fill_logic.cpp -o test_fill_logic
./test_fill_logic                     # expect: ALL TESTS PASS

g++ -std=c++17 -O2 -Wall -Wextra -I. \
    fill_worker.cpp fill_logic.cpp -lz -o fill_worker_cpu
./fill_worker_cpu <fixture>/small_hole.npz /tmp/out_small.npz 5.0
                                      # expect: exit 0
```

## 2. HIP worker

One binary: the same `fill_worker.cpp` shell compiled by hipcc with
`-DR1_USE_HIP`, linking the HIP fill path and the shared host logic (the
device path reuses the host plan; it does not reimplement it).

```bash
/opt/rocm/bin/hipcc -std=c++17 -O2 --offload-arch=gfx1201 \
    -DR1_USE_HIP -I. \
    fill_worker.cpp fill_logic.cpp fill_hip.hip.cpp \
    -lz -o fill_worker
```

- `--offload-arch=gfx1201` is the RX 9070 XT's architecture. Do not let
  hipcc auto-detection substitute another target; per-model arch flags
  belong in a spec profile, not in an assumption. (Other RDNA targets
  were not part of the proof runs.)
- The HIP file uses hipcub (`DeviceRadixSort`, `DeviceScan`), which ships
  with ROCm; no extra package is expected. If the link names a missing
  hipcub/rocprim symbol, record the exact error — do not work around it
  silently.
- Identifier audit before the first build on a new toolchain: CUDA
  spellings do not survive hipcc (campaign attempt 1 died on
  `hipMemcpyKind_t`; ROCm's type is `hipMemcpyKind`). Audit every API
  identifier against the target headers *before* shipping, not after the
  first failure. Compiles take seconds in this codebase; the expensive
  failures are never compile time.

## 3. Load probe — before any run trusts the binary

```bash
# (a) exit discipline: a missing input fails nonzero, writes nothing
./fill_worker /nonexistent.npz /tmp/never.npz 0.03 ; echo "exit=$?"

# (b) the calibration fixture fills exactly: prefix bit-exact,
#     +1 centroid at [0,0,0], +4 fan faces
./fill_worker <fixture>/small_hole.npz /tmp/probe_small.npz 5.0

# (c) the harness, check mode, whole fixture set: expect 5 PASS, exit 0
python3 run_harness.py check --worker "$PWD/fill_worker" \
    --corpus <fixtures>/chunks --max-hole-perimeter 5.0

# (d) then the corpus, check mode, before any production slot-in
```

Expected compare-mode note: against the harness's
`reference_fill_worker.py`, the real worker reports an
`EQUIVALENCE-VIOLATION` on `appended_faces` **only** — the reference
fixture encodes an early reading of the spec (vertex-mean centroid in
float64, `(centroid, b, a)` winding). The source study established the
incumbent's actual choices (edge-midpoint centroid in float32,
`(max, min, centroid)` fans), and R1 reproduces those. Centroids,
appended counts, kept counts and dropped counts are the comparable
quantities; on symmetric fixtures the centroids coincide exactly.

## 4. Slot-in

The parent changes by one command pointer: wherever the fill worker
command is constructed, point it at the new binary. The worker's argv is
exactly `in_path out_path threshold` — no interpreter, no mode flag (batch
mode is a separate argv form, see `batch-worker.md`). Census, planner,
core, fallback ladder, and every log line in the parent stay unchanged.

Keep the incumbent worker runnable, renamed aside, until the replacement
has passed calibration + equivalence + one real generation on the target
machine. Rollback is pointing the same one command back. Deleting the
incumbent is not part of any step.
