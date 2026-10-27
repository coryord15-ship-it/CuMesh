# Failure catalog — six attempts, every failure named

*Part of the CuMesh AMD bundle — start at `AMD-PORT.md`.*

Each entry: the attempt, what died, the exact cause once convicted, the
fix. Failures are included because they are the part a reader can trust,
and several are searchable fingerprints for anyone hitting the same wall
on ROCm.

1. **Attempt 1 — compile, one word.** `hipMemcpyKind_t` is CUDA spelling;
   ROCm's type is `hipMemcpyKind`. Everything before the gate passed.
   Lesson, now standing practice: audit every API identifier against the
   target toolchain *before* the first ship, not after the first failure.
2. **Attempt 2 — fixture battery, zero-hole chunks.** Compile passed
   (~9 s), the kernel ran, a small hole filled exactly — but closed-cube
   and duplicate-faces fixtures (zero boundary edges) exited 1 where the
   host returns success with 0 appends: an unguarded device call on the
   zero-count path.
3. **Attempt 3 — battery fixed, calibration failed on real chunks.** The
   B == 0 early-out worked (battery 5/5), but the worker exited 1 on 5 of
   25 real probe chunks and on the code-9 shape (B > 0 boundary edges,
   M = 0 manifold vertices — a designed clean-no-fill case).
4. **Attempt 4 — code-9 fixed by construction, five chunks still died.**
   The early-out was re-keyed on `num_qualifying_loops == 0` before any
   device call: the code-9 probe then passed clean, output bit-identical.
   The five chunks refuted the zero-append theory — they carry *large*
   plans (qualifying loops 744 / 597 / 383 / 57 / 24) and no host
   statistic separated them from the 20 passing chunks. The CPU worker
   passed all 25 through the identical seam: a device-side failure whose
   stderr never reached the log (the harness held it in memory only).
5. **Attempt 5 — instrumentation named it.** All five chunks, both runs:
   `R1-FAIL stage=fill: device/host boundary count disagreement` — R1's
   own cross-check firing. Not a HIP error, not a crash: the device and
   host were counting different boundaries.
6. **Attempt 6 — convicted and fixed.** `count_runs_kernel`
   binary-searched the exclusive scan as if it were a run-index array:
   `scan[i]` counts run starts strictly before i, so every run count
   shifted by one (`count[0] = 1` always, `count[r] = true[r−1]`, last
   run unread). The device boundary total came out exactly host + 1
   whenever the chunk's largest unique edge had count ≥ 2 — delta +1 on
   exactly the five failing chunks, 0 on all 20 passes. A host replica of
   the bug predicted 25/25 device outcomes and retroactively explained
   attempt 2's fixture deaths. Fix: scatter run-start positions and
   difference consecutive starts — exact by construction. The cross-check
   stays at full strength in the shipped code; it caught a real
   divergence every earlier gate missed.

Standing note: compiles took **seconds** in every attempt. The long waits
in this campaign were file transfers and test cycles, never compilation —
a wrong assumption newcomers to ROCm ports often make.
