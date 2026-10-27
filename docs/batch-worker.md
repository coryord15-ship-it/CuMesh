# Batch mode: one worker process per chunk plan (r35.7, job 046)

*Part of the CuMesh AMD bundle — start at `AMD-PORT.md`.*

## The problem, measured

The original seam spawns one worker process per chunk (a crash-isolation
design: a poisoned HIP context can only kill one chunk). On the RX 9070 XT
bench that isolation costs **~0.473 s per chunk** — process spawn + HIP
runtime init + `.npz` I/O — nearly independent of chunk size: a 4-vert /
2-face chunk pays the same toll as a 68,631-vert chunk, while the fill
compute itself is ~0.017 s on a no-op chunk.

Measured baseline (job 045, subject arm): 868 of 386,195 planned chunks in
329 s → **2.64 chunks/s**, projecting **~40.6 h** for one arm's fill. The
toll is orchestration, not fill work.

## The change

`fill_worker.cpp` gains a `--batch` mode. The single-shot code path is
untouched: the per-item body was extracted unchanged into
`run_one(in, out, threshold, &appended_faces)`; `argc == 4` behaves
exactly as before (same prints, same exit codes).

```
fill_worker  --batch  <manifest>
manifest: one item per line, TAB-separated: in_path <TAB> out_path <TAB> threshold-repr
stdout (flushed per line):
  R1-BATCH item <i> start
  R1-BATCH item <i> ok elapsed=<s> appended_faces=<n>
exit: 0 all items done · 10 at the first failed item (process ends there)
      2 manifest errors
```

On an item failure the usual `R1-FAIL` stage line goes to stderr, then
`R1-BATCH item <i> fail` on stdout, and the process exits 10 immediately —
fresh-HIP-context discipline: a failed item ends the process, the parent
re-batches the remainder and ladders the one failed chunk through the
untouched single-shot path (whole → halves → unfilled fallback).

Parent side (`generate.py` in the TRELLIS.2 port bundle): the chunk plan
is staged as spool `.npz` files + manifest, one worker process is spawned
per plan, progress lines are parsed and re-emitted as the same per-chunk
log lines the single-shot seam produced, and every output is validated
with the same append-only floor check. A 900 s no-progress watchdog kills
a silent worker; three no-progress rounds abandon batching wholesale and
all remaining chunks take the ladder. Batch mode activates only when both
`R1_FILL_WORKER` and `R1_FILL_BATCH` are set — unset either and behavior
is byte-for-byte the previous path. The fill math does not change:
`fill_logic.cpp` (sha256 9743e000…) and `fill_hip.hip.cpp` (9dcc851a…)
are byte-identical to the single-shot proof runs and are verified by hash
at build time.

## Proof

- CPU build, 40 real corpus chunks (25 calibration + 15 subject,
  stratified) filled single-shot and batched: **40/40 outputs
  byte-identical**. Failure contract: exit 10 at the corrupt item, items
  before it complete, items after untouched.
- Parent integration: a 14,354-chunk plan run three ways — ladder, batch,
  batch with a shim worker dying at item 3 of every round — produced
  identical merged verts, faces, and stats in all three runs.
- In-run gate (job 046): every harness fixture filled through `--batch`
  byte-compared against its single-shot output; a poisoned manifest
  (good / nonexistent / good) exits exactly 10 with the earlier output
  present and the later one absent. The batch flag is exported only after
  this gate and a full calibration + equivalence re-run pass.

## Measured results (job 046, RX 9070 XT, gfx1201)

| Arm | Chunks | Wall | Rate |
|---|---|---|---|
| Subject cascade fill | 386,195 | **1,024.1 s** (17.07 min) | **377.1 chunks/s** wall (387.7 start→last-end) |
| Control arm fill | 373,812 | 998.3 s | 374.5 chunks/s |
| Probe arm fill | 4,348 | 11.9 s | 365.3 chunks/s |

Mean per-chunk elapsed 2.8 ms vs ~473 ms under the per-chunk seam — a
~143× pace improvement, from removing process lifetime from the per-chunk
cost. Job 047 (next day, batch gate re-run in full) reproduced the rate:
386,195 chunks in 1,026.5 s = 376.2 chunks/s.

What batch mode does *not* do: the census/planning for each chunk is still
host-side, per chunk. Moving census/planning onto the device (a
device-side work queue: all chunk data staged in one bulk transfer, an
atomic counter handing kernels the next chunk, one download at the end)
is the designed next stage and is **not** part of this change.
