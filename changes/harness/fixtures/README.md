# Fixture workers and chunks — R1 harness calibration set

All workers honour the spec §2.1 seam exactly: argv
`in_path out_path repr(float(max_hole_perimeter))`, input/output
.npz with `verts` float32 (K,3) / `faces` int64 (M,3), chunk-local
indexing, exit 0 on success only. Pure numpy + stdlib.

All expected outcomes below are stated for the calibration
threshold **max_hole_perimeter = 5.0** used throughout
`../../HARNESS.md`, and were confirmed by the recorded calibration
runs (`../calibration/*.json`).

## Workers

| Worker | Behavior |
|---|---|
| `identity_worker.py` | Writes the input back unchanged. Satisfies §2.2 trivially; the correct output for any chunk with no fillable loop. |
| `reference_fill_worker.py` | **Our reading** of the spec §3.1 established behavior (labeled as such — not the incumbent, no incumbent code consulted): boundary edges = exactly one incident face; directed loop tracing; strict perimeter `< max_hole_perimeter`; one centroid (float64 mean of the loop's unique verts, cast to float32) + one fan face per loop edge, fan wound `(centroid, b, a)` for directed edge `a→b`; non-simple (branching) boundary components are skipped, never guessed. Deterministic append order (components by smallest vertex id). |
| `corrupt_worker.py` | The reference fill, then `x += 1.0` on the **first appended vertex**. Passes every single-worker invariant; detectable only by compare mode (§5.1). No-op on chunks where nothing is appended. |

## Chunks (built by `build_fixtures.py`, deterministic)

| Chunk | Construction (known truth) | Expected outcome at perim 5.0 |
|---|---|---|
| `closed_cube.npz` | 8 verts / 12 faces, closed unit cube, **no boundary edges** | identity → PASS, 0 appends; reference → PASS, 0 appends; vetting kept 0 / dropped 0 |
| `small_hole.npz` | 8 verts / 8 faces, flat square annulus: outer loop side 4 (perimeter **16**, over), inner hole side 1 (perimeter **4**, under) | reference → PASS, appends **exactly 1 vert** `(0,0,0)` + **4 fan faces** `[[8,7,4],[8,6,7],[8,5,6],[8,4,5]]`; vetting keeps 4 faces / 1 vert, drops 0; identity appends nothing |
| `big_hole.npz` | 8 verts / 8 faces, same annulus: outer side 6 (perimeter 24), inner hole side 2 (perimeter **8**, over) | reference → PASS, **0 appends** (hole stays unfilled); vetting kept 0 / dropped 0 |
| `nonmanifold_edge.npz` | 5 verts / 3 faces; edge (0,1) has **3 incident faces** (not a boundary edge by the exactly-one rule); remaining boundary graph branches (degree 3 at verts 0 and 1) | reference → PASS (invariants), 0 appends — branching component skipped; vetting kept 0 |
| `duplicate_faces.npz` | 3 verts / 2 faces, the same triangle twice → every edge has 2 incident faces, **no boundary** | reference → PASS, 0 appends; vetting kept 0 |

Rebuild the chunks any time with `python3 build_fixtures.py`
(deterministic construction — same arrays, same dtypes).
