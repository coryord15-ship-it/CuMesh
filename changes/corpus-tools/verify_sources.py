#!/usr/bin/env python3
"""Source-mesh investigation for the R1 corpus (run 2026-10-26).

Findings this script reproduces:
1. The task-named r28-010 GLBs do NOT contain job 028's meshes:
   they carry the 2^20 copy-cliff wound (all-zero face rows in a
   contiguous desert starting at exactly row 1,048,576) and their
   dimensions match no arm of job 028's log
   (r32-028/inside-utf8.log).
2. Job 028's post-fill meshes survive clean in r32-028/:
   canvas in canvas-1024cascade-raw-presimplify.glb (843,053 verts
   / 1,694,825 faces = extraction 841,188 / 1,683,908 + kept fills
   1,865 / 10,917) and the subject cascade post-fill mesh in the
   file NAMED subject-1024control-raw-presimplify.glb (1,480,211
   verts / 3,033,637 faces = extraction 1,449,764 / 2,858,434 +
   kept fills 30,447 / 175,203). The file named
   subject-1024cascade in r32-028 is a wounded readout of the
   subject 1024 (non-cascade) arm's array (zero desert
   1,048,576..3,034,499).
3. chunked_fill_core is append-only ([originals; appended fills]),
   so the exact extraction meshes the planner faced are the first
   N rows of the post-fill GLBs: head faces reference only head
   verts, and every tail face is a centroid fan (exactly one of
   its three indices points at an appended vert).
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from glb_parse import load_glb  # noqa: E402

R28 = "/home/hatch/workspace/trellis-run-20261024/r28-010/"
R32 = "/home/hatch/workspace/trellis-run-20261024/r32-028/"


def report(tag, path):
    v, f, _ = load_glb(path)
    z = np.flatnonzero((f == 0).all(axis=1))
    span = (int(z.min()), int(z.max())) if len(z) else None
    print(f"{tag}: verts={len(v)} faces={len(f)} "
          f"all-zero face rows={len(z)} span={span}")
    return v, f


print("== task-named sources (r28-010) ==")
report("r28-010 canvas", R28 + "canvas-1024cascade-raw-presimplify.glb")
report("r28-010 subject", R28 + "subject-1024cascade-raw-presimplify.glb")

print("== job 028 folder (r32-028) ==")
report("r32-028 subject-1024cascade (wounded, wrong arm)",
       R32 + "subject-1024cascade-raw-presimplify.glb")

for tag, path, ev, ef, kf, kv in (
    ("canvas", R32 + "canvas-1024cascade-raw-presimplify.glb",
     841188, 1683908, 10917, 1865),
    ("subject", R32 + "subject-1024control-raw-presimplify.glb",
     1449764, 2858434, 175203, 30447),
):
    v, f = report(f"r32-028 {tag} post-fill", path)
    assert len(v) == ev + kv and len(f) == ef + kf, "dimension mismatch"
    tail = f[ef:]
    assert int(f[:ef].max()) < ev, "head face references appended vert"
    frac = float((tail >= ev).mean())
    assert not (tail >= ev + kv).any(), "tail references past appended"
    print(f"  strip check: head={ef} faces on {ev} verts; tail={len(tail)} "
          f"fans (expected {kf}), appended-vert index share={frac:.4f} "
          f"(centroid fans = 1/3), tail verts={len(v)-ev} (expected {kv})")
print("all strip checks passed")
