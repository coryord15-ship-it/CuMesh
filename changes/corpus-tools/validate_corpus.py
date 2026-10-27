#!/usr/bin/env python3
"""Validate the serialized R1 corpus against its manifests.

Per .npz: keys exactly {verts, faces}; verts float32 (K,3);
faces int64 (M,3); K/M match the manifest entry; local indices in
[0, K); every vert row referenced or not is allowed (chunk verts
are the referenced set by construction - checked: the set of
referenced local indices must be all of [0,K) since chunk verts
are exactly the referenced verts). Per arm: file count matches
the manifest; canvas (fully serialized) face total must equal the
extraction face count.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser(
    "~/workspace/pipeline-rethink-20261026/r1-build/corpus")
EXPECT_FACES = {"canvas": 1683908}

bad = 0
for arm in ("canvas", "subject"):
    arm_dir = os.path.join(ROOT, arm)
    with open(os.path.join(arm_dir, "manifest.json")) as f:
        manifest = json.load(f)
    entries = manifest["serialized_chunks"]
    files = sorted(fn for fn in os.listdir(arm_dir) if fn.endswith(".npz"))
    assert len(files) == len(entries), (arm, len(files), len(entries))
    total_faces = 0
    total_bytes = 0
    for e in entries:
        p = os.path.join(arm_dir, e["file"])
        total_bytes += os.path.getsize(p)
        d = np.load(p)
        if set(d.files) != {"verts", "faces"}:
            print("BAD keys", p, d.files); bad += 1; continue
        v, f = d["verts"], d["faces"]
        ok = (v.dtype == np.float32 and v.ndim == 2 and v.shape[1] == 3
              and f.dtype == np.int64 and f.ndim == 2 and f.shape[1] == 3
              and len(v) == e["in_verts"] and len(f) == e["in_faces"])
        if ok and len(f):
            ok = int(f.min()) >= 0 and int(f.max()) < len(v) and \
                len(np.unique(f)) == len(v)
        if not ok:
            print("BAD chunk", p, v.shape, v.dtype, f.shape, f.dtype)
            bad += 1
        total_faces += len(f)
    print(f"{arm}: {len(entries)} chunks validated, "
          f"npz bytes={total_bytes}, serialized faces total={total_faces}")
    if arm in EXPECT_FACES:
        assert total_faces == EXPECT_FACES[arm], \
            f"{arm} face coverage {total_faces} != {EXPECT_FACES[arm]}"
        print(f"{arm}: face coverage == extraction faces "
              f"({EXPECT_FACES[arm]}) - full plan coverage confirmed")
print("BAD chunks:", bad)
sys.exit(1 if bad else 0)
