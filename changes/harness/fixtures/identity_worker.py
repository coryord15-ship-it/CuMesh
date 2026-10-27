#!/usr/bin/env python3
"""identity_worker.py - seam-compatible fixture worker (spec 2.1).

Reads the chunk .npz and writes it back unchanged (verts float32,
faces int64). The output satisfies the section 2.2 append-only
invariant trivially (appended block is empty). It is the correct
output for a chunk with no fillable loop, and the harness's
calibration floor: any check that fails this worker is a harness
bug, not a worker bug.
"""
import sys

import numpy as np


def main(argv):
    if len(argv) != 4:
        print("usage: identity_worker.py in.npz out.npz "
              "max_hole_perimeter", file=sys.stderr)
        return 2
    in_path, out_path = argv[1], argv[2]
    float(argv[3])  # the seam's third argument must parse as a float
    d = np.load(in_path)
    np.savez(out_path,
             verts=np.asarray(d["verts"], dtype=np.float32),
             faces=np.asarray(d["faces"], dtype=np.int64))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
