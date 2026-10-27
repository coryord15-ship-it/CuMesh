#!/usr/bin/env python3
"""corrupt_worker.py - deliberately corrupted fixture worker.

Runs the reference fill (imported from the sibling fixture, so the
ONLY difference from reference_fill_worker.py is the corruption),
then, if at least one vertex was appended, perturbs one coordinate
of the FIRST appended vertex: x += 1.0 (float32).

The output still satisfies every single-worker invariant the
harness checks: shapes grow, the input prefix is bit-exact, the
appended faces are in range, and it is deterministic (the same
+1.0 every run). Nothing in a single run can catch it - by design.
What catches it is the harness's compare mode (spec section 5.1's
gate): the appended-vertex block, and therefore the vetting's kept
new vertices, differ from the reference worker's on the same chunk.
If no vertex is appended the corruption has nothing to touch and
the output equals the reference output.
"""
import importlib.util
import os
import sys

import numpy as np


def _load_reference():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "reference_fill_worker.py")
    spec = importlib.util.spec_from_file_location("reference_fill_worker",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    if len(argv) != 4:
        print("usage: corrupt_worker.py in.npz out.npz "
              "max_hole_perimeter", file=sys.stderr)
        return 2
    in_path, out_path = argv[1], argv[2]
    max_hole_perimeter = float(argv[3])
    ref = _load_reference()
    d = np.load(in_path)
    in_verts = np.asarray(d["verts"], dtype=np.float32)
    out_verts, out_faces = ref.compute_fill(in_verts, d["faces"],
                                            max_hole_perimeter)
    out_verts = np.array(out_verts, dtype=np.float32, copy=True)
    if out_verts.shape[0] > in_verts.shape[0]:
        # THE corruption: one coordinate of one appended vertex.
        out_verts[in_verts.shape[0], 0] = np.float32(
            out_verts[in_verts.shape[0], 0] + np.float32(1.0))
    np.savez(out_path,
             verts=out_verts,
             faces=np.asarray(out_faces, dtype=np.int64))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
