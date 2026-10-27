#!/usr/bin/env python3
"""build_fixtures.py - construct the synthetic fixture chunks.

Each chunk is written as a section 2.1 .npz (verts float32 (K,3),
faces int64 (M,3), chunk-local indexing starting at 0). Expected
outcomes below assume the calibration threshold
max_hole_perimeter = 5.0 (mesh units) used throughout HARNESS.md.

Fixtures:
  closed_cube.npz      8 verts / 12 faces, a closed unit cube:
                       no boundary edges at all. Identity and
                       reference both append 0 verts / 0 faces.
  small_hole.npz       8 verts / 8 faces, a flat square annulus:
                       outer square side 4 (boundary loop perimeter
                       16, over threshold), inner square hole side 1
                       (loop perimeter 4, under threshold).
                       Reference fill appends exactly 1 vert (the
                       hole centroid (0,0,0)) + 4 fan faces; the
                       parent vetting keeps all 4 faces / 1 vert,
                       drops 0. Identity appends nothing.
  big_hole.npz         8 verts / 8 faces, same construction: outer
                       square side 6 (perimeter 24), inner hole side
                       2 (perimeter 8). Both loops are over the
                       5.0 threshold: reference fill appends
                       0 verts / 0 faces.
  nonmanifold_edge.npz 5 verts / 3 faces: three triangles sharing
                       the common edge (0,1), which therefore has 3
                       incident faces and is NOT a boundary edge
                       (boundary = exactly one incident face). The
                       remaining boundary graph branches at verts 0
                       and 1 (degree 3), so it is not a simple loop:
                       reference fill skips it and appends nothing.
                       The harness's invariant checks still PASS -
                       a non-manifold chunk is hostile input, not a
                       worker error.
  duplicate_faces.npz  3 verts / 2 faces: the same triangle twice.
                       Every edge then has exactly 2 incident faces,
                       so there are no boundary edges: reference
                       fill appends nothing; invariants PASS.

Run: python3 build_fixtures.py [output_dir]  (default: ./chunks
beside this file). Deterministic - no randomness anywhere.
"""
import os
import sys

import numpy as np


def _save(out_dir, name, verts, faces):
    path = os.path.join(out_dir, name)
    np.savez(path,
             verts=np.asarray(verts, dtype=np.float32),
             faces=np.asarray(faces, dtype=np.int64))
    print(f"wrote {path}: {len(verts)} verts, {len(faces)} faces")


def _annulus(half_outer, half_inner):
    """Flat square frame in z=0: outer corners 0..3 (CCW from
    (-,-)), inner corners 4..7 (CCW from (-,-)); each side quad is
    two CCW triangles. Returns (verts, faces)."""
    ho, hi = float(half_outer), float(half_inner)
    verts = [[-ho, -ho, 0.0], [ho, -ho, 0.0], [ho, ho, 0.0],
             [-ho, ho, 0.0],
             [-hi, -hi, 0.0], [hi, -hi, 0.0], [hi, hi, 0.0],
             [-hi, hi, 0.0]]
    faces = [
        [0, 1, 5], [0, 5, 4],   # bottom side quad
        [1, 2, 6], [1, 6, 5],   # right side quad
        [2, 3, 7], [2, 7, 6],   # top side quad
        [3, 0, 4], [3, 4, 7],   # left side quad
    ]
    return verts, faces


def main(argv):
    out_dir = argv[1] if len(argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "chunks")
    os.makedirs(out_dir, exist_ok=True)

    # Closed unit cube, faces wound outward.
    cube_v = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
              [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    cube_f = [[0, 2, 1], [0, 3, 2],   # bottom (z=0, normal -z)
              [4, 5, 6], [4, 6, 7],   # top (z=1, normal +z)
              [0, 1, 5], [0, 5, 4],   # front (y=0)
              [1, 2, 6], [1, 6, 5],   # right (x=1)
              [2, 3, 7], [2, 7, 6],   # back (y=1)
              [3, 0, 4], [3, 4, 7]]   # left (x=0)
    _save(out_dir, "closed_cube.npz", cube_v, cube_f)

    v, f = _annulus(half_outer=2.0, half_inner=0.5)   # hole side 1
    _save(out_dir, "small_hole.npz", v, f)

    v, f = _annulus(half_outer=3.0, half_inner=1.0)   # hole side 2
    _save(out_dir, "big_hole.npz", v, f)

    nm_v = [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0, 1],
            [0.5, -1, 0]]
    nm_f = [[0, 1, 2], [1, 0, 3], [0, 1, 4]]  # edge (0,1) x3 faces
    _save(out_dir, "nonmanifold_edge.npz", nm_v, nm_f)

    _save(out_dir, "duplicate_faces.npz",
          [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
          [[0, 1, 2], [0, 1, 2]])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
