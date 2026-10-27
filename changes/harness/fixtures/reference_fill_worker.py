#!/usr/bin/env python3
"""reference_fill_worker.py - fixture worker for the R1 harness.

OUR reading of the spec's established behavior (R1 spec section 3.1),
NOT a copy of any CuMesh code and NOT claimed to be the incumbent:

  * boundary edges = undirected edges with exactly one incident face;
  * boundary edges are traced into loops (directed edges following
    the incident face's own corner order);
  * a loop whose perimeter (summed boundary-edge lengths) is STRICTLY
    under max_hole_perimeter is filled;
  * each filled loop gains exactly ONE centroid vertex (arithmetic
    mean of the loop's unique vertices, computed in float64 from the
    float32 inputs and cast back to float32 - spec OPEN-3 is
    unresolved, this is the fixture's stated choice) appended after
    the last input vertex, and ONE fan face per loop edge appended
    after the last input face;
  * fan winding: for a directed boundary edge a->b the fan face is
    (centroid, b, a), i.e. the fan traverses the boundary edge in the
    opposite direction to the incident face, so the fan's orientation
    agrees with the surrounding faces for a consistently oriented
    mesh (spec section 3.2 item 5 is still TO BE READ against the
    incumbent; this is the fixture's stated choice);
  * non-simple boundary components (any vertex whose undirected
    boundary degree is not exactly 2 - branching, figure-8, or the
    fragmentation around a non-manifold edge) are SKIPPED, never
    guessed at: no fill for that component;
  * output is built as input arrays copied verbatim + an appended
    block (the section 2.2 append-only construction rule).

Seam (spec section 2.1): argv = in_path out_path
repr(float(max_hole_perimeter)); input .npz keys verts float32
(K,3) / faces int64 (M,3), chunk-local indexing; exit 0 on success,
nonzero on any failure.

Pure numpy, no other dependency.
"""
import sys

import numpy as np


def compute_fill(verts, faces, max_hole_perimeter):
    """Return (out_verts, out_faces) satisfying the append-only
    invariant: inputs verbatim as the prefix, fills appended."""
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.shape[0] == 0 or verts.shape[0] == 0:
        return verts.copy(), faces.copy()

    # Undirected edge census: boundary = exactly one incident face.
    edge_count = {}
    directed = {}  # undirected key -> directed (a, b) of its sole face
    for a, b, c in faces.tolist():
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_count[key] = edge_count.get(key, 0) + 1
            directed[key] = (u, v)
    boundary_keys = sorted(k for k, n in edge_count.items() if n == 1)
    if not boundary_keys:
        return verts.copy(), faces.copy()

    # Connected components over the boundary graph (union-find),
    # mirroring the parent census's component notion.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    degree = {}
    for u, v in boundary_keys:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    comps = {}
    for key in boundary_keys:
        comps.setdefault(find(key[0]), []).append(key)

    verts64 = verts.astype(np.float64)
    new_verts = []
    new_faces = []
    # Deterministic append order: components by smallest vertex id.
    for root in sorted(comps, key=lambda r: min(min(k) for k in comps[r])):
        keys = comps[root]
        comp_verts = sorted({v for k in keys for v in k})
        # Simple-cycle test only: every vertex degree exactly 2 and
        # one directed successor per vertex. Anything else: skip.
        if any(degree[v] != 2 for v in comp_verts):
            continue
        if len(keys) != len(comp_verts):
            continue
        succ = {}
        ok = True
        for key in keys:
            a, b = directed[key]
            if a in succ:
                ok = False
                break
            succ[a] = b
        if not ok or set(succ) != set(comp_verts):
            continue
        # Trace the loop by directed successors from its smallest id.
        start = comp_verts[0]
        loop = [start]
        cur = succ[start]
        while cur != start:
            loop.append(cur)
            if len(loop) > len(comp_verts):
                ok = False
                break
            cur = succ[cur]
        if not ok or len(loop) != len(comp_verts):
            continue
        # Perimeter = summed directed-edge lengths along the loop.
        perim = 0.0
        for i, a in enumerate(loop):
            b = loop[(i + 1) % len(loop)]
            perim += float(np.linalg.norm(verts64[a] - verts64[b]))
        if not perim < float(max_hole_perimeter):
            continue
        centroid = verts64[np.asarray(loop, dtype=np.int64)].mean(axis=0)
        ci = int(verts.shape[0]) + len(new_verts)
        new_verts.append(centroid.astype(np.float32))
        for i, a in enumerate(loop):
            b = loop[(i + 1) % len(loop)]
            new_faces.append([ci, b, a])

    if not new_verts:
        return verts.copy(), faces.copy()
    out_verts = np.concatenate(
        [verts, np.stack(new_verts).astype(np.float32)], axis=0)
    out_faces = np.concatenate(
        [faces, np.asarray(new_faces, dtype=np.int64)], axis=0)
    return out_verts, out_faces


def main(argv):
    if len(argv) != 4:
        print("usage: reference_fill_worker.py in.npz out.npz "
              "max_hole_perimeter", file=sys.stderr)
        return 2
    in_path, out_path = argv[1], argv[2]
    max_hole_perimeter = float(argv[3])
    d = np.load(in_path)
    out_verts, out_faces = compute_fill(d["verts"], d["faces"],
                                         max_hole_perimeter)
    np.savez(out_path,
             verts=np.asarray(out_verts, dtype=np.float32),
             faces=np.asarray(out_faces, dtype=np.int64))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
