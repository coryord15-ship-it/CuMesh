#!/usr/bin/env python3
"""Add job 028's code-9 fallback chunks to the subject corpus.

OPEN-6 resolution (partial): 028's log (r32-028/inside-utf8.log)
names two fallback triples under two subject fills. The triple
{3540, 3697, 4278} (lines 19899-19935) immediately precedes the
graded subject-cascade [r32-fill] summary (line 19953: 373,812
chunks, 3 fallback chunks) and is paired with it by line order.
The earlier triple {3723, 3728, 4276} belongs to the other subject
fill in the log (the 1024 arm, different mesh, different plan -
chunk ids are not transferable) and is NOT added.

Since the planner reproduces 028's plan exactly (counts + canvas
determinism proof), chunk ids here are the log's chunk ids.
Re-plans the subject arm, serializes the three chunks with the
'code9_fallback_028' stratum tag, and updates manifest + summary.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_corpus as bc  # noqa: E402
from glb_parse import load_glb  # noqa: E402

IDS = (3540, 3697, 4278)
ARM = "subject"


def main():
    gen = bc.load_generate()
    path = bc.ARMS[ARM]["path"]
    glb_v, glb_f, _ = load_glb(path)
    ev, ef = bc.ARMS[ARM]["extraction_verts"], bc.ARMS[ARM]["extraction_faces"]
    verts = np.ascontiguousarray(glb_v[:ev])
    faces = np.ascontiguousarray(glb_f[:ef])
    del glb_v, glb_f
    chunks, info = gen.plan_fill_chunks(verts, faces, bc.MAX_HOLE_PERIMETER)
    assert len(chunks) == 373812, len(chunks)
    h = bc.chunk_manifest_hash(chunks)
    print(f"re-plan: {len(chunks)} chunks, manifest hash {h}")
    stats = bc.plan_stats(gen, verts, faces, chunks, info)

    arm_dir = os.path.join(bc.OUT_ROOT, ARM)
    with open(os.path.join(arm_dir, "manifest.json")) as f:
        manifest = json.load(f)
    have = {e["chunk_id"] for e in manifest["serialized_chunks"]}
    for cid in IDS:
        assert cid not in have
        c = chunks[cid]
        cv = np.asarray(verts[c["verts"]], dtype=np.float32)
        cf = np.asarray(c["local_faces"], dtype=np.int64)
        fname = f"chunk-{cid:06d}.npz"
        np.savez(os.path.join(arm_dir, fname), verts=cv, faces=cf)
        manifest["serialized_chunks"].append({
            "chunk_id": int(cid), "file": fname, "source_arm": ARM,
            "source_glb": os.path.basename(path),
            "in_verts": int(len(c["verts"])),
            "in_faces": int(len(c["faces"])),
            "boundary_stats": {
                "n_boundary_edges_chunklocal":
                    int(stats["n_boundary_edges"][cid]),
                "n_cut_edges": int(stats["n_cut_edges"][cid]),
                "n_global_boundary_edges":
                    int(stats["n_global_boundary_edges"][cid]),
                "n_candidate_component_verts":
                    int(stats["n_candidate_verts"][cid]),
                "n_under_threshold_component_verts":
                    int(stats["n_under_threshold_verts"][cid]),
                "n_fillable_interior_verts":
                    int(stats["n_fillable_interior_verts"][cid]),
            },
            "contains_candidate_component":
                bool(stats["n_candidate_verts"][cid] > 0),
            "strata": ["code9_fallback_028"],
            "primary_stratum": "code9_fallback_028",
        })
        print(f"added chunk {cid}: {len(c['faces'])} faces, "
              f"{len(c['verts'])} verts, "
              f"boundary={int(stats['n_boundary_edges'][cid])}, "
              f"cut={int(stats['n_cut_edges'][cid])}")
    manifest["serialized_chunks"].sort(key=lambda e: e["chunk_id"])
    manifest["serialized_chunk_count"] = len(manifest["serialized_chunks"])
    with open(os.path.join(arm_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    spath = os.path.join(bc.OUT_ROOT, "summary.json")
    with open(spath) as f:
        summary = json.load(f)
    sa = summary["arms"][ARM]
    sa["serialized_chunk_count"] = len(manifest["serialized_chunks"])
    sa["serialized_tag_counts"]["code9_fallback_028"] = len(IDS)
    entries = manifest["serialized_chunks"]
    sa["serialized_faces_distribution"] = bc.dist(
        [e["in_faces"] for e in entries])
    sa["serialized_verts_distribution"] = bc.dist(
        [e["in_verts"] for e in entries])
    with open(spath, "w") as f:
        json.dump(summary, f, indent=1)
    print("manifest + summary updated:",
          len(manifest["serialized_chunks"]), "subject chunks total")


if __name__ == "__main__":
    main()
