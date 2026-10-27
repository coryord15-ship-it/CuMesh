#!/usr/bin/env python3
"""R1 Phase 1 corpus builder (offline; never touches a live run).

Route for generate.py's planner (binding method, spec section 6):
generate.py (~/workspace/trellis-r33-build/generate.py) is IMPORTED,
not reimplemented. Its needed functions (boundary_census,
plan_fill_chunks, _fill_edge_keys) are pure numpy, but the module
references torch/trellis2 inside other functions, so stub
sys.modules entries for torch / trellis2 (and submodules) are
installed BEFORE import in case any top-level/first-use import
path reaches them. (As read on 2026-10-26, generate.py's top level
imports stdlib only and torch/trellis2 imports are function-local;
the stubs are installed anyway per the build instruction, and the
planner functions exercised here never call them.)

SOURCES (corrected after investigation - see CORPUS.md and
verify_sources.py): the task-named r28-010 GLBs are a wounded
earlier run's dumps (2^20 copy-cliff zero desert; dimensions match
no arm of job 028's log). Job 028's post-fill meshes survive clean
in r32-028/ - canvas under its own name, the subject cascade mesh
in the file named subject-1024control-raw-presimplify.glb. The
extraction meshes the planner faced are recovered by stripping
the appended fills (first-N rows; append-only construction).

Chunks are serialized in the spec section 2.1 worker seam format:
np.savez with keys `verts` float32 (K,3) and `faces` int64 (M,3),
chunk-local indexing (chunk verts = the chunk's referenced global
verts sorted by global id; local index = position in that list) -
exactly generate.py's own _run_fill_subprocess serialization.
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glb_parse import load_glb  # noqa: E402

GEN_PATH = os.path.expanduser("~/workspace/trellis-r33-build/generate.py")
OUT_ROOT = os.path.expanduser(
    "~/workspace/pipeline-rethink-20261026/r1-build/corpus")
MAX_HOLE_PERIMETER = 3e-2  # generate.py _fill_wrapper default (line 1256)

ARMS = {
    "canvas": {"path": os.path.expanduser(
        "~/workspace/trellis-run-20261024/r32-028/"
        "canvas-1024cascade-raw-presimplify.glb"),
        "extraction_verts": 841188, "extraction_faces": 1683908},
    "subject": {"path": os.path.expanduser(
        "~/workspace/trellis-run-20261024/r32-028/"
        "subject-1024control-raw-presimplify.glb"),
        "extraction_verts": 1449764, "extraction_faces": 2858434},
}

TOP_N = 200            # largest-by-faces / largest-by-boundary / cut-heavy
RANDOM_MID_N = 500     # fixed-seed random mid-size chunks (subject)
RANDOM_MID_SEED = 20261026
CANDIDATE_CAP = 2000   # cap on candidate-component chunks (subject)
FILL_NOTHING_N = 500   # fixed-seed sample of no-candidate chunks (subject)
FILL_NOTHING_SEED = 20261027


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_generate():
    """Import generate.py with torch/trellis2 stubs pre-installed."""
    for name in ("torch", "trellis2", "trellis2.modules",
                 "trellis2.modules.image_feature_extractor",
                 "trellis2.representations",
                 "trellis2.representations.mesh",
                 "trellis2.representations.mesh.base"):
        if name not in sys.modules:
            try:
                __import__(name)
            except Exception:
                stub = types.ModuleType(name)
                stub.__path__ = []
                sys.modules[name] = stub
    spec = importlib.util.spec_from_file_location("generate_r33", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chunk_manifest_hash(chunks):
    """SHA-256 over the canonical per-chunk plan (determinism proof).

    Per chunk, in plan order: chunk id, global face rows (int64),
    global vert rows (int64), local faces (int64), fillable_interior
    (bool) - byte-exact inputs to any chunk-manifest.
    """
    h = hashlib.sha256()
    for cid, c in enumerate(chunks):
        h.update(f"chunk {cid}\n".encode())
        h.update(np.ascontiguousarray(c["faces"], dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(c["verts"], dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(c["local_faces"],
                                      dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(c["fillable_interior"],
                                      dtype=bool).tobytes())
    return h.hexdigest()


def plan_stats(gen, verts, faces, chunks, info):
    """Vectorized per-chunk boundary/candidate stats over the full plan."""
    nf = int(faces.shape[0])
    nch = len(chunks)
    chunk_of_face = np.full(nf, -1, dtype=np.int64)
    for cid, c in enumerate(chunks):
        chunk_of_face[c["faces"]] = cid
    assert (chunk_of_face >= 0).all(), "planner left faces unowned"

    faces_per = np.array([len(c["faces"]) for c in chunks], dtype=np.int64)
    verts_per = np.array([len(c["verts"]) for c in chunks], dtype=np.int64)

    # Edge instances (generate.py's own key packing).
    keys = gen._fill_edge_keys(faces).ravel()
    face_of_inst = np.repeat(np.arange(nf, dtype=np.int64), 3)
    chunk_of_inst = chunk_of_face[face_of_inst]

    # Chunk-local boundary edges: (chunk, key) runs of length 1.
    order = np.lexsort((keys, chunk_of_inst))
    skey, sch = keys[order], chunk_of_inst[order]
    run_start = np.flatnonzero(np.concatenate(
        [[True], (skey[1:] != skey[:-1]) | (sch[1:] != sch[:-1])]))
    run_len = np.diff(np.concatenate([run_start, [len(skey)]]))
    is_bnd_run = run_len == 1
    n_boundary = np.bincount(sch[run_start[is_bnd_run]],
                             minlength=nch).astype(np.int64)

    # Global edge counts + cut edges (global count 2, faces in 2 chunks).
    order2 = np.argsort(keys, kind="stable")
    gkey, gch = keys[order2], chunk_of_inst[order2]
    gstart = np.flatnonzero(np.concatenate([[True], gkey[1:] != gkey[:-1]]))
    glen = np.diff(np.concatenate([gstart, [len(gkey)]]))
    gcount_inst = np.empty(len(gkey), dtype=np.int64)
    gcount_inst[order2] = np.repeat(glen, glen)
    n_global_bnd = np.bincount(chunk_of_inst,
                               weights=(gcount_inst == 1).astype(np.float64),
                               minlength=nch).astype(np.int64)
    cut_sorted = np.zeros(len(gkey), dtype=bool)
    two = np.flatnonzero(glen == 2)
    if len(two):
        s2 = gstart[two]
        diff_pair = gch[s2] != gch[s2 + 1]
        cut_sorted[s2[diff_pair]] = True
        cut_sorted[s2[diff_pair] + 1] = True
    cut_inst = np.zeros(len(gkey), dtype=bool)
    cut_inst[order2] = cut_sorted
    n_cut = np.bincount(chunk_of_inst, weights=cut_inst.astype(np.float64),
                        minlength=nch).astype(np.int64)

    # Candidate / under-threshold component membership per chunk.
    census = info["census"]
    vc = census["vertex_component"]
    perim = np.array([c["perimeter"] for c in census["components"]],
                     dtype=np.float64)
    is_cand_vert = np.zeros(int(verts.shape[0]), dtype=bool)
    is_under_vert = np.zeros(int(verts.shape[0]), dtype=bool)
    on = vc >= 0
    if on.any():
        is_cand_vert[on] = perim[vc[on]] <= 2.0 * MAX_HOLE_PERIMETER
        is_under_vert[on] = perim[vc[on]] < MAX_HOLE_PERIMETER
    all_cv = np.concatenate([c["verts"] for c in chunks]) \
        if nch else np.zeros(0, dtype=np.int64)
    rep = np.repeat(np.arange(nch, dtype=np.int64), verts_per)
    n_cand_verts = np.bincount(rep, weights=is_cand_vert[all_cv].astype(
        np.float64), minlength=nch).astype(np.int64)
    n_under_verts = np.bincount(rep, weights=is_under_vert[all_cv].astype(
        np.float64), minlength=nch).astype(np.int64)
    all_fill = np.concatenate([c["fillable_interior"] for c in chunks]) \
        if nch else np.zeros(0, dtype=bool)
    n_fillable = np.bincount(rep, weights=all_fill.astype(np.float64),
                             minlength=nch).astype(np.int64)

    return {
        "faces_per_chunk": faces_per, "verts_per_chunk": verts_per,
        "n_boundary_edges": n_boundary, "n_cut_edges": n_cut,
        "n_global_boundary_edges": n_global_bnd,
        "n_candidate_verts": n_cand_verts,
        "n_under_threshold_verts": n_under_verts,
        "n_fillable_interior_verts": n_fillable,
        "total_boundary_edges_global": int(len(census["boundary_edges"])),
        "total_components": int(len(census["components"])),
    }


def top_ids(metric, n):
    order = sorted(range(len(metric)),
                   key=lambda i: (-int(metric[i]), i))
    return order[:n]


def dist(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        return {"min": 0, "median": 0.0, "max": 0}
    return {"min": int(arr.min()), "median": float(np.median(arr)),
            "max": int(arr.max())}


def serialize(arm, chunks, stats, selected, tags, plan_summary):
    arm_dir = os.path.join(OUT_ROOT, arm)
    os.makedirs(arm_dir, exist_ok=True)
    entries = []
    priority = ["candidate_component", "largest_faces", "largest_boundary",
                "cut_heavy", "random_mid", "fill_nothing_no_candidate",
                "fillable_interior"]
    for cid in sorted(selected):
        c = chunks[cid]
        cv = np.asarray(verts_global[arm][c["verts"]], dtype=np.float32)
        cf = np.asarray(c["local_faces"], dtype=np.int64)
        fname = f"chunk-{cid:06d}.npz"
        np.savez(os.path.join(arm_dir, fname), verts=cv, faces=cf)
        tg = sorted(tags.get(cid, set()))
        primary = next((p for p in priority if p in tg),
                       "standard" if arm == "canvas" else "unstratified")
        entries.append({
            "chunk_id": int(cid), "file": fname,
            "source_arm": arm,
            "source_glb": os.path.basename(ARMS[arm]["path"]),
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
            "strata": tg, "primary_stratum": primary,
        })
    manifest = dict(plan_summary)
    manifest["serialized_chunks"] = entries
    with open(os.path.join(arm_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return entries


verts_global = {}


def main():
    gen = load_generate()
    log(f"generate.py imported; budgets faces={gen.FILL_CHUNK_FACE_BUDGET} "
        f"verts={gen.FILL_CHUNK_VERT_BUDGET}; "
        f"max_hole_perimeter={MAX_HOLE_PERIMETER}")
    summary = {"max_hole_perimeter": MAX_HOLE_PERIMETER,
               "face_budget": gen.FILL_CHUNK_FACE_BUDGET,
               "vert_budget": gen.FILL_CHUNK_VERT_BUDGET, "arms": {}}

    for arm in ("canvas", "subject"):
        path = ARMS[arm]["path"]
        log(f"{arm}: parsing GLB {path}")
        t0 = time.time()
        glb_verts, glb_faces, meta = load_glb(path)
        glb_total = (int(len(glb_verts)), int(len(glb_faces)))
        # The saved GLB is job 028's POST-FILL mesh; chunked_fill_core
        # is append-only ([originals; appended fills]), so the exact
        # extraction mesh the planner faced is the first N rows.
        # Proven by verify_sources.py (head faces reference only head
        # verts; every tail face is a centroid fan).
        ev, ef = ARMS[arm]["extraction_verts"], ARMS[arm]["extraction_faces"]
        verts = np.ascontiguousarray(glb_verts[:ev])
        faces = np.ascontiguousarray(glb_faces[:ef])
        del glb_verts, glb_faces
        verts_global[arm] = verts
        log(f"{arm}: post-fill GLB {glb_total[0]} verts / "
            f"{glb_total[1]} faces -> extraction mesh {len(verts)} verts, "
            f"{len(faces)} faces after stripping appended fills "
            f"({time.time()-t0:.1f}s)")

        t0 = time.time()
        log(f"{arm}: plan_fill_chunks run 1 ...")
        chunks, info = gen.plan_fill_chunks(verts, faces,
                                            MAX_HOLE_PERIMETER)
        log(f"{arm}: planned {len(chunks)} chunks in "
            f"{time.time()-t0:.1f}s; candidate_components="
            f"{info['candidate_components']}, atomic_groups="
            f"{info['atomic_groups']}, dissolved={info['dissolved_groups']}")
        h1 = chunk_manifest_hash(chunks)
        log(f"{arm}: chunk-manifest SHA-256 (run 1) = {h1}")

        det = {"run1_sha256": h1}
        if arm == "canvas":
            t0 = time.time()
            log("canvas: plan_fill_chunks run 2 (determinism) ...")
            chunks2, _info2 = gen.plan_fill_chunks(verts, faces,
                                                   MAX_HOLE_PERIMETER)
            h2 = chunk_manifest_hash(chunks2)
            log(f"canvas: run 2 in {time.time()-t0:.1f}s; "
                f"SHA-256 = {h2}; identical = {h1 == h2}")
            det["run2_sha256"] = h2
            det["identical"] = bool(h1 == h2)
            del chunks2, _info2

        t0 = time.time()
        stats = plan_stats(gen, verts, faces, chunks, info)
        log(f"{arm}: plan stats in {time.time()-t0:.1f}s; census: "
            f"{stats['total_components']} components, "
            f"{stats['total_boundary_edges_global']} boundary edges")

        # --- stratification -------------------------------------------
        tags = {}
        def tag(cid, name):
            tags.setdefault(int(cid), set()).add(name)

        cand_ids = [i for i in range(len(chunks))
                    if stats["n_candidate_verts"][i] > 0]
        for cid in top_ids(stats["faces_per_chunk"], TOP_N):
            tag(cid, "largest_faces")
        for cid in top_ids(stats["n_boundary_edges"], TOP_N):
            tag(cid, "largest_boundary")
        for cid in top_ids(stats["n_cut_edges"], TOP_N):
            tag(cid, "cut_heavy")
        for cid in range(len(chunks)):
            if stats["n_fillable_interior_verts"][cid] > 0:
                tag(cid, "fillable_interior")

        if arm == "canvas":
            for cid in cand_ids:
                tag(cid, "candidate_component")
            for cid in range(len(chunks)):
                if stats["n_candidate_verts"][cid] == 0:
                    tag(cid, "fill_nothing_no_candidate")
            selected = set(range(len(chunks)))
            stratum_counts_total = {
                "candidate_component_chunks_total": len(cand_ids),
                "fill_nothing_no_candidate_chunks_total":
                    int((stats["n_candidate_verts"] == 0).sum()),
            }
        else:
            for cid in cand_ids[:CANDIDATE_CAP]:
                tag(cid, "candidate_component")
            fpc = stats["faces_per_chunk"]
            q25, q75 = np.percentile(fpc, [25, 75])
            mid_pool = np.flatnonzero((fpc >= q25) & (fpc <= q75))
            rng = np.random.default_rng(RANDOM_MID_SEED)
            mid = rng.choice(mid_pool, size=min(RANDOM_MID_N, len(mid_pool)),
                             replace=False)
            for cid in mid:
                tag(int(cid), "random_mid")
            fn_pool = np.flatnonzero(stats["n_candidate_verts"] == 0)
            rng2 = np.random.default_rng(FILL_NOTHING_SEED)
            fn = rng2.choice(fn_pool,
                             size=min(FILL_NOTHING_N, len(fn_pool)),
                             replace=False)
            for cid in fn:
                tag(int(cid), "fill_nothing_no_candidate")
            selected = set()
            for cid, tg in tags.items():
                if tg & {"candidate_component", "largest_faces",
                         "largest_boundary", "cut_heavy", "random_mid",
                         "fill_nothing_no_candidate"}:
                    selected.add(cid)
            stratum_counts_total = {
                "candidate_component_chunks_total": len(cand_ids),
                "candidate_component_serialized_capped":
                    min(len(cand_ids), CANDIDATE_CAP),
                "random_mid_pool_size_faces_between_p25_p75":
                    int(len(mid_pool)),
                "random_mid_p25_faces": float(q25),
                "random_mid_p75_faces": float(q75),
                "fill_nothing_no_candidate_pool_size": int(len(fn_pool)),
            }

        plan_summary = {
            "arm": arm, "source_glb": os.path.basename(path),
            "source_glb_path": path,
            "source_glb_bytes": os.path.getsize(path),
            "glb_postfill_verts": glb_total[0],
            "glb_postfill_faces": glb_total[1],
            "strip_rule": "extraction mesh = first extraction_verts "
                          "vert rows / first extraction_faces face rows "
                          "of the post-fill GLB (chunked_fill_core is "
                          "append-only); see verify_sources.py",
            "mesh_verts": int(len(verts)), "mesh_faces": int(len(faces)),
            "planned_chunks": int(len(chunks)),
            "candidate_components": int(info["candidate_components"]),
            "atomic_groups": int(info["atomic_groups"]),
            "dissolved_groups": int(info["dissolved_groups"]),
            "census_components_total": stats["total_components"],
            "census_boundary_edges_total":
                stats["total_boundary_edges_global"],
            "chunk_faces_distribution": dist(stats["faces_per_chunk"]),
            "chunk_verts_distribution": dist(stats["verts_per_chunk"]),
            "determinism": det,
            "stratum_totals": stratum_counts_total,
            "serialized_chunk_count": int(len(selected)),
        }
        t0 = time.time()
        entries = serialize(arm, chunks, stats, selected, tags,
                            plan_summary)
        log(f"{arm}: serialized {len(entries)} chunks in "
            f"{time.time()-t0:.1f}s")

        tag_counts = {}
        for e in entries:
            for t in e["strata"]:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        summary["arms"][arm] = dict(
            plan_summary,
            serialized_faces_distribution=dist(
                [e["in_faces"] for e in entries]),
            serialized_verts_distribution=dist(
                [e["in_verts"] for e in entries]),
            serialized_tag_counts=tag_counts)
        del chunks, info, stats
        verts_global.pop(arm, None)

    with open(os.path.join(OUT_ROOT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("done; summary written")


if __name__ == "__main__":
    main()
