#!/usr/bin/env python3
"""run_harness.py - standalone correctness harness for R1 hole-fill
chunk workers (R1 spec section 6 Phase 2 / gate section 5.1).

WHAT IT DOES
Feeds a chunk (.npz, spec section 2.1 format) to a worker through
the REAL worker seam - a real subprocess whose argv is
[worker..., in_path, out_path, repr(float(max_hole_perimeter))],
with the chunk round-tripped through a real .npz file on disk -
then checks the output against the section 2.2 append-only
invariant and runs the PARENT'S OWN vetting on it.

CHECKS (per chunk, "check" mode)
  (a) SHAPE  - output rows >= input rows for verts and faces, and
               both outputs are (N, 3) arrays;
  (b) PREFIX - the input prefix comes back bit-exact, in order:
               np.array_equal (float32 equality, NO tolerance) on
               the verts prefix AND the faces prefix;
  (c) INDEX  - every appended face index is in range
               [0, out_verts rows): appended faces may reference
               input verts + appended verts only, and those are
               exactly the output's rows;
  (d) DETERMINISM - the worker is run TWICE on the same chunk; the
               two outputs must be bit-identical;
  (e) VETTING - the parent's real extract_chunk_fills is run on the
               output and the verdict recorded: kept faces (as
               arrays), kept new verts (as arrays), dropped count
               (= appended faces - kept faces). If it returns None
               the chunk would fall back to unfilled in the parent;
               that is recorded, and the invariant failure that
               caused it is already in the FAIL list.

FAIL CLASSES (a chunk verdict is FAIL if any is present)
  PREFIX-VIOLATION   SHAPE-VIOLATION   INDEX-VIOLATION
  NONDETERMINISM     WORKER-FAILURE (nonzero exit / signal /
                     timeout / missing or unreadable output)
  EQUIVALENCE-VIOLATION (compare mode only, below)

COMPARE MODE (the section 5.1 gate, in miniature)
Runs check on TWO workers over the same chunk and diffs their
canonical (first-run) outputs: the appended verts arrays, the
appended faces arrays, and the vetting verdicts (kept faces, kept
new verts, dropped count) must ALL be exactly equal. This is the
mode that catches a worker whose appended block is internally
consistent but wrong - e.g. fixtures/corrupt_worker.py, whose
single-run checks all PASS. A mismatch is an
EQUIVALENCE-VIOLATION and the JSON names exactly which fields
differed (appended_verts / appended_faces / kept_faces /
kept_new_verts / dropped_count).

PARENT CODE REUSE
extract_chunk_fills, boundary_census and plan_fill_chunks are the
parent's, from ~/workspace/trellis-r33-build/generate.py (override
with --generate-py). They are loaded by IMPORT-WITH-STUBS: torch /
trellis2 are not installed on this VM, so empty stub modules are
placed in sys.modules first (generate.py only imports them inside
functions this harness never calls; its top level is stdlib-only),
then the file is imported as a module. If that import ever fails,
the FALLBACK is extract-and-exec: the three functions' source (plus
_fill_edge_keys, which the census needs) is sliced out of the file
text by AST line spans and exec'd - still the parent's code, never
a rewrite. Every emitted JSON records which route was used in its
"parent_code_route" field.

THE STANDALONE-CHUNK VETTING CONTEXT (honest approximation)
extract_chunk_fills needs fillable_interior: a per-vertex bool,
"vertex of ONE global boundary component whose perimeter is under
max_hole_perimeter AND whose incident faces are ALL inside this
chunk" (spec section 2.5). A standalone harness chunk cannot know
the source mesh it was cut from, so the harness constructs the
context from the chunk itself, using the parent's boundary_census
on the chunk:
  * component perimeters are the IN-CHUNK component perimeters;
  * the "all incident faces inside the chunk" clause is taken as
    true for every vertex, because every face in the input IS in
    the chunk by construction.
What this cannot provide, stated plainly: (1) a vertex whose other
incident faces live in a DIFFERENT chunk of the real mesh would be
non-fillable in the real run but reads fillable here; (2) a global
component that the partition cut would have a larger true
perimeter than its in-chunk part shows. For a chunk that IS a
whole mesh (all fixtures, and any future whole-mesh corpus chunk)
the context is exact, not approximate. For equivalence comparison
the approximation is safe regardless: both workers are vetted
against the IDENTICAL context, so a verdict difference can only
come from the workers.

USAGE
  python3 run_harness.py check   --worker "python3 /path/worker.py" \
      (--chunk f.npz | --corpus dir) --max-hole-perimeter 5.0 \
      [--timeout 900] [--out verdict.json]
  python3 run_harness.py compare --worker-a "..." --worker-b "..." \
      (--chunk f.npz | --corpus dir) --max-hole-perimeter 5.0 \
      [--timeout 900] [--out verdict.json]

The worker command is a shell-like string (shlex-split) or, if it
begins with '[', a JSON argv list. The full JSON verdict (per-chunk
results + summary) is printed to stdout and, with --out, written
to that file. Exit code: 0 iff every chunk verdict is PASS.

Pure stdlib + numpy 1.26.4.
"""
import argparse
import ast
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import types

import numpy as np

DEFAULT_GENERATE_PY = os.path.expanduser(
    "~/workspace/trellis-r33-build/generate.py")
DEFAULT_TIMEOUT_S = 900  # spec section 2.1: the parent's per-attempt value

FAIL_PREFIX = "PREFIX-VIOLATION"
FAIL_SHAPE = "SHAPE-VIOLATION"
FAIL_INDEX = "INDEX-VIOLATION"
FAIL_NONDET = "NONDETERMINISM"
FAIL_WORKER = "WORKER-FAILURE"
FAIL_EQUIV = "EQUIVALENCE-VIOLATION"


# ---------------------------------------------------------------------
# Parent-code loading: the parent's functions, never rewrites.
# ---------------------------------------------------------------------

def _install_stubs():
    """Place empty stub modules in sys.modules for packages that
    generate.py imports lazily but which are not installed here
    (torch, trellis2). Existing real modules are never replaced."""
    for name in ("torch", "trellis2",
                 "trellis2.representations",
                 "trellis2.representations.mesh",
                 "trellis2.representations.mesh.base"):
        if name in sys.modules:
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue  # really installed - leave it alone
        except (ImportError, ValueError):
            pass
        mod = types.ModuleType(name)
        mod.__path__ = []  # mark as a package so submodule imports parse
        sys.modules[name] = mod
    # The one attribute generate.py's lazily-imported paths name at
    # class level in some branches; harmless on the stub.
    if not hasattr(sys.modules["trellis2.representations.mesh.base"],
                   "Mesh"):
        sys.modules["trellis2.representations.mesh.base"].Mesh = object


def load_parent_functions(generate_py):
    """Return (funcs_dict, route) where funcs_dict holds the parent's
    extract_chunk_fills, boundary_census and plan_fill_chunks.
    Route 1: import-with-stubs. Route 2 (fallback): extract-and-exec
    of the exact function definitions from the file text."""
    _install_stubs()
    try:
        spec = importlib.util.spec_from_file_location(
            "generate_r33_parent", generate_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        funcs = {name: getattr(mod, name) for name in
                 ("extract_chunk_fills", "boundary_census",
                  "plan_fill_chunks")}
        return funcs, "import-with-stubs"
    except Exception:
        pass  # fall through to the extract-and-exec route

    with open(generate_py, "r", encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    wanted = {"_fill_edge_keys", "boundary_census", "plan_fill_chunks",
              "extract_chunk_fills"}
    lines = src.splitlines(keepends=True)
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            start = node.lineno - 1
            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list) - 1
            chunks.append("".join(lines[start:node.end_lineno]))
    missing = wanted - {n.name for n in tree.body
                        if isinstance(n, ast.FunctionDef)}
    if missing:
        raise RuntimeError(
            f"could not extract {sorted(missing)} from {generate_py}")
    # The extracted functions import numpy internally per call and
    # otherwise need only builtins; exec them in a clean namespace.
    namespace = {}
    exec("\n\n".join(chunks), namespace)
    funcs = {name: namespace[name] for name in
             ("extract_chunk_fills", "boundary_census",
              "plan_fill_chunks")}
    return funcs, "extract-and-exec-fallback"


# ---------------------------------------------------------------------
# Worker invocation: the real section 2.1 seam.
# ---------------------------------------------------------------------

def parse_worker_cmd(text):
    """Worker command: a JSON argv list if it starts with '[',
    otherwise a shlex-split shell-like string."""
    text = text.strip()
    if text.startswith("["):
        return [str(x) for x in json.loads(text)]
    return shlex.split(text)


def run_worker_once(worker_argv, in_verts, in_faces, max_hole_perimeter,
                    timeout_s, work_root):
    """Run the worker ONE time through the real seam: the chunk is
    written to a fresh in.npz (verts float32 / faces int64, exactly
    as the parent writes it), the subprocess is invoked as
    [worker..., in_path, out_path, repr(float(max_hole_perimeter))],
    and the output .npz is loaded back. Returns a dict:
    {ok, returncode, elapsed_s, error, out_verts, out_faces} where
    the arrays are present only when ok. Never raises for a worker
    failure - failures are data (the FAIL class), not exceptions."""
    tmp = tempfile.mkdtemp(prefix="r1-harness-", dir=work_root)
    in_path = os.path.join(tmp, "in.npz")
    out_path = os.path.join(tmp, "out.npz")
    np.savez(in_path,
             verts=np.asarray(in_verts, dtype=np.float32),
             faces=np.asarray(in_faces, dtype=np.int64))
    cmd = list(worker_argv) + [in_path, out_path,
                               repr(float(max_hole_perimeter))]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, timeout=timeout_s,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None,
                "elapsed_s": time.time() - t0,
                "error": f"timeout after {timeout_s}s"}
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"ok": False, "returncode": proc.returncode,
                "elapsed_s": elapsed,
                "error": f"worker exited {proc.returncode}",
                "stderr_tail": proc.stderr[-500:]}
    if not os.path.isfile(out_path):
        return {"ok": False, "returncode": proc.returncode,
                "elapsed_s": elapsed,
                "error": "worker exited 0 but wrote no output file"}
    try:
        d = np.load(out_path)
        out_verts = np.asarray(d["verts"])
        out_faces = np.asarray(d["faces"])
    except Exception as exc:
        return {"ok": False, "returncode": proc.returncode,
                "elapsed_s": elapsed,
                "error": f"output .npz unreadable: {exc}"}
    return {"ok": True, "returncode": 0, "elapsed_s": elapsed,
            "error": None, "out_verts": out_verts,
            "out_faces": out_faces}


# ---------------------------------------------------------------------
# The in-chunk vetting context (see the module docstring for exactly
# what this can and cannot claim).
# ---------------------------------------------------------------------

def build_vetting_context(in_verts, in_faces, max_hole_perimeter, funcs):
    """fillable_interior (bool (K,)) for a standalone chunk, built
    with the parent's boundary_census on the chunk itself: a vertex
    is fillable iff it sits on an in-chunk boundary component whose
    (in-chunk) perimeter is STRICTLY under max_hole_perimeter - the
    planner's own test (plan_fill_chunks: components' perimeters
    compared with '<'). The planner's second clause (every incident
    face inside the chunk) is true by construction here.
    Also returns a JSON-able census summary for the record."""
    census = funcs["boundary_census"](in_verts, in_faces)
    nv = int(np.asarray(in_verts).shape[0])
    fillable = np.zeros(nv, dtype=bool)
    vcomp = census["vertex_component"]
    comp_summaries = []
    for ci, comp in enumerate(census["components"]):
        perim = float(comp["perimeter"])
        comp_summaries.append({
            "component": ci,
            "verts": [int(v) for v in comp["verts"].tolist()],
            "perimeter": perim,
            "under_threshold": bool(perim < float(max_hole_perimeter)),
        })
        if perim < float(max_hole_perimeter):
            members = np.flatnonzero(vcomp == ci)
            fillable[members] = True
    return fillable, {
        "source": "boundary_census(parent) on the chunk itself",
        "boundary_edges": int(census["boundary_edges"].shape[0]),
        "components": comp_summaries,
        "fillable_interior_count": int(fillable.sum()),
        "approximation": (
            "in-chunk context: component perimeters are in-chunk "
            "only, and the all-incident-faces-inside-the-chunk "
            "clause is taken as true (every input face IS in the "
            "chunk). Exact for whole-mesh chunks; identical for "
            "both workers in compare mode."),
    }


def run_vetting(funcs, in_verts, in_faces, out_verts, out_faces,
                fillable):
    """Run the parent's real extract_chunk_fills and record its
    verdict. Returns a JSON-able dict. kept faces / new verts are
    recorded as arrays (lists) so compare mode can diff them
    exactly, not just by count."""
    appended_faces = max(0, int(np.asarray(out_faces).shape[0])
                         - int(np.asarray(in_faces).shape[0]))
    try:
        res = funcs["extract_chunk_fills"](in_verts, in_faces,
                                           out_verts, out_faces,
                                           fillable)
    except Exception as exc:
        # extract_chunk_fills trusts indices to be in range; an
        # output with out-of-range appended faces (already flagged
        # INDEX-VIOLATION by check (c)) makes it raise while
        # indexing the appended verts. The harness must still emit
        # its verdict, so the exception is recorded, not propagated.
        return {"returned_none": None, "error": f"{type(exc).__name__}: "
                f"{exc}", "kept_faces": [], "kept_faces_count": None,
                "kept_new_verts": [], "kept_new_verts_count": None,
                "dropped_faces_count": None}
    if res is None:
        return {"returned_none": True, "kept_faces": [],
                "kept_faces_count": 0, "kept_new_verts": [],
                "kept_new_verts_count": 0,
                "dropped_faces_count": appended_faces}
    kept_faces, new_verts = res
    return {"returned_none": False,
            "kept_faces": np.asarray(kept_faces).tolist(),
            "kept_faces_count": int(len(kept_faces)),
            "kept_new_verts":
                np.asarray(new_verts, dtype=np.float32).tolist(),
            "kept_new_verts_count": int(len(new_verts)),
            "dropped_faces_count": appended_faces - int(len(kept_faces))}


# ---------------------------------------------------------------------
# Per-chunk check: one worker, two runs, all invariant checks.
# ---------------------------------------------------------------------

def check_one_chunk(funcs, route, chunk_label, in_verts, in_faces,
                    worker_argv, max_hole_perimeter, timeout_s,
                    work_root):
    """The full check battery for one (worker, chunk) pair.
    Returns (verdict_dict, canonical_output) where the canonical
    output is (out_verts, out_faces) of run 1, or None if the
    worker failed. The verdict dict is JSON-able end to end."""
    nv_in = int(in_verts.shape[0])
    nf_in = int(in_faces.shape[0])
    verdict = {
        "chunk": chunk_label,
        "worker_cmd": list(worker_argv),
        "max_hole_perimeter": float(max_hole_perimeter),
        "timeout_s": timeout_s,
        "parent_code_route": route,
        "input": {"verts": nv_in, "faces": nf_in},
        "runs": [],
        "failures": [],
        "checks": {},
        "appended_verts_count": None,
        "appended_faces_count": None,
        "appended_verts": None,
        "appended_faces": None,
        "vetting_context": None,
        "vetting": None,
    }

    def fail(cls):
        if cls not in verdict["failures"]:
            verdict["failures"].append(cls)

    # Two runs: determinism is a first-class check, not a rerun on
    # suspicion. A worker failure on run 1 ends the battery (run 2
    # would fail the same way and add no information but time).
    runs = []
    for _ in range(2):
        r = run_worker_once(worker_argv, in_verts, in_faces,
                            max_hole_perimeter, timeout_s, work_root)
        runs.append(r)
        verdict["runs"].append({
            "ok": bool(r["ok"]),
            "returncode": r["returncode"],
            "elapsed_s": round(float(r["elapsed_s"]), 4),
            "error": r["error"],
        })
        if not r["ok"]:
            break
    if not runs[0]["ok"]:
        fail(FAIL_WORKER)
        verdict["status"] = "FAIL"
        return verdict, None
    out_v, out_f = runs[0]["out_verts"], runs[0]["out_faces"]

    # (a) Shape: 2-D (N,3) and no smaller than the input.
    shape_ok = (out_v.ndim == 2 and out_v.shape[1] == 3
                and out_f.ndim == 2 and out_f.shape[1] == 3
                and out_v.shape[0] >= nv_in and out_f.shape[0] >= nf_in)
    verdict["checks"]["shape_ge_input"] = bool(shape_ok)
    if not shape_ok:
        fail(FAIL_SHAPE)
    verdict["appended_verts_count"] = int(out_v.shape[0]) - nv_in \
        if out_v.ndim == 2 else None
    verdict["appended_faces_count"] = int(out_f.shape[0]) - nf_in \
        if out_f.ndim == 2 else None

    # (b) Prefix bit-exactness (only meaningful at sane shapes).
    prefix_ok = None
    if out_v.ndim == 2 and out_f.ndim == 2 \
            and out_v.shape[0] >= nv_in and out_f.shape[0] >= nf_in:
        prefix_ok = bool(np.array_equal(out_v[:nv_in], in_verts)
                         and np.array_equal(out_f[:nf_in], in_faces))
        if not prefix_ok:
            fail(FAIL_PREFIX)
    verdict["checks"]["prefix_bit_exact"] = prefix_ok

    # (c) Index range of appended faces.
    index_ok = None
    if out_f.ndim == 2 and out_f.shape[0] >= nf_in:
        app_f = np.asarray(out_f[nf_in:], dtype=np.int64)
        verdict["appended_faces"] = app_f.tolist()
        if app_f.size:
            index_ok = bool(app_f.min() >= 0
                            and app_f.max() < out_v.shape[0])
        else:
            index_ok = True
        if not index_ok:
            fail(FAIL_INDEX)
    verdict["checks"]["appended_indices_in_range"] = index_ok
    if out_v.ndim == 2 and out_v.shape[0] >= nv_in:
        verdict["appended_verts"] = np.asarray(
            out_v[nv_in:], dtype=np.float32).tolist()

    # (d) Determinism: run 2 bit-identical to run 1.
    det_ok = None
    if len(runs) == 2 and runs[1]["ok"]:
        det_ok = bool(np.array_equal(runs[1]["out_verts"], out_v)
                      and np.array_equal(runs[1]["out_faces"], out_f))
        if not det_ok:
            fail(FAIL_NONDET)
    elif len(runs) == 2 and not runs[1]["ok"]:
        fail(FAIL_WORKER)
    verdict["checks"]["determinism_two_runs_bit_identical"] = det_ok

    # (e) The parent's vetting, on the canonical (run 1) output.
    fillable, ctx = build_vetting_context(in_verts, in_faces,
                                          max_hole_perimeter, funcs)
    verdict["vetting_context"] = ctx
    if out_v.ndim == 2 and out_f.ndim == 2:
        verdict["vetting"] = run_vetting(funcs, in_verts, in_faces,
                                          out_v, out_f, fillable)

    verdict["status"] = "PASS" if not verdict["failures"] else "FAIL"
    return verdict, (out_v, out_f)


# ---------------------------------------------------------------------
# Compare mode: the section 5.1 gate between two workers.
# ---------------------------------------------------------------------

def compare_one_chunk(funcs, route, chunk_label, in_verts, in_faces,
                      argv_a, argv_b, max_hole_perimeter, timeout_s,
                      work_root):
    """check both workers, then diff their canonical outputs and
    vetting verdicts field by field. A worker that fails its own
    checks fails the comparison under its own class; two passing
    workers whose outputs differ fail as EQUIVALENCE-VIOLATION
    with the differing fields named."""
    va, out_a = check_one_chunk(funcs, route, chunk_label, in_verts,
                                in_faces, argv_a, max_hole_perimeter,
                                timeout_s, work_root)
    vb, out_b = check_one_chunk(funcs, route, chunk_label, in_verts,
                                in_faces, argv_b, max_hole_perimeter,
                                timeout_s, work_root)
    result = {"chunk": chunk_label, "mode": "compare",
              "worker_a": va, "worker_b": vb,
              "comparison": None, "failures": [], "status": None}

    def fail(cls):
        if cls not in result["failures"]:
            result["failures"].append(cls)

    for cls in va["failures"] + vb["failures"]:
        fail(cls)
    if out_a is None or out_b is None:
        result["comparison"] = {"compared": False,
                                "reason": "a worker failed; no pair "
                                          "of outputs to compare"}
        result["status"] = "FAIL"
        return result

    mismatched = []
    detail = {}
    pairs = {
        "appended_verts": (
            np.asarray(out_a[0][in_verts.shape[0]:], dtype=np.float32),
            np.asarray(out_b[0][in_verts.shape[0]:], dtype=np.float32)),
        "appended_faces": (
            np.asarray(out_a[1][in_faces.shape[0]:], dtype=np.int64),
            np.asarray(out_b[1][in_faces.shape[0]:], dtype=np.int64)),
    }
    for field, (xa, xb) in pairs.items():
        eq = bool(xa.shape == xb.shape and np.array_equal(xa, xb))
        detail[field + "_equal"] = eq
        if not eq:
            mismatched.append(field)
    va_v, vb_v = va["vetting"], vb["vetting"]
    kept_eq = (va_v["kept_faces"] == vb_v["kept_faces"])
    detail["kept_faces_equal"] = bool(kept_eq)
    if not kept_eq:
        mismatched.append("kept_faces")
    knew_eq = (va_v["kept_new_verts"] == vb_v["kept_new_verts"])
    detail["kept_new_verts_equal"] = bool(knew_eq)
    if not knew_eq:
        mismatched.append("kept_new_verts")
    drop_eq = (va_v["dropped_faces_count"] == vb_v["dropped_faces_count"])
    detail["dropped_count_equal"] = bool(drop_eq)
    if not drop_eq:
        mismatched.append("dropped_count")
    result["comparison"] = {"compared": True, **detail,
                            "fields_mismatched": mismatched}
    if mismatched:
        fail(FAIL_EQUIV)
    result["status"] = "PASS" if not result["failures"] else "FAIL"
    return result


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def load_chunks(args):
    """[(label, verts float32, faces int64)] from --chunk (one
    file) or --corpus (every *.npz in the dir, sorted by name)."""
    paths = []
    if args.chunk:
        paths = [args.chunk]
    else:
        paths = sorted(os.path.join(args.corpus, n)
                       for n in os.listdir(args.corpus)
                       if n.endswith(".npz"))
    out = []
    for p in paths:
        d = np.load(p)
        out.append((os.path.basename(p),
                    np.asarray(d["verts"], dtype=np.float32),
                    np.asarray(d["faces"], dtype=np.int64)))
    return out


def build_parser():
    p = argparse.ArgumentParser(
        description="R1 hole-fill worker harness (spec 6 Phase 2)")
    sub = p.add_subparsers(dest="mode", required=True)

    def add_common(sp):
        """Options every mode shares, declared per-subparser so the
        natural order is: run_harness.py <mode> --worker ... 
        --chunk ... --max-hole-perimeter ..."""
        sp.add_argument("--generate-py", default=DEFAULT_GENERATE_PY,
                        help="parent generate.py to take vetting/"
                             "census from (default: %(default)s)")
        sp.add_argument("--max-hole-perimeter", type=float,
                        default=0.03,
                        help="fill threshold, mesh units (default: "
                             "the pipeline's 3e-2)")
        sp.add_argument("--timeout", type=float,
                        default=DEFAULT_TIMEOUT_S,
                        help="per-run worker timeout, seconds "
                             "(default: %(default)s)")
        sp.add_argument("--out", default=None,
                        help="also write the JSON verdict to this file")
        src = sp.add_mutually_exclusive_group(required=True)
        src.add_argument("--chunk", help="one chunk .npz")
        src.add_argument("--corpus",
                         help="directory of chunk .npz files")

    pc = sub.add_parser("check", help="one worker, invariant battery")
    pc.add_argument("--worker", required=True,
                    help="worker command (shlex string or JSON argv)")
    add_common(pc)
    pm = sub.add_parser("compare",
                        help="two workers, section 5.1 equivalence")
    pm.add_argument("--worker-a", required=True)
    pm.add_argument("--worker-b", required=True)
    add_common(pm)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    funcs, route = load_parent_functions(args.generate_py)
    chunks = load_chunks(args)
    results = []
    with tempfile.TemporaryDirectory(prefix="r1-harness-root-") as root:
        for label, verts, faces in chunks:
            if args.mode == "check":
                verdict, _ = check_one_chunk(
                    funcs, route, label, verts, faces,
                    parse_worker_cmd(args.worker),
                    args.max_hole_perimeter, args.timeout, root)
                results.append(verdict)
            else:
                results.append(compare_one_chunk(
                    funcs, route, label, verts, faces,
                    parse_worker_cmd(args.worker_a),
                    parse_worker_cmd(args.worker_b),
                    args.max_hole_perimeter, args.timeout, root))
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    report = {
        "mode": args.mode,
        "parent_code_route": route,
        "generate_py": os.path.abspath(args.generate_py),
        "max_hole_perimeter": float(args.max_hole_perimeter),
        "results": results,
        "summary": {"chunks": len(results), "pass": n_pass,
                    "fail": len(results) - n_pass},
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
