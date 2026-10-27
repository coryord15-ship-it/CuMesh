"""Direct binary GLB parser (no trimesh on this VM).

Parses a GLB 2.0 file: JSON chunk -> accessors/bufferViews -> BIN chunk.
Returns positions as float32 (V, 3) and triangle indices as int64 (F, 3),
concatenating all triangle primitives of all meshes with vertex offsets.

Only what the R1 corpus needs: POSITION (float32 VEC3) + indices
(uint8/uint16/uint32 SCALAR), primitive mode 4 (triangles, the GLB
default when `mode` is absent). Interleaved bufferViews (byteStride
different from the element size) are handled via strided reads.
"""
import json
import struct

import numpy as np

_COMP_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_NCOMP = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    "MAT2": 4, "MAT3": 9, "MAT4": 16,
}


def _read_accessor(data_bin, js, accessor_idx):
    acc = js["accessors"][accessor_idx]
    bv = js["bufferViews"][acc["bufferView"]]
    dtype = _COMP_DTYPE[acc["componentType"]]
    ncomp = _TYPE_NCOMP[acc["type"]]
    count = acc["count"]
    elem_size = np.dtype(dtype).itemsize * ncomp
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride", elem_size)
    raw = data_bin[base: base + stride * (count - 1) + elem_size] if count else b""
    if stride == elem_size:
        arr = np.frombuffer(raw, dtype=dtype, count=count * ncomp)
    else:
        arr = np.empty(count * ncomp, dtype=dtype)
        view = np.frombuffer(raw, dtype=np.uint8)
        for i in range(ncomp):
            off = i * np.dtype(dtype).itemsize
            col = view[off: off + stride * count: stride]
            arr[i::ncomp] = np.frombuffer(col.tobytes(), dtype=dtype,
                                          count=count)
    return arr.reshape(count, ncomp) if ncomp > 1 else arr


def load_glb(path):
    """Return (verts float32 (V,3), faces int64 (F,3), meta dict)."""
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != b"glTF":
        raise ValueError(f"{path}: not a GLB (bad magic)")
    version, total_len = struct.unpack_from("<II", blob, 4)
    if version != 2:
        raise ValueError(f"{path}: GLB version {version}, expected 2")
    off = 12
    js = None
    data_bin = None
    while off < total_len:
        clen, ctype = struct.unpack_from("<II", blob, off)
        cdata = blob[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:  # 'JSON'
            js = json.loads(cdata.decode("utf-8"))
        elif ctype == 0x004E4942:  # 'BIN\0'
            data_bin = cdata
        off += 8 + clen
    if js is None or data_bin is None:
        raise ValueError(f"{path}: missing JSON or BIN chunk")

    all_verts = []
    all_faces = []
    voff = 0
    n_prim = 0
    for mesh in js.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4:
                raise ValueError(f"{path}: non-triangle primitive "
                                 f"mode {prim.get('mode')}")
            pos = _read_accessor(data_bin, js,
                                 prim["attributes"]["POSITION"])
            if pos.dtype != np.float32 or pos.shape[1] != 3:
                raise ValueError(f"{path}: POSITION dtype/shape "
                                 f"{pos.dtype} {pos.shape}")
            if "indices" not in prim:
                idx = np.arange(pos.shape[0], dtype=np.int64)
            else:
                idx = _read_accessor(data_bin, js, prim["indices"])
                idx = idx.astype(np.int64)
            if len(idx) % 3 != 0:
                raise ValueError(f"{path}: index count {len(idx)} "
                                 f"not divisible by 3")
            all_verts.append(pos.astype(np.float32, copy=False))
            all_faces.append(idx.reshape(-1, 3) + voff)
            voff += pos.shape[0]
            n_prim += 1
    verts = np.concatenate(all_verts, axis=0) if all_verts else \
        np.zeros((0, 3), dtype=np.float32)
    faces = np.concatenate(all_faces, axis=0) if all_faces else \
        np.zeros((0, 3), dtype=np.int64)
    meta = {"glb_path": path, "n_primitives": n_prim,
            "n_meshes": len(js.get("meshes", []))}
    return verts, faces, meta
