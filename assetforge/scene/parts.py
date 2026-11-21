"""Shared 3D part segmentation: split a mesh into named parts via multiview SAM3 lifting.

Keystone for equipment separation (sword/armor) AND M2 mechanical decomposition.
  segment_parts(mesh_glb, concepts, work_dir) -> {label: np.array(face_indices)}
  split_mesh(mesh_glb, face_labels, out_dir)  -> {label: part_glb_path}

Pipeline: render_faceids.py (beauty + face-id buffers per view) -> SAM3 text masks per view ->
decode face-id under each mask -> vote per face per concept -> argmax label per face.
"""
from __future__ import annotations
import os, json, subprocess, numpy as np
from pathlib import Path
from PIL import Image

from ..rocm_env import BLENDER, gpu_env
SCRIPTS = Path(__file__).parent.parent / "blender_scripts"
VIEWS = ["front", "right", "back", "left", "top", "bottom"]


def _decode_fid(fid_exr):
    import os as _os
    _os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    a = cv2.imread(fid_exr, cv2.IMREAD_UNCHANGED)   # float32 BGR linear
    if a is None:
        raise RuntimeError("cannot read " + fid_exr)
    rgb = a[..., ::-1] if a.ndim == 3 else a
    b = np.rint(np.clip(rgb, 0, 1) * 255.0).astype(np.int64)
    return b[..., 0] + (b[..., 1] << 8) + (b[..., 2] << 16)   # 0 = background


def segment_parts(mesh_glb, concepts, work_dir, threshold=0.4) -> dict:
    os.makedirs(work_dir, exist_ok=True)
    env = gpu_env(); env.pop("LD_LIBRARY_PATH", None)
    subprocess.run([BLENDER, "-b", "-noaudio", "-P", str(SCRIPTS / "render_faceids.py"),
                    "--", str(mesh_glb), work_dir, str(len(VIEWS))],
                   env=env, check=True, capture_output=True, text=True, timeout=600)
    meta = json.load(open(os.path.join(work_dir, "cams.json")))
    nF = meta["n_faces"]
    from .decompose import _load
    import torch
    model, proc = _load()
    dev = next(model.parameters()).device
    votes = np.zeros((nF + 1, len(concepts)), np.float32)
    for v in VIEWS:
        bp = os.path.join(work_dir, f"{v}_beauty.png"); fp = os.path.join(work_dir, f"{v}_fid.exr")
        if not (os.path.exists(bp) and os.path.exists(fp)):
            continue
        fid = _decode_fid(fp)
        img = Image.open(bp).convert("RGB")
        for ci, c in enumerate(concepts):
            inp = proc(images=img, text=c, return_tensors="pt").to(dev)
            with torch.no_grad():
                res = model(**inp)
            r = proc.post_process_instance_segmentation(res, threshold=threshold, mask_threshold=0.5,
                                                        target_sizes=[(img.height, img.width)])[0]
            masks = r.get("masks")
            if masks is None or len(masks) == 0:
                continue
            mask = masks.cpu().numpy().max(0) > 0.5
            fids = fid[mask]
            fids = fids[(fids > 0) & (fids <= nF)]
            if len(fids):
                np.add.at(votes[:, ci], fids, 1.0)
    # argmax label per face; faces with no votes -> unlabeled (-1)
    total = votes.sum(1)
    lbl = np.where(total > 0, votes.argmax(1), -1)
    out = {}
    for ci, c in enumerate(concepts):
        faces = np.where(lbl == ci)[0]
        faces = faces[faces >= 1] - 1   # back to 0-based face indices
        if len(faces):
            out[c] = faces
    return out


def split_mesh(mesh_glb, face_labels, out_dir) -> dict:
    import trimesh
    os.makedirs(out_dir, exist_ok=True)
    m = trimesh.load(mesh_glb, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    out = {}
    assigned = np.zeros(len(m.faces), bool)
    for label, faces in face_labels.items():
        faces = np.asarray(faces)
        faces = faces[faces < len(m.faces)]
        if len(faces) == 0:
            continue
        sub = m.submesh([faces], append=True)
        p = os.path.join(out_dir, f"part_{label}.glb"); sub.export(p)
        out[label] = p; assigned[faces] = True
    # remainder = body
    rest = np.where(~assigned)[0]
    if len(rest):
        body = m.submesh([rest], append=True)
        p = os.path.join(out_dir, "part_body.glb"); body.export(p); out["body"] = p
    return out


def refine_labels(mesh_glb, face_labels, min_comp_frac=0.08, split_instances=False) -> dict:
    """Clean face labels via connected components (scipy) on mesh face-adjacency:
    drop scattered noise islands (< min_comp_frac of a label); optionally split into per-instance
    parts (label_0, label_1, ...). Returns refined {label: face_array}.
    """
    import trimesh, numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components as cc
    m = trimesh.load(mesh_glb, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    adj = m.face_adjacency  # (E,2)
    out = {}
    for label, faces in face_labels.items():
        faces = np.unique(np.asarray(faces).astype(int))
        if len(faces) == 0:
            continue
        # remap faces -> 0..k-1; keep adjacency edges fully inside the subset
        remap = {int(f): i for i, f in enumerate(faces)}
        if len(adj):
            sub = adj[np.isin(adj[:, 0], faces) & np.isin(adj[:, 1], faces)]
            ei = np.array([[remap[int(a)], remap[int(b)]] for a, b in sub], int) if len(sub) else np.zeros((0, 2), int)
        else:
            ei = np.zeros((0, 2), int)
        n = len(faces)
        if len(ei):
            g = csr_matrix((np.ones(len(ei)*2), (np.r_[ei[:,0], ei[:,1]], np.r_[ei[:,1], ei[:,0]])), shape=(n, n))
        else:
            g = csr_matrix((n, n))
        ncomp, lab = cc(g, directed=False)
        comps = [faces[lab == c] for c in range(ncomp)]
        total = sum(len(c) for c in comps)
        comps = [c for c in comps if len(c) >= max(1, min_comp_frac * total)]
        comps.sort(key=len, reverse=True)
        if not comps:
            continue
        if split_instances:
            for i, c in enumerate(comps):
                out[f"{label}_{i}"] = c
        else:
            out[label] = np.concatenate(comps)
    return out


def split_parts_blender(mesh_glb, face_labels, out_dir, min_comp_frac=0.08, split_instances=False) -> dict:
    """Split into part GLBs IN BLENDER (consistent face indexing + correct welded components).
    Supersedes split_mesh + refine_labels (which broke on trimesh reordering + unwelded soup).
    """
    import subprocess, json, os
    from ..rocm_env import BLENDER, gpu_env
    from pathlib import Path
    os.makedirs(out_dir, exist_ok=True)
    lj = os.path.join(out_dir, "labels.json")
    json.dump({k: [int(x) for x in v] for k, v in face_labels.items()}, open(lj, "w"))
    env = gpu_env(); env.pop("LD_LIBRARY_PATH", None)
    proc = subprocess.run(
        [BLENDER, "-b", "-noaudio", "-P", str(Path(__file__).parent.parent / "blender_scripts" / "split_by_faceids.py"),
         "--", str(mesh_glb), lj, out_dir, str(min_comp_frac), "true" if split_instances else "false"],
        env=env, capture_output=True, text=True, timeout=600)
    for line in proc.stdout.splitlines():
        if line.startswith("SPLIT "):
            return json.loads(line[6:])
    raise RuntimeError("split_by_faceids produced no SPLIT line:\n" + proc.stdout[-800:] + proc.stderr[-400:])
