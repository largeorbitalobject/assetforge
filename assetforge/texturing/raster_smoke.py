#!/usr/bin/env python
"""Smoke-test custom_rasterizer on gfx1151 to detect ROCm #5981 corruption.
Loads minotaur GLB, projects to clip space with MeshRender's own camera utils,
rasterizes a front view, and dumps colorized face-id / barycentric / coverage PNGs.
"""
import os, sys, math
import numpy as np

OUT = "/home/orbital/.claude/jobs/fde14dc0/tmp/raster_smoke"
os.makedirs(OUT, exist_ok=True)

import torch  # MUST be before custom_rasterizer
import custom_rasterizer as cr
import trimesh

sys.path.insert(0, "/home/orbital/ComfyUI/app/custom_nodes/ComfyUI-Hunyuan3d-2-1.broken/hy3dpaint/DifferentiableRenderer")
from camera_utils import get_mv_matrix, get_orthographic_projection_matrix, transform_pos

MESH = "/home/orbital/projects/orbitis-station/assets/01_raw_glb/minotaur_00001_.glb"
RES = 512
device = "cuda"

# ---- load mesh ----
scene = trimesh.load(MESH, process=False)
if isinstance(scene, trimesh.Scene):
    mesh = trimesh.util.concatenate([g for g in scene.geometry.values()])
else:
    mesh = scene
V = np.asarray(mesh.vertices, dtype=np.float32)
F = np.asarray(mesh.faces, dtype=np.int32)
print(f"mesh: {V.shape[0]} verts, {F.shape[0]} faces, bbox={V.min(0)}..{V.max(0)}")

# ---- MeshRender's load-time coord swap (set_mesh :703-706): x,y *= -1 ; swap(y,z) ----
Vt = V.copy()
Vt[:, [0, 1]] *= -1.0
Vt = Vt[:, [0, 2, 1]]  # swap y and z

# normalize: center + scale to unit-ish (mimic MeshRender auto-center/scale)
center = (Vt.max(0) + Vt.min(0)) / 2.0
Vt = Vt - center
scale = 0.5 / np.abs(Vt).max()
Vt = Vt * scale  # now within ~[-0.5,0.5]

vtx = torch.from_numpy(Vt).to(device)            # [V,3]
tri = torch.from_numpy(F).to(device).int()       # [F,3]

def render_view(elev, azim, name):
    mv = get_mv_matrix(elev=elev, azim=azim, camera_distance=2.0)
    proj = get_orthographic_projection_matrix(left=-0.6, right=0.6, bottom=-0.6, top=0.6, near=0.1, far=100)
    mvp = (torch.from_numpy(proj) @ torch.from_numpy(mv)).to(device)  # [4,4]
    pos_clip = transform_pos(mvp, vtx)  # [1,V,4]
    pos_clip = pos_clip.contiguous().float()
    findices, bary = cr.rasterize(pos_clip, tri, (RES, RES))
    fi = findices.detach().cpu().numpy()        # [H,W] int, 1-based, 0=bg
    bc = bary.detach().cpu().numpy()            # [H,W,3]
    cov = (fi > 0).mean()
    nfaces = len(np.unique(fi)) - (1 if 0 in fi else 0)
    bsum = bc[fi > 0].sum(-1)
    print(f"[{name}] coverage={cov:.3f} faces_hit={nfaces} "
          f"bary_rowsum mean={bsum.mean():.4f} min={bsum.min():.4f} max={bsum.max():.4f} "
          f"bary_finite={np.isfinite(bc).all()}")
    return fi, bc

def save_png(arr, path):
    import imageio.v2 as imageio
    imageio.imwrite(path, arr.astype(np.uint8))

import imageio.v2 as imageio
results = {}
for elev, azim, name in [(0,0,"front"), (0,90,"side"), (20,45,"three_quarter")]:
    fi, bc = render_view(elev, azim, name)
    results[name] = (fi, bc)
    # (a) face-id colorized via hashed palette
    H, W = fi.shape
    rng = np.random.default_rng(0)
    maxid = int(fi.max()) + 1
    palette = (rng.random((maxid + 1, 3)) * 255).astype(np.uint8)
    palette[0] = [0, 0, 0]  # bg black
    faceimg = palette[fi]
    save_png(faceimg, f"{OUT}/{name}_faceid.png")
    # (b) barycentric as RGB
    baryimg = (np.clip(bc, 0, 1) * 255).astype(np.uint8)
    save_png(baryimg, f"{OUT}/{name}_bary.png")
    # (c) coverage / mask
    mask = ((fi > 0).astype(np.uint8) * 255)
    save_png(np.stack([mask]*3, -1), f"{OUT}/{name}_mask.png")

print("OUTPUTS:")
for f in sorted(os.listdir(OUT)):
    print(" ", os.path.join(OUT, f))
