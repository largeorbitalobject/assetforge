#!/usr/bin/env python
"""Full native Hunyuan3D-2.1 PBR paint on gfx1151 -> textured GLB.
Uses the native custom_rasterizer (raster_mode='cr', bake_mode='back_sample')."""
import os, sys, time
import numpy as np
import torch
from PIL import Image

NODE = "/home/orbital/ComfyUI/app/custom_nodes/ComfyUI-Hunyuan3d-2-1.broken"
HY = NODE + "/hy3dpaint"
sys.path.insert(0, NODE)   # so 'hy3dpaint' is importable as a package
sys.path.insert(0, HY)     # so 'utils', 'DifferentiableRenderer' etc resolve too
sys.path.insert(0, "/home/orbital/ComfyUI/app")  # for folder_paths import
os.chdir(HY)  # for relative ckpt/ and cfgs/ paths

OUT = "/home/orbital/.claude/jobs/fde14dc0/tmp/paint_out"
os.makedirs(OUT, exist_ok=True)

MESH = "/home/orbital/projects/orbitis-station/assets/01_raw_glb/minotaur_00001_.glb"
IMG = "/home/orbital/ComfyUI/app/input/orbitis_minotaur.png"

try:
    from utils.torchvision_fix import apply_fix; apply_fix()
except Exception as e:
    print("torchvision fix skip:", e)

import trimesh
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
from hy3dpaint.utils.uvwrap_utils import mesh_uv_wrap

# 6 standard views
azims  = [0, 90, 180, 270, 0, 180]
elevs  = [0, 0, 0, 0, 90, -90]
weights= [1, 0.1, 0.5, 0.1, 0.05, 0.05]

conf = Hunyuan3DPaintConfig(
    resolution=512, camera_azims=azims, camera_elevs=elevs,
    view_weights=weights, ortho_scale=1.2, texture_size=1024,
)
print("config built. raster_mode=", conf.raster_mode, "bake_mode=", conf.bake_mode)

pipe = Hunyuan3DPaintPipeline(conf)
print("pipeline + MeshRender constructed")

mesh = trimesh.load(MESH, process=False)
if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

t0 = time.time()
UNWRAP_CACHE = "/home/orbital/.claude/jobs/fde14dc0/tmp/minotaur_unwrapped.npz"
if os.path.exists(UNWRAP_CACHE):
    print("loading cached unwrapped mesh (npz) ...")
    d = np.load(UNWRAP_CACHE)
    vis = trimesh.visual.TextureVisuals(uv=d["uv"])
    mesh = trimesh.Trimesh(vertices=d["v"], faces=d["f"], visual=vis, process=False)
else:
    print("Unwrapping mesh ...")
    mesh = mesh_uv_wrap(mesh)
    try:
        np.savez(UNWRAP_CACHE, v=np.asarray(mesh.vertices, np.float32),
                 f=np.asarray(mesh.faces, np.int32),
                 uv=np.asarray(mesh.visual.uv, np.float32)); print("cached unwrap (npz)")
    except Exception as e:
        print("unwrap cache failed:", e)
pipe.render.load_mesh(mesh=mesh)
print(f"mesh loaded ({time.time()-t0:.1f}s)")

# multiview generation (loads multiview model lazily inside __call__)
img = Image.open(IMG).convert("RGB")

# manual driving: replicate __call__ multiview portion then bake
vp = pipe.view_processor
print("rendering normal/position control maps ...")
normal_maps = vp.render_normal_multiview(elevs, azims, use_abs_coor=True)
position_maps = vp.render_position_multiview(elevs, azims)
print(f"control maps done ({time.time()-t0:.1f}s)")

if pipe.model is None:
    from hy3dpaint.utils.multiview_utils import multiviewDiffusionNet
    print("loading multiview diffusion model (this loads the PBR UNet + SD2.1 base) ...")
    pipe.model = multiviewDiffusionNet(pipe.config)
    print(f"multiview model loaded ({time.time()-t0:.1f}s)")

print("running multiview PBR diffusion ...")
multiviews_pbr = pipe.model(
    [img], normal_maps + position_maps, prompt="high quality",
    custom_view_size=conf.resolution, resize_input=True,
    num_steps=10, guidance_scale=3.0, seed=0,
)
print(f"multiview PBR done ({time.time()-t0:.1f}s); albedo views:", len(multiviews_pbr["albedo"]))

# save raw multiview albedo for inspection
for i, im in enumerate(multiviews_pbr["albedo"]):
    im.resize((512,512)).save(f"{OUT}/mv_albedo_{i}.png")

# bake albedo
albedo_views = [im.resize((conf.render_size, conf.render_size)) for im in multiviews_pbr["albedo"]]
print("baking albedo via native rasterizer back_project/back_sample ...")
texture, mask = vp.bake_from_multiview(albedo_views, elevs, azims, weights)
print(f"bake done ({time.time()-t0:.1f}s); texture shape {tuple(texture.shape)}")

# inpaint holes
mask_np = (mask.detach().cpu().numpy()).astype(np.uint8)
texture = vp.texture_inpaint(texture, mask_np)
print(f"inpaint done ({time.time()-t0:.1f}s)")

# ---- pull EVERYTHING to CPU immediately, while the HIP context is still healthy ----
tex_np = texture.detach().cpu().numpy()                # [H,W,3] float [0,1]
vtx_pos = pipe.render.vtx_pos.detach().cpu().numpy().copy()
pos_idx = pipe.render.pos_idx.detach().cpu().numpy().copy()
vtx_uv  = pipe.render.vtx_uv.detach().cpu().numpy().copy()
uv_idx  = pipe.render.uv_idx.detach().cpu().numpy().copy()
norm_scale = float(pipe.render.mesh_normalize_scale_factor)
norm_center = np.asarray(pipe.render.mesh_normalize_scale_center).reshape(3).copy()
print("pulled mesh+texture to CPU")

# save the texture atlas
tex_img = (tex_np * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(tex_img).save(f"{OUT}/albedo_atlas.png")
print("texture atlas saved")

# ---- replicate MeshRender.get_mesh(normalize=False) inverse transform on CPU ----
vtx_pos = vtx_pos / norm_scale
vtx_pos = vtx_pos + norm_center
vtx_pos[:, [1, 2]] = vtx_pos[:, [2, 1]]
vtx_pos[:, [0, 1]] = -vtx_pos[:, [0, 1]]
vtx_uv = vtx_uv.copy()
vtx_uv[:, 1] = 1.0 - vtx_uv[:, 1]

# Build a trimesh with UVs + texture. pos_idx and uv_idx may differ; remap to per-uv-vertex.
import trimesh as _tm
from PIL import Image as _PImage

pos_idx = pos_idx.astype(np.int64); uv_idx = uv_idx.astype(np.int64)
if pos_idx.shape == uv_idx.shape and np.array_equal(pos_idx, uv_idx):
    # shared topology: direct per-vertex UVs
    V, Fp, UV = vtx_pos, pos_idx, vtx_uv
    print("pos_idx == uv_idx (shared topology)")
else:
    # differing topology: unweld to per-corner vertices so each corner gets its own UV
    print("pos_idx != uv_idx -> unwelding to per-corner vertices")
    corners_pos = pos_idx.reshape(-1)   # 3F
    corners_uv = uv_idx.reshape(-1)     # 3F
    V = vtx_pos[corners_pos]
    UV = vtx_uv[corners_uv]
    Fp = np.arange(len(corners_pos), dtype=np.int64).reshape(-1, 3)

visual = _tm.visual.TextureVisuals(uv=UV, image=_PImage.fromarray(tex_img))
out_mesh = _tm.Trimesh(vertices=V, faces=Fp, visual=visual, process=False)
out_glb = f"{OUT}/minotaur_textured_native.glb"
out_mesh.export(out_glb)
print(f"SAVED GLB: {out_glb} ({time.time()-t0:.1f}s total)")
print("OUTPUTS:", sorted(os.listdir(OUT)))
