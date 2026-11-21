#!/usr/bin/env python
"""Native Hunyuan3D-2.1 PBR painter -> textured GLB, generalized for any mesh+image.

MUST be run by the ComfyUI venv python (it needs that torch + the installed custom_rasterizer):

    python -m assetforge.texturing.paint --mesh in.glb --image ref.png --out out_dir [opts]

Key facts baked in (see docs/TEXTURING-ROCM.md):
  * `import torch` happens before any hy3dpaint import (custom_rasterizer DT_NEEDED satisfied).
  * raster_mode='cr' (native rasterizer — no #5981 corruption on this torch nightly), bake_mode default.
  * UV-unwrap is cached to <out>/unwrap.npz (mesh_uv_wrap is ~270s CPU-bound).
  * Everything is pulled to CPU promptly to dodge the transient long-chain HIP fault.
  * texture_size kept square (latent back_project row-major indexing bug otherwise).
Generalized verbatim from the proven prototype paint_run.py.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

# --- import order is load-bearing: torch FIRST ---
import numpy as np
import torch  # noqa: F401  (preloads ROCm libs for custom_rasterizer)
from PIL import Image

# Resolve the Hunyuan paint install relative to this box's ComfyUI.
NODE = os.environ.get("HY_NODE",
                      "/home/orbital/ComfyUI/app/custom_nodes/ComfyUI-Hunyuan3d-2-1.broken")
HY = NODE + "/hy3dpaint"
COMFY_APP = os.environ.get("COMFY_APP", "/home/orbital/ComfyUI/app")


def _setup_paths() -> None:
    for p in (NODE, HY, COMFY_APP):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(HY)  # relative ckpt/ and cfgs/ paths


def paint(mesh_path: str, image_path: str, out_dir: str, *,
          texture_size: int = 1024, resolution: int = 512, steps: int = 10,
          guidance: float = 3.0, ortho_scale: float = 1.2,
          azims=(0, 90, 180, 270, 0, 180), elevs=(0, 0, 0, 0, 90, -90),
          weights=(1, 0.1, 0.5, 0.1, 0.05, 0.05), prompt: str = "high quality",
          seed: int = 0) -> str:
    """Texture `mesh_path` using `image_path` as reference; write a textured GLB into out_dir.

    Returns the path to the exported GLB.
    """
    # Resolve to absolute BEFORE _setup_paths() chdir's into the hy3dpaint dir.
    mesh_path = os.path.abspath(mesh_path)
    image_path = os.path.abspath(image_path)
    out_dir = os.path.abspath(out_dir)
    _setup_paths()
    os.makedirs(out_dir, exist_ok=True)
    azims, elevs, weights = list(azims), list(elevs), list(weights)

    try:
        from utils.torchvision_fix import apply_fix
        apply_fix()
    except Exception as e:  # pragma: no cover
        print("torchvision fix skip:", e)

    import trimesh
    from hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
    from hy3dpaint.utils.uvwrap_utils import mesh_uv_wrap

    conf = Hunyuan3DPaintConfig(
        resolution=resolution, camera_azims=azims, camera_elevs=elevs,
        view_weights=weights, ortho_scale=ortho_scale, texture_size=texture_size,
    )
    print(f"config: raster_mode={conf.raster_mode} bake_mode={conf.bake_mode}")
    pipe = Hunyuan3DPaintPipeline(conf)

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    t0 = time.time()
    import hashlib
    _mh = hashlib.md5(open(mesh_path, "rb").read()).hexdigest()[:12]
    cache = os.path.join(out_dir, f"unwrap_{_mh}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        mesh = trimesh.Trimesh(vertices=d["v"], faces=d["f"],
                               visual=trimesh.visual.TextureVisuals(uv=d["uv"]), process=False)
        print("loaded cached UV unwrap")
    else:
        print("UV unwrapping (CPU, slow) ...")
        mesh = mesh_uv_wrap(mesh)
        try:
            np.savez(cache, v=np.asarray(mesh.vertices, np.float32),
                     f=np.asarray(mesh.faces, np.int32),
                     uv=np.asarray(mesh.visual.uv, np.float32))
        except Exception as e:
            print("unwrap cache failed:", e)
    pipe.render.load_mesh(mesh=mesh)
    print(f"mesh loaded ({time.time()-t0:.1f}s)")

    img = Image.open(image_path).convert("RGB")
    vp = pipe.view_processor
    normal_maps = vp.render_normal_multiview(elevs, azims, use_abs_coor=True)
    position_maps = vp.render_position_multiview(elevs, azims)
    print(f"control maps ({time.time()-t0:.1f}s)")

    if pipe.model is None:
        from hy3dpaint.utils.multiview_utils import multiviewDiffusionNet
        pipe.model = multiviewDiffusionNet(pipe.config)
        print(f"multiview model loaded ({time.time()-t0:.1f}s)")

    mv = pipe.model([img], normal_maps + position_maps, prompt=prompt,
                    custom_view_size=conf.resolution, resize_input=True,
                    num_steps=steps, guidance_scale=guidance, seed=seed)
    print(f"multiview PBR ({time.time()-t0:.1f}s); {len(mv['albedo'])} albedo views")
    for i, im in enumerate(mv["albedo"]):
        im.resize((512, 512)).save(os.path.join(out_dir, f"mv_albedo_{i}.png"))

    albedo_views = [im.resize((conf.render_size, conf.render_size)) for im in mv["albedo"]]
    texture, mask = vp.bake_from_multiview(albedo_views, elevs, azims, weights)
    texture = vp.texture_inpaint(texture, mask.detach().cpu().numpy().astype(np.uint8))
    print(f"bake+inpaint ({time.time()-t0:.1f}s)")

    # pull everything to CPU while the HIP context is healthy
    tex_np = texture.detach().cpu().numpy()
    vtx_pos = pipe.render.vtx_pos.detach().cpu().numpy().copy()
    pos_idx = pipe.render.pos_idx.detach().cpu().numpy().astype(np.int64)
    vtx_uv = pipe.render.vtx_uv.detach().cpu().numpy().copy()
    uv_idx = pipe.render.uv_idx.detach().cpu().numpy().astype(np.int64)
    norm_scale = float(pipe.render.mesh_normalize_scale_factor)
    norm_center = np.asarray(pipe.render.mesh_normalize_scale_center).reshape(3).copy()

    tex_img = (tex_np * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(tex_img).save(os.path.join(out_dir, "albedo_atlas.png"))

    # inverse of MeshRender.get_mesh(normalize=False)
    vtx_pos = vtx_pos / norm_scale + norm_center
    vtx_pos[:, [1, 2]] = vtx_pos[:, [2, 1]]
    vtx_pos[:, [0, 1]] = -vtx_pos[:, [0, 1]]
    vtx_uv = vtx_uv.copy()
    vtx_uv[:, 1] = 1.0 - vtx_uv[:, 1]

    if pos_idx.shape == uv_idx.shape and np.array_equal(pos_idx, uv_idx):
        V, F, UV = vtx_pos, pos_idx, vtx_uv
    else:  # differing topology -> unweld per corner so each corner carries its own UV
        cp, cu = pos_idx.reshape(-1), uv_idx.reshape(-1)
        V, UV = vtx_pos[cp], vtx_uv[cu]
        F = np.arange(len(cp), dtype=np.int64).reshape(-1, 3)

    import trimesh as _tm
    visual = _tm.visual.TextureVisuals(uv=UV, image=Image.fromarray(tex_img))
    out_glb = os.path.join(out_dir, "textured.glb")
    _tm.Trimesh(vertices=V, faces=F, visual=visual, process=False).export(out_glb)
    print(f"SAVED {out_glb} ({time.time()-t0:.1f}s total)")
    return out_glb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Native Hunyuan3D-2.1 PBR painter")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--prompt", default="high quality")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = paint(a.mesh, a.image, a.out, texture_size=a.texture_size, resolution=a.resolution,
                steps=a.steps, guidance=a.guidance, prompt=a.prompt, seed=a.seed)
    print("RESULT_GLB:" + out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
