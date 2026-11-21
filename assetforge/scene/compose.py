"""Track B orchestrator: single image -> compositional, editable 3D scene.

decompose (SAM3) -> MoGe metric layout -> per-hero gen (reuse `assetforge run --type object`)
-> backdrop mesh (MoGe) -> Blender assembly -> scene GLB.

    python -m assetforge.scene.compose <image> <out_dir> [concepts] [max_heroes]
"""
from __future__ import annotations
import os, sys, json, subprocess, numpy as np
from pathlib import Path
from PIL import Image

from ..rocm_env import COMFY_VENV_PY, BLENDER, gpu_env, COMFY_APP
PKG = Path(__file__).parent.parent.parent


def _masked_crop(image_path, mask, out_png, pad=0.06):
    im = np.asarray(Image.open(image_path).convert("RGB"))
    H, W = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return False
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = int((x1-x0)*pad)+4, int((y1-y0)*pad)+4
    x0, x1 = max(0, x0-px), min(W, x1+px); y0, y1 = max(0, y0-py), min(H, y1+py)
    rgba = np.zeros((H, W, 4), np.uint8); rgba[..., :3] = im; rgba[..., 3] = (mask*255).astype(np.uint8)
    Image.fromarray(rgba[y0:y1, x0:x1]).save(out_png)
    return True


def _gen_hero(crop_png, run_name, env):
    """Run the proven object pipeline on a hero crop -> textured GLB path."""
    cmd = [COMFY_VENV_PY, "-m", "assetforge.cli", "run", str(crop_png),
           "--type", "object", "--until", "texture", "--no-resume"]
    e = dict(env); e["PYTHONPATH"] = str(PKG)
    subprocess.run(cmd, env=e, cwd=str(PKG), check=True, capture_output=True, text=True, timeout=2400)
    # CLI writes to assets/runs/<slug>/03_texture/textured.glb
    from ..workspace import slugify
    glb = PKG / "assets" / "runs" / slugify(str(crop_png)) / "03_texture" / "textured.glb"
    return str(glb) if glb.exists() else None


def compose(image_path, out_dir, concepts=None, max_heroes=2):
    os.makedirs(out_dir, exist_ok=True)
    env = gpu_env(); env.pop("LD_LIBRARY_PATH", None)
    from . import decompose as DC, depth as DEP, layout as LAY, point_mesh as PM

    print("[1/5] decompose (VLM hero-naming + SAM3)...")
    if concepts is None:
        try:
            from .. import vlm
            if vlm.available():
                named = vlm.name_heroes(image_path)
                if named:
                    concepts = named; print("   VLM heroes:", concepts)
        except Exception as e:
            print("   VLM naming skipped:", str(e)[:100])
    masks = DC.decompose(image_path, concepts=concepts)
    # pick the most distinct heroes by area, drop heavily-overlapping ones
    items = sorted(masks.items(), key=lambda kv: -kv[1].sum())
    heroes, used = [], np.zeros(next(iter(masks.values())).shape, bool) if masks else None
    for name, m in items:
        if used is not None and (m & used).sum() > 0.5 * m.sum():
            continue
        heroes.append((name, m)); used |= m
        if len(heroes) >= max_heroes:
            break
    print("   heroes:", [h[0] for h in heroes])

    print("[2/5] MoGe metric geometry + layout...")
    pts, vmask, intr = DEP.infer_points(image_path)
    lay = DEP.layout_from_points(pts, {n: m for n, m in heroes})

    print("[3/5] backdrop mesh...")
    backdrop, nv, nf = PM.points_to_mesh(pts, vmask, image_path, os.path.join(out_dir, "backdrop.glb"))
    print(f"   backdrop {nv} verts")

    print("[4/5] per-hero generation (reuse object pipeline)...")
    spec_heroes = []
    for name, m in heroes:
        crop = os.path.join(out_dir, f"hero_{name}.png")
        if not _masked_crop(image_path, m, crop):
            continue
        try:
            glb = _gen_hero(crop, name, env)
        except Exception as e:
            print(f"   {name}: gen failed {str(e)[:100]}"); glb = None
        if glb:
            pl = lay["heroes"][name]
            spec_heroes.append({"glb": glb, "pos": pl["pos"], "scale": 1.0, "yaw": 0.0})
            print(f"   {name}: {glb}")

    print("[5/5] assemble scene...")
    spec = {"skydome": image_path, "ground": {"size": 16.0, "heightmap": None, "z": 0.0},
            "heroes": spec_heroes, "backdrop": backdrop}
    spec_path = os.path.join(out_dir, "scene_spec.json"); json.dump(spec, open(spec_path, "w"), indent=2)
    out_glb = os.path.join(out_dir, "scene.glb"); render = os.path.join(out_dir, "scene_render.png")
    subprocess.run([BLENDER, "-b", "-noaudio", "-P", str(PKG / "assetforge" / "blender_scripts" / "assemble_scene.py"),
                    "--", spec_path, out_glb, render], env=env, check=True, capture_output=True, text=True)
    print("SCENE", out_glb)
    return out_glb


if __name__ == "__main__":
    a = sys.argv
    compose(a[1], a[2],
            concepts=a[3].split(",") if len(a) > 3 else None,
            max_heroes=int(a[4]) if len(a) > 4 else 2)
