"""Texturing via the native Hunyuan3D-2.1 painter (proven on gfx1151/ROCm).

Runs assetforge.texturing.paint in the ComfyUI venv (which has the right torch + the installed
custom_rasterizer). See docs/TEXTURING-ROCM.md for why this works and the import-order rule.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

from .. import comfy
from ..rocm_env import COMFY_VENV_PY, gpu_env, HY_NODE, COMFY_APP

PKG_ROOT = Path(__file__).parent.parent.parent  # repo root holding the assetforge/ package


def run(ws, cfg, upstream, log) -> dict:
    mesh = upstream["cleanup"]["mesh"] if "cleanup" in upstream else upstream["mesh"]["mesh"]
    image = upstream["ingest"]["image"] if "ingest" in upstream else ws.manifest["input_image"]
    out_dir = ws.dir("texture")

    # Free ComfyUI's resident models first — painter shares the same unified GPU pool.
    if comfy.free():
        log("freed ComfyUI GPU memory before painting")

    env = gpu_env()
    env["HY_NODE"] = HY_NODE
    env["COMFY_APP"] = COMFY_APP
    env["PYTHONPATH"] = str(PKG_ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [
        COMFY_VENV_PY, "-u", "-m", "assetforge.texturing.paint",
        "--mesh", str(mesh), "--image", str(image), "--out", str(out_dir),
        "--texture-size", str(cfg.texture_size), "--resolution", str(cfg.paint_resolution),
        "--steps", str(cfg.paint_steps), "--guidance", str(cfg.paint_guidance),
    ]
    log(f"painting (Hunyuan native, {cfg.texture_size}px) via {Path(COMFY_VENV_PY).name} ...")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    sys.stderr.write(proc.stdout[-2000:] if proc.stdout else "")
    if proc.returncode != 0:
        raise RuntimeError(f"paint failed (rc={proc.returncode}):\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")

    glb = out_dir / "textured.glb"
    if not glb.exists():
        raise RuntimeError("painter reported success but textured.glb is missing")
    log(f"textured -> {glb}")
    return {"mesh": str(glb), "atlas": str(out_dir / "albedo_atlas.png")}
