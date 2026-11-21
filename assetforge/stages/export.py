"""Export: copy the final asset into 06_final/ and render a preview montage.

Picks the latest available mesh (textured > cleaned > raw) and renders it headless via
blender_scripts/render_glb.py when Blender is available.
"""
from __future__ import annotations
import shutil
from pathlib import Path

from .. import blender


def _latest_mesh(upstream) -> str:
    for stage in ("texture", "cleanup", "mesh"):
        if stage in upstream and upstream[stage].get("mesh"):
            return upstream[stage]["mesh"]
    raise RuntimeError("no mesh produced upstream")


def run(ws, cfg, upstream, log) -> dict:
    src = _latest_mesh(upstream)
    final = ws.dir("export") / "asset.glb"
    shutil.copy2(src, final)
    out = {"asset": str(final)}

    preview = ws.dir("export") / "preview.png"
    if blender.available() and (blender.SCRIPTS / "render_glb.py").exists():
        try:
            blender.run("render_glb.py", [str(final), str(preview)])
            out["preview"] = str(preview)
            log(f"preview -> {preview}")
        except Exception as e:  # render is best-effort
            log(f"preview render skipped: {e}")
    log(f"final asset -> {final}")
    return out
