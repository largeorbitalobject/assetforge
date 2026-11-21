"""Mesh cleanup (Blender headless): join parts, drop floaters, decimate to a face budget.

Decimation is the biggest speed lever: it slashes the downstream CPU UV-unwrap (which scales
with face count) and is a quality requirement anyway — raw Hunyuan surface-net meshes (~300k
faces) are over-dense and not shippable. Set cfg.target_faces=0 to passthrough.
"""
from __future__ import annotations
import shutil
from pathlib import Path

from .. import blender


def run(ws, cfg, upstream, log) -> dict:
    src = upstream["mesh"]["mesh"]
    dst = ws.dir("cleanup") / "mesh.glb"

    # OPT-IN retopo (MeshFlow): replaces decimation with clean artist topology (runs pre-texture)
    if getattr(cfg, "retopo", False):
        from ..scene import retopo as RT
        if RT.available():
            log("retopo via MeshFlow (clean topology) ...")
            out = RT.retopo(str(Path(src).resolve()), str((dst.parent / "_retopo").resolve()))
            shutil.copy2(out, dst)
            log(f"  retopo -> {dst.name}")
            return {"mesh": str(dst)}
        log("(MeshFlow not set up — skipping retopo; see docs/RETOPO-MESHFLOW.md)")

    if cfg.target_faces and blender.available():
        log(f"decimating to ~{cfg.target_faces} faces (speeds UV-unwrap) ...")
        proc = blender.run("decimate.py", [str(Path(src).resolve()), str(dst.resolve()), cfg.target_faces])
        for line in proc.stdout.splitlines():
            if line.startswith("DECIMATE"):
                log("  " + line)
        return {"mesh": str(dst)}

    shutil.copy2(src, dst)
    if cfg.target_faces:
        log("(Blender unavailable — decimation skipped, passthrough)")
    return {"mesh": str(dst)}
