"""Animate stage — organic (roles-driven walk) or mechanical (procedural DOF driver)."""
from __future__ import annotations
from pathlib import Path
from .. import blender


def _input_mesh(upstream, ws):
    for s in ("texture", "cleanup", "mesh"):
        if s in upstream and upstream[s].get("mesh"):
            return upstream[s]["mesh"]
    return ws.manifest["input_image"]


def run(ws, cfg, upstream, log) -> dict:
    kind = cfg.kind
    animdir = ws.dir("animate"); rdir = animdir / "render"
    rig = upstream.get("rig", {})
    quality = "draft"

    if kind == "organic":
        npz, graph = rig.get("npz"), rig.get("graph")
        if not (npz and graph):
            raise RuntimeError("organic animate needs rig.npz + rig.graph")
        mesh = _input_mesh(upstream, ws)
        out_glb = animdir / "animation.glb"
        log("animating (roles-driven walk) ...")
        blender.run("anim_organic.py", [npz, mesh, graph, str(out_glb), str(rdir), quality])
        return {"mesh": str(out_glb), "render": str(rdir)}

    if kind == "mechanical":
        blend = rig.get("blend"); graph = rig.get("graph")
        if not (blend and graph):
            raise RuntimeError("mechanical animate needs rig.blend + rig.graph")
        out_glb = animdir / "animation.glb"
        log("animating (procedural DOF driver) ...")
        # driver_artic opens the rig .blend
        from ..rocm_env import BLENDER, gpu_env
        import subprocess
        cmd = [BLENDER, str(blend), "-b", "-noaudio", "-P",
               str(blender.SCRIPTS / "driver_artic.py"), "--",
               graph, str(out_glb), str(rdir), "48"]
        subprocess.run(cmd, env=gpu_env(), check=True, capture_output=True, text=True)
        return {"mesh": str(out_glb), "render": str(rdir)}

    raise NotImplementedError(f"animate: asset_type {cfg.asset_type!r} has no animate branch")
