"""Mesh generation via ComfyUI.

creature/mechanical/object/building -> Hunyuan3D-2.1 (per-object mesh, proven).
scene                              -> HunyuanWorld-Mirror (whole-scene mesh, weights cached).
The input image is copied into ComfyUI/app/input/ (LoadImage reads from there).
"""
from __future__ import annotations
import json, shutil, time, glob, os
from pathlib import Path

from .. import comfy
from ..rocm_env import COMFY_APP

GRAPH_DIR = Path(__file__).parent.parent / "comfy_graphs"
COMFY_INPUT = Path(COMFY_APP) / "input"
COMFY_OUTPUT = Path(COMFY_APP) / "output"


def _hunyuan3d_graph(image_name: str, cfg) -> dict:
    g = json.loads((GRAPH_DIR / "hunyuan3d-2.1-image-to-glb.json").read_text())["prompt"]
    g["2"]["inputs"]["image"] = image_name
    g["8"]["inputs"]["octree_resolution"] = cfg.octree_resolution
    g["9"]["inputs"]["threshold"] = cfg.voxel_threshold
    g["7"]["inputs"]["steps"] = cfg.ksampler_steps
    g["10"]["inputs"]["filename_prefix"] = "assetforge/mesh"
    return g


def _hunyuanworld_graph(image_name: str) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "HunyuanWorldMirrorModelLoader",
              "inputs": {"model_name": "tencent/HunyuanWorld-Mirror",
                         "precision": "auto", "offload_to_cpu": False}},
        "3": {"class_type": "HunyuanWorldMirrorImageTo3D",
              "inputs": {"model": ["2", 0], "images": ["1", 0],
                         "preprocessing_mode": "crop", "save_outputs": True,
                         "filename_prefix": "assetforge/scene",
                         "preprocessing_resolution": 518, "confidence_filter": 5.0,
                         "edge_normal_threshold": 8.0, "edge_depth_threshold": 0.05,
                         "boundary_erosion": 2}},
    }


def _newest_glb_since(t0: float) -> Path | None:
    cands = [Path(p) for p in glob.glob(str(COMFY_OUTPUT / "**" / "*.glb"), recursive=True)]
    cands = [p for p in cands if p.stat().st_mtime >= t0 - 1]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def run(ws, cfg, upstream, log) -> dict:
    image = Path(upstream["ingest"]["image"]) if "ingest" in upstream else Path(ws.manifest["input_image"])
    staged = COMFY_INPUT / image.name
    shutil.copy2(image, staged)

    if cfg.mesh_backend == "hunyuanworld" or cfg.asset_type == "scene":
        log(f"submitting HunyuanWorld scene graph for {image.name} ...")
        t0 = time.time()
        comfy.run_graph(_hunyuanworld_graph(image.name), ext=".glb", timeout=1800)
        out = _newest_glb_since(t0)
        if not out:
            raise RuntimeError("HunyuanWorld produced no GLB")
    else:
        log(f"submitting Hunyuan3D-2.1 graph for {image.name} (octree={cfg.octree_resolution})")
        outs = comfy.run_graph(_hunyuan3d_graph(image.name, cfg), ext=".glb", timeout=1800)
        if not outs:
            raise RuntimeError("ComfyUI produced no GLB")
        out = outs[0]

    dst = ws.dir("mesh") / "mesh.glb"
    shutil.copy2(out, dst)
    log(f"mesh -> {dst}")
    return {"mesh": str(dst)}
