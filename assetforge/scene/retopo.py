"""OPT-IN retopology via MeshFlow (facebookresearch/meshflow): dense Hunyuan mesh -> clean
artist-topology triangle mesh. OFF by default.

ROCm: LIKELY-RUNS (pure PyTorch / SDPA, no spconv/CUDA kernels) — only edit MeshFlow's
requirements.txt to drop the cu128 extra-index-url (you already have ROCm torch).
GATED weights + NON-COMMERCIAL license (FAIR Noncommercial Research) — research/internal use only;
cannot ship in a commercial product. See docs/RETOPO-MESHFLOW.md.

Setup (one-time, user action): clone facebookresearch/meshflow, accept the HF gate + set HF token,
then set ASSETFORGE_MESHFLOW=/path/to/meshflow. Output has NO UVs/texture -> re-texture after.
"""
from __future__ import annotations
import os, glob, subprocess
from ..rocm_env import COMFY_VENV_PY, gpu_env

MESHFLOW_DIR = os.environ.get("ASSETFORGE_MESHFLOW", os.path.expanduser("~/meshflow"))


def available() -> bool:
    return bool(MESHFLOW_DIR) and os.path.exists(os.path.join(MESHFLOW_DIR, "inference_dit.py"))


def retopo(mesh_glb: str, out_dir: str, num_verts: int = 2048, steps: int = 25,
           python_bin: str = COMFY_VENV_PY) -> str:
    """Retopologize mesh_glb -> clean GLB. Requires MeshFlow set up (see module docstring)."""
    if not available():
        raise RuntimeError("MeshFlow not set up: set ASSETFORGE_MESHFLOW to a cloned facebookresearch/meshflow "
                           "(gated weights + non-commercial license). See docs/RETOPO-MESHFLOW.md")
    os.makedirs(out_dir, exist_ok=True)
    from huggingface_hub import snapshot_download
    snap = snapshot_download("facebook/meshflow")
    model_dir = snap
    for root, _, files in os.walk(snap):
        if "config.yaml" in files and "model.pth" in files:
            model_dir = root; break
    env = gpu_env(); env.pop("LD_LIBRARY_PATH", None); env["PYTHONPATH"] = MESHFLOW_DIR
    cmd = [python_bin, os.path.join(MESHFLOW_DIR, "inference_dit.py"),
           "--model_path", model_dir,
           "--input", os.path.abspath(mesh_glb), "--output", os.path.abspath(out_dir),
           "--steps", str(steps), "--dtype", "bf16", "--device", "cuda"]
    subprocess.run(cmd, cwd=MESHFLOW_DIR, env=env, check=True, capture_output=True, text=True, timeout=900)
    outs = sorted(glob.glob(os.path.join(out_dir, "generated_meshes", "*.glb")), key=os.path.getmtime)
    if not outs:
        raise RuntimeError("MeshFlow produced no GLB in generated_meshes/")
    return outs[-1]
