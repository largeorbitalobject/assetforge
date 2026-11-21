"""Run RigAnything (organic auto-rig) as a subprocess in the ComfyUI ROCm venv.

RigAnything is verified clean-torch on gfx1151 (SDPA attention, no spconv). It outputs an npz with
joints(J,3)/parents(J,)/skinning_weights(N,J)/pointcloud(N,3). LICENSE: Adobe non-commercial — fine
as an internal tool; swap to Make-It-Animatable for commercial output.
"""
from __future__ import annotations
import os, glob, subprocess

from ..rocm_env import COMFY_VENV_PY, gpu_env

RA_DIR = "/home/orbital/projects/orbitis-station/tools/RigAnything"
CKPT = "ckpt/riganything_ckpt.pt"
CONFIG = "config.yaml"


def run_riganything(mesh_path: str, out_dir: str, timeout: float = 1800) -> str:
    """Rig `mesh_path`; return the path to the produced .npz."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [COMFY_VENV_PY, "inference.py", "--config", CONFIG, "--load", CKPT,
           "-s", "inference", "true",
           "-s", "inference_out_dir", os.path.abspath(out_dir),
           "--mesh_path", os.path.abspath(mesh_path)]
    proc = subprocess.run(cmd, cwd=RA_DIR, env=gpu_env(), capture_output=True,
                          text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"RigAnything failed (rc={proc.returncode}):\n"
                           f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    npzs = sorted(glob.glob(os.path.join(out_dir, "*.npz")), key=os.path.getmtime)
    if not npzs:
        raise RuntimeError(f"RigAnything produced no npz in {out_dir}")
    return npzs[-1]
