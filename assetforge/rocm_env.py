"""The proven GPU environment for gfx1151 / ROCm 7.12 on this box.

These are the exact exports from ComfyUI's start script, verified necessary for the
Hunyuan native painter and other GPU work. CRITICAL: keep LD_LIBRARY_PATH clean — adding
the venv's _rocm_sdk dirs or system /opt/rocm-7.1.0 shadows the venv's bundled ROCm 7.12
libs and causes real undefined-symbol crashes (ncclCommDump / hsa_ext_image_create_v2).

The custom_rasterizer "missing libc10.so / librocm-openblas.so.0" errors are a red herring:
they only occur when the kernel is imported with no torch in the process. `import torch`
BEFORE `import custom_rasterizer` and torch's rocm_sdk loader preloads everything by abs path.
"""
from __future__ import annotations
import os

COMFY_ROOT = "/home/orbital/ComfyUI"
COMFY_APP = COMFY_ROOT + "/app"
COMFY_VENV_PY = COMFY_ROOT + "/venv/bin/python"

# Hunyuan3D-2.1 paint install (the dir is currently suffixed .broken; see docs/TEXTURING-ROCM.md)
HY_NODE = COMFY_APP + "/custom_nodes/ComfyUI-Hunyuan3d-2-1.broken"
HY_PAINT = HY_NODE + "/hy3dpaint"

BLENDER = "/home/orbital/blender/blender"

GPU_ENV = {
    "HSA_OVERRIDE_GFX_VERSION": "11.5.1",
    "HIP_VISIBLE_DEVICES": "0",
    "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:8192",
    "HSA_ENABLE_SDMA": "0",
    "HSA_USE_SVM": "0",
    "HSA_FORCE_FINE_GRAIN_PCIE": "1",
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
    "PYTORCH_TUNABLEOP_ENABLED": "0",
    "MIOPEN_FIND_MODE": "FAST",
}


def gpu_env(base: dict | None = None) -> dict:
    """Return an env dict with the proven GPU vars applied over `base` (default os.environ).

    Deliberately does NOT touch LD_LIBRARY_PATH (keeping it clean is load-bearing).
    """
    env = dict(os.environ if base is None else base)
    env.update(GPU_ENV)
    return env


def apply_gpu_env() -> None:
    """Apply the GPU env to the current process (for in-process torch/ROCm use)."""
    os.environ.update(GPU_ENV)
