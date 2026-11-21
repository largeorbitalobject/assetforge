"""Run headless Blender scripts from blender_scripts/ as subprocesses.

Each script takes its file args after a `--` separator (Blender convention). We pass the GPU
env so Cycles/torch-free scripts behave; LD_LIBRARY_PATH is left clean by gpu_env().
"""
from __future__ import annotations
from pathlib import Path
import subprocess

from .rocm_env import BLENDER, gpu_env

SCRIPTS = Path(__file__).parent / "blender_scripts"


def run(script: str, args: list[str], timeout: float = 1800) -> subprocess.CompletedProcess:
    """Run blender_scripts/<script> headless with the given trailing args.

    Raises CalledProcessError on non-zero exit; returns the completed process otherwise.
    """
    script_path = SCRIPTS / script if not Path(script).is_absolute() else Path(script)
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    cmd = [BLENDER, "-b", "-noaudio", "-P", str(script_path), "--", *map(str, args)]
    return subprocess.run(
        cmd, env=gpu_env(), timeout=timeout,
        check=True, capture_output=True, text=True,
    )


def available() -> bool:
    return Path(BLENDER).exists()
