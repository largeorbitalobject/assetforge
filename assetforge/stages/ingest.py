"""Ingest: stage the input image and isolate the subject.

scene -> passthrough (HunyuanWorld wants the whole scene).
creature/object/mechanical/building -> rembg cutout (Hunyuan3D needs an ISOLATED subject;
scene/background bleed produces junk meshes). Falls back to the raw image if rembg fails.
"""
from __future__ import annotations
import shutil, subprocess
from pathlib import Path
from ..rocm_env import COMFY_VENV_PY

_REMBG = ("import sys;from rembg import remove;from PIL import Image;"
          "im=Image.open(sys.argv[1]).convert('RGBA');"
          "remove(im).save(sys.argv[2])")


def run(ws, cfg, upstream, log) -> dict:
    src = Path(ws.manifest["input_image"])
    if not src.exists():
        raise FileNotFoundError(src)
    if cfg.asset_type == "scene":
        dst = ws.dir("ingest") / ("subject" + src.suffix.lower())
        shutil.copy2(src, dst)
        return {"image": str(dst)}

    dst = ws.dir("ingest") / "subject.png"
    try:
        subprocess.run([COMFY_VENV_PY, "-c", _REMBG, str(src), str(dst)],
                       check=True, capture_output=True, text=True, timeout=300)
        log("isolated subject via rembg")
        return {"image": str(dst)}
    except Exception as e:
        log(f"rembg failed ({str(e)[:120]}); using image as-is")
        dst = ws.dir("ingest") / ("subject" + src.suffix.lower())
        shutil.copy2(src, dst)
        return {"image": str(dst)}
