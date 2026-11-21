"""assetforge CLI.

    assetforge run IMG --type creature|object|building|scene [--until STAGE] [--only STAGE] [--no-resume]
    assetforge info                 # show environment / readiness
"""
from __future__ import annotations
import argparse
import sys

from . import config as cfgmod
from . import comfy, blender, pipeline
from .rocm_env import COMFY_VENV_PY, HY_PAINT


DRAFT_OVERRIDES = {
    "ksampler_steps": 18,        # mesh: 30 -> 18
    "octree_resolution": 192,    # mesh: 256 -> 192
    "paint_steps": 6,            # texture: 10 -> 6
    "target_faces": 60000,       # aggressive decimation — fast unwrap, lower fidelity (preview-only)
}


def _cmd_run(a) -> int:
    overrides = {}
    if a.draft:
        overrides.update(DRAFT_OVERRIDES)
    if a.octree:
        overrides["octree_resolution"] = a.octree
    if a.texture_size:
        overrides["texture_size"] = a.texture_size
    if a.target_faces is not None:
        overrides["target_faces"] = a.target_faces  # 0 = no decimation (full budget)
    if a.retopo:
        overrides["retopo"] = True
    pipeline.run(a.image, a.type, overrides=overrides, resume=not a.no_resume,
                 until=a.until, only=a.only)
    return 0


def _cmd_info(a) -> int:
    import os.path
    print("AssetForge environment")
    print(f"  ComfyUI live (:8188)   : {comfy.is_live()}")
    print(f"  Blender available      : {blender.available()}")
    print(f"  ComfyUI venv python    : {COMFY_VENV_PY}  ({'ok' if os.path.exists(COMFY_VENV_PY) else 'MISSING'})")
    print(f"  Hunyuan paint install  : {HY_PAINT}  ({'ok' if os.path.isdir(HY_PAINT) else 'MISSING'})")
    print(f"  asset types            : {', '.join(cfgmod.ASSET_TYPES)}")
    for t in cfgmod.ASSET_TYPES:
        print(f"    {t:9s}: {' -> '.join(cfgmod.STAGE_MATRIX[t])}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="assetforge", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="generate an asset from an image")
    r.add_argument("image")
    r.add_argument("--type", required=True, choices=cfgmod.ASSET_TYPES)
    r.add_argument("--until", help="stop after this stage")
    r.add_argument("--only", help="run only this stage")
    r.add_argument("--no-resume", action="store_true", help="re-run all stages")
    r.add_argument("--retopo", action="store_true", help="MeshFlow clean retopo in cleanup (opt-in; needs gated weights)")
    r.add_argument("--draft", action="store_true",
                   help="fast/low-fidelity preset (fewer steps, lighter mesh) for quick iteration")
    r.add_argument("--octree", type=int, help="override octree_resolution (mesh detail)")
    r.add_argument("--texture-size", type=int, help="override texture atlas size (keep square)")
    r.add_argument("--target-faces", type=int, default=None,
                   help="override decimation budget; 0 = full budget / no decimation")
    r.set_defaults(func=_cmd_run)

    i = sub.add_parser("info", help="show environment readiness")
    i.set_defaults(func=_cmd_info)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
