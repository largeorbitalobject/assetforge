"""Metric geometry for scene layout via MoGe-2 (preferred) with a graceful interface.

MoGe-2 returns a METRIC point map + mask + intrinsics directly from a single image — exactly the
substrate scene/layout.py needs (skip the pinhole back-projection guess). Runs on the ComfyUI ROCm
venv (pure torch + DINOv2 backbone, cached). GPU step — call when the GPU is free.

    points, mask, intrinsics = infer_points(image_path)
points: [H,W,3] metric camera-space (x right, y up, z toward camera-negative per MoGe convention).
"""
from __future__ import annotations
import numpy as np

_MODEL = None
_MODEL_ID = "Ruicheng/moge-2-vitl-normal"


def _load():
    global _MODEL
    if _MODEL is None:
        import torch
        from moge.model.v2 import MoGeModel
        _MODEL = MoGeModel.from_pretrained(_MODEL_ID).to("cuda").eval()
    return _MODEL


def infer_points(image_path: str):
    """Return (points[H,W,3] float32, mask[H,W] bool, intrinsics[3,3]). NaN where invalid."""
    import torch
    from PIL import Image
    model = _load()
    im = np.asarray(Image.open(image_path).convert("RGB"), np.float32) / 255.0
    t = torch.from_numpy(im).permute(2, 0, 1).to("cuda")
    with torch.no_grad():
        out = model.infer(t)
    pts = out["points"].detach().cpu().numpy().astype(np.float32)   # [H,W,3] metric
    mask = out["mask"].detach().cpu().numpy().astype(bool)          # valid pixels
    intr = out["intrinsics"].detach().cpu().numpy().astype(np.float32)
    pts[~mask] = np.nan
    return pts, mask, intr


def layout_from_points(points, instance_masks: dict) -> dict:
    """Bridge MoGe points -> scene/layout placement (reuses layout.fit_ground_plane/place_hero)."""
    from . import layout as L
    union = np.zeros(points.shape[:2], bool)
    for m in instance_masks.values():
        union |= m.astype(bool)
    H = points.shape[0]
    gmask = ~union
    gmask[: H * 2 // 3, :] = False
    if gmask.sum() < 50:
        gmask = ~union
    n, c, _ = L.fit_ground_plane(points, gmask)
    heroes = {name: L.place_hero(points, m, n, c) for name, m in instance_masks.items()}
    return {"ground": {"normal": [float(x) for x in n], "point": [float(x) for x in c]},
            "heroes": heroes}
