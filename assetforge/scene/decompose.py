"""Decompose a scene image into hero instance masks via SAM3 (text-promptable, ROCm-clean).

decompose(image_path, concepts) -> {label: mask[H,W] bool}. One merged mask per concept (the
union of its instances). Concepts come from a VLM caption or a default scene vocabulary.
"""
from __future__ import annotations
import os, numpy as np
from PIL import Image

_MODEL = _PROC = None
_REPO = os.environ.get("SAM3_REPO", "facebook/sam3")

DEFAULT_SCENE_CONCEPTS = ["tree", "plant", "rock", "building", "statue", "creature", "structure"]


def _load():
    global _MODEL, _PROC
    if _MODEL is None:
        import torch
        from transformers import Sam3Processor, Sam3Model
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = Sam3Model.from_pretrained(_REPO).to(dev).eval()
        _PROC = Sam3Processor.from_pretrained(_REPO)
    return _MODEL, _PROC


def decompose(image_path: str, concepts=None, threshold=0.4, min_area_frac=0.005) -> dict:
    import torch
    model, proc = _load()
    dev = next(model.parameters()).device
    img = Image.open(image_path).convert("RGB"); W, H = img.size
    concepts = concepts or DEFAULT_SCENE_CONCEPTS
    out = {}
    for c in concepts:
        inp = proc(images=img, text=c, return_tensors="pt").to(dev)
        with torch.no_grad():
            res = model(**inp)
        r = proc.post_process_instance_segmentation(
            res, threshold=threshold, mask_threshold=0.5, target_sizes=[(H, W)])[0]
        masks = r.get("masks")
        if masks is None or len(masks) == 0:
            continue
        m = masks.cpu().numpy()
        merged = (m.max(0) > 0.5)
        if merged.sum() >= min_area_frac * H * W:
            out[c] = merged
    return out


if __name__ == "__main__":
    import sys
    masks = decompose(sys.argv[1] if len(sys.argv) > 1 else "examples/inputs/landscape1.webp",
                      concepts=sys.argv[2].split(",") if len(sys.argv) > 2 else None)
    for k, v in masks.items():
        print(f"{k:12s} area {v.mean()*100:5.1f}%")
    print("found", len(masks), "hero concepts")
