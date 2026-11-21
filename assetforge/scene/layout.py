"""Metric layout for compositional scenes: depth map + instance masks -> per-hero 3D placement.

Pure numpy (CPU). Depth comes from Metric3D / Depth-Anything-V2 (on-box via controlnet_aux);
this module turns a depth map + per-instance masks into a ground plane + each hero's
position / scale / footprint, so the Blender assembler can place generated hero GLBs.

Pinhole back-projection with an assumed FOV (monocular depth has scale ambiguity; absolute scale
is approximate and meant for art-directed nudging — see docs/ENVIRONMENTS.md ceilings).
"""
from __future__ import annotations
import numpy as np


def backproject(depth: np.ndarray, fov_deg: float = 55.0) -> np.ndarray:
    """depth[H,W] -> points[H,W,3] in camera space (x right, y up, z forward = -into screen)."""
    H, W = depth.shape
    f = 0.5 * W / np.tan(np.radians(fov_deg) / 2)
    cx, cy = W / 2.0, H / 2.0
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    X = (xs - cx) / f * depth
    Y = -(ys - cy) / f * depth   # image y down -> world y up
    Z = -depth                   # forward into scene
    return np.stack([X, Y, Z], axis=-1)


def fit_ground_plane(points: np.ndarray, mask: np.ndarray | None = None,
                     iters: int = 200, thresh: float = 0.03, seed: int = 0):
    """RANSAC plane fit. Returns (normal[3], point_on_plane[3], inlier_mask_flat)."""
    P = points.reshape(-1, 3)
    if mask is not None:
        idx = np.where(mask.reshape(-1))[0]
    else:
        idx = np.arange(len(P))
    P = points.reshape(-1, 3)
    rng = np.random.RandomState(seed)
    best_n, best_c, best_in = None, None, -1
    cand = idx[np.isfinite(points.reshape(-1, 3)[idx]).all(1)]
    if len(cand) < 3:
        return np.array([0, 1, 0.0]), np.zeros(3), np.zeros(len(P), bool)
    for _ in range(iters):
        s = P[rng.choice(cand, 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln
        d = np.abs((P[cand] - s[0]) @ n)
        inl = (d < thresh).sum()
        if inl > best_in:
            best_in, best_n, best_c = inl, n, s[0]
    if best_n[1] < 0:   # orient normal up-ish
        best_n = -best_n
    inmask = np.zeros(len(P), bool)
    dall = np.abs((P - best_c) @ best_n)
    inmask[np.isfinite(dall) & (dall < thresh)] = True
    return best_n, best_c, inmask


def place_hero(points: np.ndarray, mask: np.ndarray, ground_n, ground_c) -> dict:
    """For one instance mask, return {pos, scale, footprint, height} on the ground plane."""
    pm = points[mask.astype(bool)]
    pm = pm[np.isfinite(pm).all(1)]
    if len(pm) == 0:
        return {"pos": [0, 0, 0], "scale": 1.0, "footprint": 0.0, "height": 0.0}
    # height along ground normal
    h = (pm - ground_c) @ ground_n
    height = float(h.max() - h.min())
    # footprint center = centroid projected to ground; pos = lowest-contact point
    centroid = pm.mean(0)
    base = pm[np.argmin(h)]
    pos = base - ((base - ground_c) @ ground_n) * ground_n  # drop to plane
    # lateral footprint extent
    tangential = pm - np.outer(h, ground_n) - ground_c
    footprint = float(np.linalg.norm(tangential - tangential.mean(0), axis=1).max() * 2)
    return {"pos": [float(pos[0]), float(pos[2]), float(height * 0.0)],  # x, z(forward)->y, on-ground
            "scale": 1.0, "footprint": footprint, "height": height,
            "centroid": [float(c) for c in centroid]}


def layout(depth: np.ndarray, instance_masks: dict[str, np.ndarray], fov_deg: float = 55.0) -> dict:
    """depth + named instance masks -> {ground:{normal,point}, heroes:{name:placement}}."""
    pts = backproject(depth, fov_deg)
    # ground = lower-third pixels not in any instance (rough), else all
    union = np.zeros(depth.shape, bool)
    for m in instance_masks.values():
        union |= m.astype(bool)
    H = depth.shape[0]
    gmask = (~union)
    gmask[: H * 2 // 3, :] = False   # prefer lower image for ground
    if gmask.sum() < 50:
        gmask = ~union
    n, c, _ = fit_ground_plane(pts, gmask)
    heroes = {name: place_hero(pts, m, n, c) for name, m in instance_masks.items()}
    return {"ground": {"normal": [float(x) for x in n], "point": [float(x) for x in c],
                       "fov_deg": fov_deg},
            "heroes": heroes}


if __name__ == "__main__":
    # self-test on a synthetic scene: a ground plane + a box and a sphere at known depths
    H, W = 120, 160
    depth = np.full((H, W), 8.0, np.float32)
    # ground: depth increases toward top (farther) — simple ramp in lower half
    for y in range(H):
        depth[y, :] = 4.0 + (H - y) / H * 6.0
    # box instance: a near patch
    boxm = np.zeros((H, W), bool); boxm[70:95, 40:70] = True; depth[boxm] = 3.0
    # sphere instance: farther patch
    spm = np.zeros((H, W), bool); spm[55:75, 100:130] = True; depth[spm] = 5.0
    out = layout(depth, {"box": boxm, "sphere": spm})
    import json
    print(json.dumps(out, indent=2))
    g = out["ground"]["normal"]
    print("ground normal ~up?", abs(g[1]) > 0.7, g)
    print("box nearer than sphere (forward z larger/less-neg)?",
          out["heroes"]["box"]["centroid"][2], ">", out["heroes"]["sphere"]["centroid"][2])
