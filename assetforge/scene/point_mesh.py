"""Triangulate a MoGe metric point map into a clean textured backdrop mesh (the un-torn relief).

Builds a grid mesh over valid pixels, dropping quads across depth discontinuities (anti-stretch),
UV = pixel coords, textured with the source image. This is the scene backdrop geometry that the
compositional assembler places hero objects in front of. CPU (numpy + trimesh).
"""
from __future__ import annotations
import numpy as np


def points_to_mesh(points, mask, image_path, out_glb, max_w=480, edge_rel=0.06):
    import trimesh
    from PIL import Image
    H, W = points.shape[:2]
    step = max(1, W // max_w)
    P = points[::step, ::step]; M = mask[::step, ::step]
    h, w = P.shape[:2]
    z = -P[..., 2]
    scene_extent = np.nanpercentile(z[np.isfinite(z) & M], 95) - np.nanpercentile(z[np.isfinite(z) & M], 5)
    thr = max(edge_rel * float(scene_extent), 1e-3)

    idx = -np.ones((h, w), np.int64)
    verts, uvs = [], []
    for j in range(h):
        for i in range(w):
            if M[j, i] and np.isfinite(P[j, i]).all():
                idx[j, i] = len(verts)
                verts.append(P[j, i])
                u = i / (w - 1) if w > 1 else 0.0
                v = j / (h - 1) if h > 1 else 0.0
                uvs.append([u, 1.0 - v])
    verts = np.array(verts, np.float32); uvs = np.array(uvs, np.float32)

    faces = []
    for j in range(h - 1):
        for i in range(w - 1):
            a, b, c, d = idx[j, i], idx[j, i + 1], idx[j + 1, i], idx[j + 1, i + 1]
            if min(a, b, c, d) < 0:
                continue
            quad = [P[j, i], P[j, i + 1], P[j + 1, i], P[j + 1, i + 1]]
            zr = max(q[2] for q in quad) - min(q[2] for q in quad)
            if zr > thr:        # depth discontinuity -> skip (don't stretch across it)
                continue
            faces.append([a, b, c]); faces.append([b, d, c])
    faces = np.array(faces, np.int64)

    # MoGe OpenCV (x right, y down, z forward) -> Blender Z-up (x, z, -y)
    verts = np.stack([verts[:, 0], verts[:, 2], -verts[:, 1]], axis=1).astype(np.float32)
    vis = trimesh.visual.TextureVisuals(uv=uvs, image=Image.open(image_path).convert("RGB"))
    m = trimesh.Trimesh(vertices=verts, faces=faces, visual=vis, process=False)
    m.export(out_glb)
    return out_glb, len(verts), len(faces)
