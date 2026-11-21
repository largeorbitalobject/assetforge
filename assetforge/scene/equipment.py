"""Equipment separation: pull held objects (sword/shield) and worn armor off a character mesh.

Reuses the shared part-seg primitive (scene/parts.py). Outputs separate editable GLBs per
equipment concept + the body remainder. A held weapon can then be socketed to the wrist bone
(from graph.json roles) so it follows the hand; armor pieces stay as swappable layers.

  separate(mesh_glb, work_dir, concepts) -> {label: part_glb}   (+ 'body')
  socket_to_wrist(part_glb, npz, roles, side, out_glb)          -> weapon parented to wrist bone
"""
from __future__ import annotations
import os
from . import parts as P

DEFAULT_EQUIPMENT = ["sword", "weapon", "shield", "staff", "armor", "helmet", "cape"]


def separate(mesh_glb, work_dir, concepts=None) -> dict:
    concepts = concepts or DEFAULT_EQUIPMENT
    os.makedirs(work_dir, exist_ok=True)
    face_labels = P.segment_parts(mesh_glb, concepts, os.path.join(work_dir, "seg"))
    return P.split_parts_blender(mesh_glb, face_labels, os.path.join(work_dir, "parts"), min_comp_frac=0.10)


def socket_to_wrist(part_glb, npz, roles, side, out_glb):
    """Parent a held part to the wrist bone of the given side ('L'/'R') as a fixed attachment."""
    import numpy as np
    arms = roles.get("arms", [])
    wrist_idx = None
    for a in arms:
        if a.get("side") == side:
            wrist_idx = a.get("wrist")
    if wrist_idx is None and arms:
        wrist_idx = arms[0].get("wrist")
    d = np.load(npz, allow_pickle=True)
    J = d["joints"].astype(float) * np.array([1, -1, -1], float)
    wrist_pos = J[wrist_idx].tolist() if wrist_idx is not None else [0, 0, 0]
    # author as a fixed-joint attachment via build_artic-style: write a tiny graph + assemble
    # (placeholder hook — full socketing happens in the rigged-character export)
    return {"wrist_index": wrist_idx, "wrist_pos": wrist_pos, "part": part_glb}
