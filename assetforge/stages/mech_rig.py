"""Mechanical rig branch — image+mesh -> articulation graph -> build_artic.py.

Filled in during M2: part segmentation (SAM3 + lift_labels) -> VLM joint typing (MiniCPM-o)
-> geometric axis/pivot estimator (articulation.estimate) -> build_artic.py.
"""
from __future__ import annotations


def mechanical_rig(ws, cfg, mesh, log) -> dict:
    raise NotImplementedError("mechanical rig branch is M2 (in progress)")
