"""Rig stage — dispatches on cfg.kind to the organic or mechanical articulation branch.

Both branches write <run>/04_rig/graph.json + a rigged GLB + a preview render.
  organic    : RigAnything -> roles auto-detect -> build_rig.py (skinned skeleton)
  mechanical : part-seg + VLM + geometric estimator -> build_artic.py (typed joints)
"""
from __future__ import annotations
import glob
import numpy as np
from pathlib import Path

from .. import blender
from ..articulation import roles as rolesmod
from ..articulation.graph import ArticulationGraph
from ..articulation.riganything import run_riganything


def _input_mesh(upstream, ws) -> str:
    for s in ("texture", "cleanup", "mesh"):
        if s in upstream and upstream[s].get("mesh"):
            return upstream[s]["mesh"]
    return ws.manifest["input_image"]


def _organic(ws, cfg, mesh, log) -> dict:
    rigdir = ws.dir("rig")
    # reuse a pre-seeded npz if present (resume / test), else run RigAnything
    existing = glob.glob(str(rigdir / "*.npz"))
    if existing:
        npz = existing[0]; log(f"reusing skeleton npz {Path(npz).name}")
    else:
        log("running RigAnything (organic auto-rig) ...")
        npz = run_riganything(mesh, str(rigdir))
        log(f"skeleton -> {Path(npz).name}")

    d = np.load(npz, allow_pickle=True)
    roles = rolesmod.classify_roles(d["joints"], d["parents"])
    H = rolesmod.height_zup(d["joints"])
    log(f"roles: pelvis={roles['pelvis']} spine={roles['spine']} "
        f"legs={len(roles['legs'])} arms={len(roles['arms'])} tail={len(roles['tail'])} H={H:.2f}")

    # organic articulation graph (skin + roles)
    Jflip = d["joints"].astype(float).copy(); Jflip[:, 1] *= -1; Jflip[:, 2] *= -1
    g = ArticulationGraph(
        kind="organic", mesh=str(mesh),
        skin={"joints_xyz": Jflip.tolist(), "parents": [int(p) for p in d["parents"]],
              "weights_npz": npz, "height": H},
        roles=roles,
    )
    graph_path = str(rigdir / "graph.json"); g.save(graph_path)

    rigged = rigdir / "rigged.glb"; preview = rigdir / "rig_preview.png"
    blender.run("build_rig.py", [npz, str(mesh), str(preview), str(rigged)])
    log(f"rigged -> {rigged.name}")
    return {"mesh": str(rigged), "graph": graph_path, "npz": npz}


def run(ws, cfg, upstream, log) -> dict:
    mesh = _input_mesh(upstream, ws)
    kind = cfg.kind
    if kind == "organic":
        return _organic(ws, cfg, mesh, log)
    if kind == "mechanical":
        from .mech_rig import mechanical_rig
        return mechanical_rig(ws, cfg, mesh, log)
    raise NotImplementedError(f"rig stage: asset_type {cfg.asset_type!r} has no rig branch")
