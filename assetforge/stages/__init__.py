"""Pipeline stages. Each exposes run(ws, cfg, upstream, log) -> dict[str, str] (named outputs).

Implemented: mesh, texture, ingest, cleanup, export.
Scaffolded (Phase 3 — needs the SkeletonConfig generalization): rig, animate.
"""
from . import ingest, mesh, cleanup, texture, rig, animate, export

REGISTRY = {
    "ingest": ingest.run,
    "mesh": mesh.run,
    "cleanup": cleanup.run,
    "texture": texture.run,
    "rig": rig.run,
    "animate": animate.run,
    "export": export.run,
}
