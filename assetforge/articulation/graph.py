"""The articulation graph: the contract between rig INFERENCE and Blender AUTHORING.

A strict superset of a URDF (parts + typed joints + axes + limits) and a skinned skeleton
(skin + roles). Both the organic and mechanical rig branches write one of these to
<run>/04_rig/graph.json; build_artic.py / build_rig.py read it to author the rig.

Joint types: fixed | revolute | prismatic | ball.
Coordinates are mesh/Blender space (Z-up). See docs/PHASE3-RIGGING.md.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json

JOINT_TYPES = ("fixed", "revolute", "prismatic", "ball")


@dataclass
class Joint:
    id: str
    parent: str            # part id, or "world"
    child: str             # part id
    type: str              # one of JOINT_TYPES
    origin: list           # [x,y,z] pivot
    axis: list             # [ax,ay,az] unit; revolute=rot axis, prismatic=slide dir
    limits: dict | None = None     # {"min":..,"max":..} radians or meters; None = free
    rest: float = 0.0
    motion: dict | None = None     # {"preset": "scan"|"elevate"|"open"|..., ...}

    def __post_init__(self):
        if self.type not in JOINT_TYPES:
            raise ValueError(f"bad joint type {self.type!r}; expected {JOINT_TYPES}")


@dataclass
class Part:
    id: str
    object: str | None = None      # named mesh object in the parts GLB (mechanical)
    vgroup: str | None = None      # vertex group (organic / single-mesh split)
    face_ids: list | None = None   # face membership on a single mesh
    centroid: list | None = None
    bbox: list | None = None       # [[min],[max]]
    principal_axes: list | None = None


@dataclass
class ArticulationGraph:
    kind: str                       # "organic" | "mechanical"
    mesh: str
    parts: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    up_axis: list = field(default_factory=lambda: [0, 0, 1])
    skin: dict | None = None        # organic: {joints_xyz, parents, weights_npz}
    roles: dict | None = None       # organic: {pelvis, spine[], neck, legs[], arms[], tail[]}

    # ---- IO ----
    def to_dict(self) -> dict:
        d = {"kind": self.kind, "mesh": self.mesh, "up_axis": self.up_axis,
             "parts": [asdict(p) if isinstance(p, Part) else p for p in self.parts],
             "joints": [asdict(j) if isinstance(j, Joint) else j for j in self.joints]}
        if self.skin is not None:
            d["skin"] = self.skin
        if self.roles is not None:
            d["roles"] = self.roles
        return d

    def save(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "ArticulationGraph":
        d = json.load(open(path))
        return cls(
            kind=d["kind"], mesh=d.get("mesh", ""), up_axis=d.get("up_axis", [0, 0, 1]),
            parts=[Part(**p) if isinstance(p, dict) else p for p in d.get("parts", [])],
            joints=[Joint(**j) if isinstance(j, dict) else j for j in d.get("joints", [])],
            skin=d.get("skin"), roles=d.get("roles"),
        )

    def validate(self) -> list[str]:
        """Return a list of problems; empty = OK."""
        errs = []
        if self.kind not in ("organic", "mechanical"):
            errs.append(f"bad kind {self.kind!r}")
        part_ids = {p.id if isinstance(p, Part) else p["id"] for p in self.parts}
        for j in self.joints:
            jj = j if isinstance(j, Joint) else Joint(**j)
            if jj.parent not in part_ids and jj.parent != "world":
                errs.append(f"joint {jj.id}: parent {jj.parent!r} not a part")
            if jj.child not in part_ids:
                errs.append(f"joint {jj.id}: child {jj.child!r} not a part")
        if self.kind == "organic" and self.skin is None:
            errs.append("organic graph missing skin block")
        return errs
