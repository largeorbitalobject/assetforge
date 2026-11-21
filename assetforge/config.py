"""Per-asset-type configuration: which stages run, and their parameters.

Presets live in presets/*.yaml and are loaded by type. The dataclass gives defaults so the
package works even before a preset exists; YAML values override field-by-field.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml

PRESET_DIR = Path(__file__).parent / "presets"

# Input branches. creature/person -> organic rig; mechanical -> articulated (turret/door/etc.).
ASSET_TYPES = ("creature", "mechanical", "object", "building", "scene")

# Ordered pipeline stages. The matrix below decides which run per type.
ALL_STAGES = ("ingest", "mesh", "cleanup", "texture", "rig", "animate", "export")

STAGE_MATRIX = {
    "creature":   ("ingest", "mesh", "cleanup", "texture", "rig", "animate", "export"),
    "mechanical": ("ingest", "mesh", "cleanup", "texture", "rig", "animate", "export"),
    "object":     ("ingest", "mesh", "cleanup", "texture", "export"),
    "building":   ("ingest", "mesh", "cleanup", "texture", "export"),
    "scene":      ("mesh", "cleanup", "export"),  # HunyuanWorld emits its own texture; Phase 1
}

# Rig "kind" per asset type — selects the rig/animate branch (articulation graph).
RIG_KIND = {"creature": "organic", "mechanical": "mechanical"}


@dataclass
class Config:
    asset_type: str = "object"

    # mesh (Hunyuan3D-2.1 / HunyuanWorld via ComfyUI)
    mesh_backend: str = "hunyuan3d"          # hunyuan3d | hunyuanworld
    octree_resolution: int = 256             # buildings bump this for detail
    voxel_threshold: float = 0.6
    ksampler_steps: int = 30

    # cleanup (Blender)
    target_faces: int = 0                    # 0 = no decimation
    remesh: bool = False
    retopo: bool = False   # OPT-IN MeshFlow retopo (gated, non-commercial); see docs/RETOPO-MESHFLOW.md

    # texture (native Hunyuan painter — see texturing/paint.py)
    texture_size: int = 1024                 # keep SQUARE (back_project row-major bug)
    paint_resolution: int = 512
    paint_steps: int = 10
    paint_guidance: float = 3.0
    camera_azims: list[int] = field(default_factory=lambda: [0, 90, 180, 270, 0, 180])
    camera_elevs: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 90, -90])
    view_weights: list[float] = field(default_factory=lambda: [1, 0.1, 0.5, 0.1, 0.05, 0.05])

    # articulation / rig
    vlm_endpoint: str = "http://localhost:8100/v1"   # MiniCPM-o (joint typing / part naming)
    part_segmenter: str = "sam3_lift"                # sam3_lift | partfield
    default_revolute_limit: float = 3.14159

    @property
    def stages(self) -> tuple[str, ...]:
        return STAGE_MATRIX[self.asset_type]

    @property
    def kind(self) -> str | None:
        """Rig branch: 'organic' | 'mechanical' | None (static)."""
        return RIG_KIND.get(self.asset_type)


def load(asset_type: str, overrides: dict | None = None) -> Config:
    if asset_type not in ASSET_TYPES:
        raise ValueError(f"unknown asset type {asset_type!r}; choose from {ASSET_TYPES}")
    cfg = Config(asset_type=asset_type)
    preset = PRESET_DIR / f"{asset_type}.yaml"
    if preset.exists():
        data = yaml.safe_load(preset.read_text()) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    for k, v in (overrides or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def as_dict(cfg: Config) -> dict:
    return asdict(cfg)
