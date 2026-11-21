"""Run workspace: a per-asset folder with a manifest that makes the pipeline resumable.

Layout:  assets/runs/<slug>/
    00_input/  01_mesh/  02_clean/  03_texture/  04_rig/  05_anim/  06_final/
    manifest.json

Each stage records status + an input fingerprint (config + upstream outputs). On --resume a
stage is skipped when its fingerprint is unchanged and it previously completed.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import time

STAGE_DIRS = {
    "ingest": "00_input",
    "mesh": "01_mesh",
    "cleanup": "02_clean",
    "texture": "03_texture",
    "rig": "04_rig",
    "animate": "05_anim",
    "export": "06_final",
}

RUNS_ROOT = Path("assets/runs")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(name).stem).strip("-").lower()
    return s or "asset"


@dataclass
class Workspace:
    root: Path
    manifest_path: Path
    manifest: dict

    @classmethod
    def create(cls, image: str, asset_type: str, runs_root: Path = RUNS_ROOT) -> "Workspace":
        root = Path(runs_root) / slugify(image)
        root.mkdir(parents=True, exist_ok=True)
        for d in STAGE_DIRS.values():
            (root / d).mkdir(exist_ok=True)
        mpath = root / "manifest.json"
        if mpath.exists():
            manifest = json.loads(mpath.read_text())
        else:
            manifest = {
                "input_image": str(Path(image).resolve()),
                "asset_type": asset_type,
                "created": time.time(),
                "stages": {},   # name -> {status, fingerprint, outputs, updated}
            }
        ws = cls(root=root, manifest_path=mpath, manifest=manifest)
        ws.save()
        return ws

    def dir(self, stage: str) -> Path:
        return self.root / STAGE_DIRS[stage]

    def save(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def record(self, stage: str, fingerprint: str, outputs: dict, status: str = "completed") -> None:
        self.manifest["stages"][stage] = {
            "status": status,
            "fingerprint": fingerprint,
            "outputs": outputs,
            "updated": time.time(),
        }
        self.save()

    def is_fresh(self, stage: str, fingerprint: str) -> bool:
        s = self.manifest["stages"].get(stage)
        return bool(s and s.get("status") == "completed" and s.get("fingerprint") == fingerprint)

    def outputs(self, stage: str) -> dict:
        return self.manifest["stages"].get(stage, {}).get("outputs", {})


def fingerprint(*parts) -> str:
    """Stable hash of config + upstream outputs to decide resume-skip."""
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]
