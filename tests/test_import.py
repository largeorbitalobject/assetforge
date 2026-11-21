"""Smoke tests that don't require torch/GPU — just that the package wires together."""
import assetforge
from assetforge import config, workspace, comfy, blender, pipeline
from assetforge.stages import REGISTRY


def test_version():
    assert assetforge.__version__


def test_config_types():
    for t in config.ASSET_TYPES:
        cfg = config.load(t)
        assert cfg.asset_type == t
        assert cfg.stages  # non-empty stage list
        # every stage in the matrix has a registered implementation
        for s in cfg.stages:
            assert s in REGISTRY


def test_creature_full_chain():
    assert config.STAGE_MATRIX["creature"][-1] == "export"
    assert "rig" in config.STAGE_MATRIX["creature"]
    assert "rig" not in config.STAGE_MATRIX["object"]


def test_fingerprint_stable():
    a = workspace.fingerprint({"x": 1}, {"y": 2})
    b = workspace.fingerprint({"x": 1}, {"y": 2})
    assert a == b
