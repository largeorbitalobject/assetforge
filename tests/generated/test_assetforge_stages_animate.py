"""Unit tests for assetforge.stages.animate (pure logic only)."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from assetforge.stages.animate import _input_mesh, run


# ---------------------------------------------------------------------------
# _input_mesh
# ---------------------------------------------------------------------------

def test_input_mesh_finds_texture(tmp_path):
    ws = MagicMock()
    ws.manifest = {"input_image": str(tmp_path / "img.png")}
    upstream = {"texture": {"mesh": str(tmp_path / "tex.glb")}}
    assert _input_mesh(upstream, ws) == str(tmp_path / "tex.glb")


def test_input_mesh_finds_cleanup(tmp_path):
    ws = MagicMock()
    ws.manifest = {"input_image": str(tmp_path / "img.png")}
    upstream = {"cleanup": {"mesh": str(tmp_path / "cln.glb")}}
    assert _input_mesh(upstream, ws) == str(tmp_path / "cln.glb")


def test_input_mesh_finds_mesh(tmp_path):
    ws = MagicMock()
    ws.manifest = {"input_image": str(tmp_path / "img.png")}
    upstream = {"mesh": {"mesh": str(tmp_path / "m.glb")}}
    assert _input_mesh(upstream, ws) == str(tmp_path / "m.glb")


def test_input_mesh_prefers_texture_over_cleanup(tmp_path):
    ws = MagicMock()
    ws.manifest = {"input_image": str(tmp_path / "img.png")}
    upstream = {
        "texture": {"mesh": str(tmp_path / "tex.glb")},
        "cleanup": {"mesh": str(tmp_path / "cln.glb")},
    }
    assert _input_mesh(upstream, ws) == str(tmp_path / "tex.glb")


def test_input_mesh_prefers_cleanup_over_mesh(tmp_path):
    ws = MagicMock()
    ws.manifest = {"input_image": str(tmp_path / "img.png")}
    upstream = {
        "cleanup": {"mesh": str(tmp_path / "cln.glb")},
        "mesh": {"mesh": str(tmp_path / "m.glb")},
    }
    assert _input_mesh(upstream, ws) == str(tmp_path / "cln.glb")


def test_input_mesh_skips_stage_without_mesh_key(tmp_path):
    ws = MagicMock()
    ws.manifest = {"input_image": str(tmp_path / "img.png")}
    upstream = {
        "texture": {"other": "value"},
        "cleanup": {"mesh": str(tmp_path / "cln.glb")},
    }
    assert _input_mesh(upstream, ws) == str(tmp_path / "cln.glb")


def test_input_mesh_skips_stage_with_empty_mesh(tmp_path):
    ws = MagicMock()
    ws.manifest = {"input_image": str(tmp_path / "img.png")}
    upstream = {
        "texture": {"mesh": ""},
        "cleanup": {"mesh": str(tmp_path / "cln.glb")},
    }
    # empty string is falsy, so texture is skipped
    assert _input_mesh(upstream, ws) == str(tmp_path / "cln.glb")


def test_input_mesh_fallback_to_manifest_input_image(tmp_path):
    ws = MagicMock()
    ws.manifest = {"input_image": str(tmp_path / "img.png")}
    upstream = {}
    assert _input_mesh(upstream, ws) == str(tmp_path / "img.png")


def test_input_mesh_fallback_with_stages_but_no_mesh(tmp_path):
    ws = MagicMock()
    ws.manifest = {"input_image": str(tmp_path / "img.png")}
    upstream = {"texture": {"color": "red"}, "cleanup": {}}
    assert _input_mesh(upstream, ws) == str(tmp_path / "img.png")


# ---------------------------------------------------------------------------
# run — error paths (no blender/subprocess invoked)
# ---------------------------------------------------------------------------

def _make_ws(tmp_path):
    """Create a mock ws whose .dir() returns a real Path under tmp_path."""
    ws = MagicMock()
    ws.dir.return_value = tmp_path / "animate"
    ws.manifest = {"input_image": str(tmp_path / "input.png")}
    return ws


def _make_cfg(kind, asset_type="robot"):
    cfg = MagicMock()
    cfg.kind = kind
    cfg.asset_type = asset_type
    return cfg


def test_run_organic_missing_npz_raises(tmp_path):
    ws = _make_ws(tmp_path)
    cfg = _make_cfg("organic")
    upstream = {"rig": {"graph": "some_graph.json"}}
    log = MagicMock()
    with pytest.raises(RuntimeError, match="organic animate needs rig.npz"):
        run(ws, cfg, upstream, log)


def test_run_organic_missing_graph_raises(tmp_path):
    ws = _make_ws(tmp_path)
    cfg = _make_cfg("organic")
    upstream = {"rig": {"npz": "some.npz"}}
    log = MagicMock()
    with pytest.raises(RuntimeError, match="organic animate needs rig.npz"):
        run(ws, cfg, upstream, log)


def test_run_organic_missing_both_raises(tmp_path):
    ws = _make_ws(tmp_path)
    cfg = _make_cfg("organic")
    upstream = {"rig": {}}
    log = MagicMock()
    with pytest.raises(RuntimeError, match="organic animate needs rig.npz"):
        run(ws, cfg, upstream, log)


def test_run_organic_no_rig_key_raises(tmp_path):
    ws = _make_ws(tmp_path)
    cfg = _make_cfg("organic")
    upstream = {}
    log = MagicMock()
    with pytest.raises(RuntimeError, match="organic animate needs rig.npz"):
        run(ws, cfg, upstream, log)


def test_run_mechanical_missing_blend_raises(tmp_path):
    ws = _make_ws(tmp_path)
    cfg = _make_cfg("mechanical")
    upstream = {"rig": {"graph": "some_graph.json"}}
    log = MagicMock()
    with pytest.raises(RuntimeError, match="mechanical animate needs rig.blend"):
        run(ws, cfg, upstream, log)


def test_run_mechanical_missing_graph_raises(tmp_path):
    ws = _make_ws(tmp_path)
    cfg = _make_cfg("mechanical")
    upstream = {"rig": {"blend": "some.blend"}}
    log = MagicMock()
    with pytest.raises(RuntimeError, match="mechanical animate needs rig.blend"):
        run(ws, cfg, upstream, log)


def test_run_mechanical_missing_both_raises(tmp_path):
    ws = _make_ws(tmp_path)
    cfg = _make_cfg("mechanical")
    upstream = {"rig": {}}
    log = MagicMock()
    with pytest.raises(RuntimeError, match="mechanical animate needs rig.blend"):
        run(ws, cfg, upstream, log)


def test_run_mechanical_no_rig_key_raises(tmp_path):
    ws = _make_ws(tmp_path)
    cfg = _make_cfg("mechanical")
    upstream = {}
    log = MagicMock()
    with pytest.raises(RuntimeError, match="mechanical animate needs rig.blend"):
        run(ws, cfg, upstream, log)


def test_run_unknown_kind_raises_not_implemented(tmp_path):
    ws = _make_ws(tmp_path)
    cfg = _make_cfg("unknown_type", asset_type="spaceship")
    upstream = {"rig": {}}
    log = MagicMock()
    with pytest.raises(NotImplementedError, match="spaceship"):
        run(ws, cfg, upstream, log)


def test_run_unknown_kind_message_includes_asset_type(tmp_path):
    ws = _make_ws(tmp_path)
    cfg = _make_cfg("wat", asset_type="mech_bot")
    upstream = {}
    log = MagicMock()
    with pytest.raises(NotImplementedError) as exc_info:
        run(ws, cfg, upstream, log)
    assert "mech_bot" in str(exc_info.value)