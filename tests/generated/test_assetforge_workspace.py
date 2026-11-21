"""Tests for assetforge.workspace module."""
from pathlib import Path

import pytest

from assetforge.workspace import (
    STAGE_DIRS,
    Workspace,
    fingerprint,
    slugify,
)


# --- slugify ---


def test_slugify_simple_name():
    assert slugify("myasset") == "myasset"


def test_slugify_uppercase_lowered():
    assert slugify("MyAsset") == "myasset"


def test_slugify_replaces_spaces():
    assert slugify("my asset") == "my-asset"


def test_slugify_replaces_special_chars():
    assert slugify("my@asset#name!") == "my-asset-name"


def test_slugify_strips_extension():
    assert slugify("image.png") == "image"


def test_slugify_strips_path():
    assert slugify("/some/path/file.obj") == "file"


def test_slugify_only_special_chars_returns_asset():
    assert slugify("!!!") == "asset"


def test_slugify_preserves_underscores_and_hyphens():
    assert slugify("my_asset-name") == "my_asset-name"


def test_slugify_strips_leading_trailing_hyphens():
    assert slugify("-hello-") == "hello"


# --- fingerprint ---


def test_fingerprint_is_16_hex_chars():
    result = fingerprint({"a": 1})
    assert len(result) == 16
    assert all(c in "0123456789abcdef" for c in result)


def test_fingerprint_deterministic():
    assert fingerprint({"a": 1}, "config") == fingerprint({"a": 1}, "config")


def test_fingerprint_different_inputs_different_hash():
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_fingerprint_order_matters():
    assert fingerprint("x", "y") != fingerprint("y", "x")


def test_fingerprint_no_arguments():
    result = fingerprint()
    assert len(result) == 16


def test_fingerprint_sort_keys_gives_same_result():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


# --- STAGE_DIRS ---


def test_stage_dirs_has_seven_stages():
    assert len(STAGE_DIRS) == 7


def test_stage_dirs_keys():
    expected = {"ingest", "mesh", "cleanup", "texture", "rig", "animate", "export"}
    assert set(STAGE_DIRS.keys()) == expected


def test_stage_dirs_values_are_numbered_dirs():
    expected = {
        "00_input",
        "01_mesh",
        "02_clean",
        "03_texture",
        "04_rig",
        "05_anim",
        "06_final",
    }
    assert set(STAGE_DIRS.values()) == expected


# --- Workspace.create ---


def test_workspace_create_makes_root_dir(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("myasset.png", "character", runs_root=runs_root)
    assert ws.root == runs_root / "myasset"
    assert ws.root.is_dir()


def test_workspace_create_makes_stage_dirs(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("myasset.png", "character", runs_root=runs_root)
    for stage_dir in STAGE_DIRS.values():
        assert (ws.root / stage_dir).is_dir(), f"Missing stage dir: {stage_dir}"


def test_workspace_create_makes_manifest(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("myasset.png", "character", runs_root=runs_root)
    assert ws.manifest_path.exists()
    assert ws.manifest["asset_type"] == "character"
    assert ws.manifest["stages"] == {}


def test_workspace_create_reuses_existing_manifest(tmp_path):
    runs_root = tmp_path / "runs"
    ws1 = Workspace.create("myasset.png", "character", runs_root=runs_root)
    ws1.record("ingest", "fp1", {"file": "input.obj"})

    ws2 = Workspace.create("myasset.png", "character", runs_root=runs_root)
    assert ws2.manifest["stages"]["ingest"]["fingerprint"] == "fp1"
    assert ws2.manifest["stages"]["ingest"]["status"] == "completed"


# --- Workspace.dir ---


def test_workspace_dir_returns_correct_path(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("test.png", "prop", runs_root=runs_root)
    assert ws.dir("mesh") == ws.root / "01_mesh"
    assert ws.dir("export") == ws.root / "06_final"


def test_workspace_dir_raises_for_unknown_stage(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("test.png", "prop", runs_root=runs_root)
    with pytest.raises(KeyError):
        ws.dir("nonexistent")


# --- Workspace.record ---


def test_workspace_record_stores_stage_data(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("test.png", "prop", runs_root=runs_root)
    ws.record("mesh", "abc123", {"mesh": "out.obj"})

    stage = ws.manifest["stages"]["mesh"]
    assert stage["status"] == "completed"
    assert stage["fingerprint"] == "abc123"
    assert stage["outputs"] == {"mesh": "out.obj"}
    assert "updated" in stage


def test_workspace_record_persists_to_disk(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("test.png", "prop", runs_root=runs_root)
    ws.record("mesh", "abc123", {"mesh": "out.obj"})

    # Reload from disk
    import json

    raw = json.loads(ws.manifest_path.read_text())
    assert raw["stages"]["mesh"]["fingerprint"] == "abc123"


def test_workspace_record_custom_status(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("test.png", "prop", runs_root=runs_root)
    ws.record("mesh", "abc123", {}, status="failed")
    assert ws.manifest["stages"]["mesh"]["status"] == "failed"


# --- Workspace.is_fresh ---


def test_is_fresh_returns_true_when_matching(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("test.png", "prop", runs_root=runs_root)
    ws.record("mesh", "abc123", {"mesh": "out.obj"})
    assert ws.is_fresh("mesh", "abc123") is True


def test_is_fresh_returns_false_when_fingerprint_differs(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("test.png", "prop", runs_root=runs_root)
    ws.record("mesh", "abc123", {"mesh": "out.obj"})
    assert ws.is_fresh("mesh", "different") is False


def test_is_fresh_returns_false_for_missing_stage(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("test.png", "prop", runs_root=runs_root)
    assert ws.is_fresh("mesh", "abc123") is False


def test_is_fresh_returns_false_when_not_completed(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("test.png", "prop", runs_root=runs_root)
    ws.record("mesh", "abc123", {}, status="failed")
    assert ws.is_fresh("mesh", "abc123") is False


# --- Workspace.outputs ---


def test_outputs_returns_recorded_outputs(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("test.png", "prop", runs_root=runs_root)
    ws.record("mesh", "abc", {"file": "result.obj", "vert_count": 1000})
    assert ws.outputs("mesh") == {"file": "result.obj", "vert_count": 1000}


def test_outputs_returns_empty_dict_for_missing_stage(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("test.png", "prop", runs_root=runs_root)
    assert ws.outputs("mesh") == {}


# --- Workspace.save ---


def test_save_writes_valid_json(tmp_path):
    runs_root = tmp_path / "runs"
    ws = Workspace.create("test.png", "prop", runs_root=runs_root)
    ws.manifest["custom_key"] = "value"
    ws.save()

    import json

    data = json.loads(ws.manifest_path.read_text())
    assert data["custom_key"] == "value"