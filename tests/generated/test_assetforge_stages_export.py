"""Tests for assetforge.stages.export"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import assetforge.stages.export as export_module
from assetforge.stages.export import _latest_mesh, run


class TestLatestMesh:
    def test_returns_texture_mesh_when_available(self):
        upstream = {
            "texture": {"mesh": "/path/to/textured.glb"},
            "cleanup": {"mesh": "/path/to/cleaned.glb"},
            "mesh": {"mesh": "/path/to/raw.glb"},
        }
        assert _latest_mesh(upstream) == "/path/to/textured.glb"

    def test_returns_cleanup_mesh_when_no_texture(self):
        upstream = {
            "cleanup": {"mesh": "/path/to/cleaned.glb"},
            "mesh": {"mesh": "/path/to/raw.glb"},
        }
        assert _latest_mesh(upstream) == "/path/to/cleaned.glb"

    def test_returns_mesh_when_only_raw(self):
        upstream = {
            "mesh": {"mesh": "/path/to/raw.glb"},
        }
        assert _latest_mesh(upstream) == "/path/to/raw.glb"

    def test_raises_when_no_mesh_upstream(self):
        upstream = {}
        with pytest.raises(RuntimeError, match="no mesh produced upstream"):
            _latest_mesh(upstream)

    def test_raises_when_stages_present_but_no_mesh_key(self):
        upstream = {
            "texture": {"something_else": "value"},
            "cleanup": {"something_else": "value"},
            "mesh": {"something_else": "value"},
        }
        with pytest.raises(RuntimeError, match="no mesh produced upstream"):
            _latest_mesh(upstream)

    def test_skips_falsy_mesh_value(self):
        upstream = {
            "texture": {"mesh": ""},
            "cleanup": {"mesh": "/path/to/cleaned.glb"},
            "mesh": {"mesh": "/path/to/raw.glb"},
        }
        assert _latest_mesh(upstream) == "/path/to/cleaned.glb"

    def test_skips_none_mesh_value(self):
        upstream = {
            "texture": {"mesh": None},
            "cleanup": {"mesh": None},
            "mesh": {"mesh": "/path/to/raw.glb"},
        }
        assert _latest_mesh(upstream) == "/path/to/raw.glb"


class TestRun:
    def _make_ws(self, tmp_path):
        ws = MagicMock()
        ws.dir.return_value = tmp_path / "export"
        ws.dir.return_value.mkdir(parents=True, exist_ok=True)
        return ws

    def test_copies_asset_and_returns_path(self, tmp_path):
        # Create a fake source file
        src = tmp_path / "source.glb"
        src.write_bytes(b"fake glb data")

        ws = self._make_ws(tmp_path)
        log = MagicMock()
        upstream = {"mesh": {"mesh": str(src)}}

        with patch.object(export_module.blender, "available", return_value=False):
            result = run(ws, None, upstream, log)

        expected = tmp_path / "export" / "asset.glb"
        assert result["asset"] == str(expected)
        assert expected.exists()
        assert expected.read_bytes() == b"fake glb data"

    def test_log_called_with_final_asset_path(self, tmp_path):
        src = tmp_path / "source.glb"
        src.write_bytes(b"data")

        ws = self._make_ws(tmp_path)
        log = MagicMock()
        upstream = {"mesh": {"mesh": str(src)}}

        with patch.object(export_module.blender, "available", return_value=False):
            run(ws, None, upstream, log)

        expected_final = tmp_path / "export" / "asset.glb"
        log.assert_any_call(f"final asset -> {expected_final}")

    def test_no_preview_when_blender_unavailable(self, tmp_path):
        src = tmp_path / "source.glb"
        src.write_bytes(b"data")

        ws = self._make_ws(tmp_path)
        log = MagicMock()
        upstream = {"mesh": {"mesh": str(src)}}

        with patch.object(export_module.blender, "available", return_value=False):
            result = run(ws, None, upstream, log)

        assert "preview" not in result

    def test_no_preview_when_script_missing(self, tmp_path):
        src = tmp_path / "source.glb"
        src.write_bytes(b"data")

        ws = self._make_ws(tmp_path)
        log = MagicMock()
        upstream = {"mesh": {"mesh": str(src)}}

        with patch.object(export_module.blender, "available", return_value=True):
            with patch.object(export_module.blender, "SCRIPTS", tmp_path / "nonexistent"):
                result = run(ws, None, upstream, log)

        assert "preview" not in result

    def test_preview_rendered_when_blender_available(self, tmp_path):
        src = tmp_path / "source.glb"
        src.write_bytes(b"data")

        # Create the script path so .exists() returns True
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "render_glb.py").write_text("# fake")

        ws = self._make_ws(tmp_path)
        log = MagicMock()
        upstream = {"mesh": {"mesh": str(src)}}

        with patch.object(export_module.blender, "available", return_value=True):
            with patch.object(export_module.blender, "SCRIPTS", scripts_dir):
                with patch.object(export_module.blender, "run") as mock_run:
                    result = run(ws, None, upstream, log)

        expected_preview = tmp_path / "export" / "preview.png"
        assert result["preview"] == str(expected_preview)
        expected_final = tmp_path / "export" / "asset.glb"
        mock_run.assert_called_once_with("render_glb.py", [str(expected_final), str(expected_preview)])

    def test_render_failure_is_best_effort(self, tmp_path):
        src = tmp_path / "source.glb"
        src.write_bytes(b"data")

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "render_glb.py").write_text("# fake")

        ws = self._make_ws(tmp_path)
        log = MagicMock()
        upstream = {"mesh": {"mesh": str(src)}}

        with patch.object(export_module.blender, "available", return_value=True):
            with patch.object(export_module.blender, "SCRIPTS", scripts_dir):
                with patch.object(export_module.blender, "run", side_effect=Exception("render failed")):
                    result = run(ws, None, upstream, log)

        # Asset should still be present
        expected_final = tmp_path / "export" / "asset.glb"
        assert result["asset"] == str(expected_final)
        # Preview should NOT be in result since render failed
        assert "preview" not in result
        log.assert_any_call("preview render skipped: render failed")

    def test_picks_texture_over_cleanup(self, tmp_path):
        src_texture = tmp_path / "textured.glb"
        src_texture.write_bytes(b"textured")
        src_cleanup = tmp_path / "cleaned.glb"
        src_cleanup.write_bytes(b"cleaned")

        ws = self._make_ws(tmp_path)
        log = MagicMock()
        upstream = {
            "texture": {"mesh": str(src_texture)},
            "cleanup": {"mesh": str(src_cleanup)},
        }

        with patch.object(export_module.blender, "available", return_value=False):
            result = run(ws, None, upstream, log)

        expected = tmp_path / "export" / "asset.glb"
        assert expected.read_bytes() == b"textured"