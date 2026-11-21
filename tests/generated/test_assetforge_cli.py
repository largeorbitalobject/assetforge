"""Tests for assetforge.cli module."""
from __future__ import annotations

import types
from unittest.mock import patch, MagicMock

import pytest

from assetforge.cli import DRAFT_OVERRIDES, _cmd_run, main


class TestDraftOverrides:
    def test_keys(self):
        assert set(DRAFT_OVERRIDES.keys()) == {
            "ksampler_steps",
            "octree_resolution",
            "paint_steps",
            "target_faces",
        }

    def test_values(self):
        assert DRAFT_OVERRIDES["ksampler_steps"] == 18
        assert DRAFT_OVERRIDES["octree_resolution"] == 192
        assert DRAFT_OVERRIDES["paint_steps"] == 6
        assert DRAFT_OVERRIDES["target_faces"] == 60000


class TestCmdRun:
    def _make_args(self, **kwargs) -> types.SimpleNamespace:
        defaults = dict(
            image="test.png",
            type="creature",
            draft=False,
            octree=None,
            texture_size=None,
            target_faces=None,
            retopo=False,
            no_resume=False,
            until=None,
            only=None,
        )
        defaults.update(kwargs)
        return types.SimpleNamespace(**defaults)

    @patch("assetforge.cli.pipeline.run")
    def test_no_flags_empty_overrides(self, mock_run):
        args = self._make_args()
        result = _cmd_run(args)
        assert result == 0
        mock_run.assert_called_once_with(
            "test.png", "creature", overrides={}, resume=True, until=None, only=None
        )

    @patch("assetforge.cli.pipeline.run")
    def test_draft_applies_overrides(self, mock_run):
        args = self._make_args(draft=True)
        _cmd_run(args)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["overrides"] == DRAFT_OVERRIDES

    @patch("assetforge.cli.pipeline.run")
    def test_octree_overrides_draft(self, mock_run):
        args = self._make_args(draft=True, octree=128)
        _cmd_run(args)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["overrides"]["octree_resolution"] == 128
        # Other draft overrides should still be present
        assert call_kwargs["overrides"]["ksampler_steps"] == 18

    @patch("assetforge.cli.pipeline.run")
    def test_texture_size_override(self, mock_run):
        args = self._make_args(texture_size=512)
        _cmd_run(args)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["overrides"]["texture_size"] == 512

    @patch("assetforge.cli.pipeline.run")
    def test_target_faces_override(self, mock_run):
        args = self._make_args(target_faces=0)
        _cmd_run(args)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["overrides"]["target_faces"] == 0

    @patch("assetforge.cli.pipeline.run")
    def test_retopo_flag(self, mock_run):
        args = self._make_args(retopo=True)
        _cmd_run(args)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["overrides"]["retopo"] is True

    @patch("assetforge.cli.pipeline.run")
    def test_retopo_not_set_absent_from_overrides(self, mock_run):
        args = self._make_args(retopo=False)
        _cmd_run(args)
        call_kwargs = mock_run.call_args[1]
        assert "retopo" not in call_kwargs["overrides"]

    @patch("assetforge.cli.pipeline.run")
    def test_no_resume_sets_resume_false(self, mock_run):
        args = self._make_args(no_resume=True)
        _cmd_run(args)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["resume"] is False

    @patch("assetforge.cli.pipeline.run")
    def test_resume_default_true(self, mock_run):
        args = self._make_args(no_resume=False)
        _cmd_run(args)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["resume"] is True

    @patch("assetforge.cli.pipeline.run")
    def test_until_passed_through(self, mock_run):
        args = self._make_args(until="mesh")
        _cmd_run(args)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["until"] == "mesh"

    @patch("assetforge.cli.pipeline.run")
    def test_only_passed_through(self, mock_run):
        args = self._make_args(only="texture")
        _cmd_run(args)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["only"] == "texture"


class TestMain:
    def test_invalid_command_raises_system_exit(self):
        with pytest.raises(SystemExit):
            main(["badcommand"])

    def test_run_missing_type_raises_system_exit(self):
        with pytest.raises(SystemExit):
            main(["run", "img.png"])

    @patch("assetforge.cli.pipeline.run")
    def test_run_command_invokes_pipeline(self, mock_run):
        result = main(["run", "img.png", "--type", "creature"])
        assert result == 0
        mock_run.assert_called_once()

    @patch("assetforge.cli.pipeline.run")
    def test_run_with_draft(self, mock_run):
        main(["run", "img.png", "--type", "object", "--draft"])
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["overrides"]["ksampler_steps"] == 18

    @patch("assetforge.cli.pipeline.run")
    def test_run_with_octree(self, mock_run):
        main(["run", "img.png", "--type", "building", "--octree", "256"])
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["overrides"]["octree_resolution"] == 256

    @patch("assetforge.cli.pipeline.run")
    def test_run_with_target_faces(self, mock_run):
        main(["run", "img.png", "--type", "scene", "--target-faces", "10000"])
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["overrides"]["target_faces"] == 10000

    @patch("assetforge.cli.pipeline.run")
    def test_run_no_resume_flag(self, mock_run):
        main(["run", "img.png", "--type", "creature", "--no-resume"])
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["resume"] is False

    @patch("assetforge.cli.blender.available", return_value=False)
    @patch("assetforge.cli.comfy.is_live", return_value=False)
    def test_info_command(self, mock_live, mock_blender, capsys):
        result = main(["info"])
        assert result == 0
        captured = capsys.readouterr()
        assert "AssetForge environment" in captured.out
        assert "ComfyUI live" in captured.out
        assert "Blender available" in captured.out

    def test_invalid_type_raises_system_exit(self):
        with pytest.raises(SystemExit):
            main(["run", "img.png", "--type", "invalid_type"])