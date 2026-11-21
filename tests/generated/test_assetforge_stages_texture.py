"""Tests for assetforge.stages.texture module."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import assetforge.stages.texture as texture_module
from assetforge.stages.texture import run, PKG_ROOT


class TestPKGRoot:
    """Test the PKG_ROOT constant."""

    def test_pkg_root_is_path(self):
        assert isinstance(PKG_ROOT, Path)

    def test_pkg_root_exists(self):
        assert PKG_ROOT.exists()

    def test_pkg_root_contains_assetforge(self):
        assert (PKG_ROOT / "assetforge").exists()


class TestRunFunction:
    """Test the run() function logic."""

    def _make_ws(self, tmp_path):
        """Create a mock workspace."""
        ws = MagicMock()
        ws.dir.return_value = tmp_path / "texture"
        ws.manifest = {"input_image": str(tmp_path / "input.png")}
        return ws

    def _make_cfg(self):
        """Create a mock config."""
        cfg = MagicMock()
        cfg.texture_size = 1024
        cfg.paint_resolution = 512
        cfg.paint_steps = 20
        cfg.paint_guidance = 5.0
        return cfg

    def _make_upstream_with_cleanup(self, tmp_path):
        """Create upstream dict with cleanup key."""
        return {
            "cleanup": {"mesh": str(tmp_path / "mesh.glb")},
            "ingest": {"image": str(tmp_path / "image.png")},
        }

    def _make_upstream_without_cleanup(self, tmp_path):
        """Create upstream dict without cleanup key."""
        return {
            "mesh": {"mesh": str(tmp_path / "mesh.glb")},
        }

    @patch("assetforge.stages.texture.subprocess.run")
    @patch("assetforge.stages.texture.comfy.free", return_value=True)
    @patch("assetforge.stages.texture.gpu_env", return_value={"PYTHONPATH": "/existing/path"})
    def test_run_with_cleanup_upstream(
        self, mock_gpu_env, mock_comfy_free, mock_subprocess, tmp_path
    ):
        """Test run() when upstream has 'cleanup' key."""
        ws = self._make_ws(tmp_path)
        cfg = self._make_cfg()
        upstream = self._make_upstream_with_cleanup(tmp_path)
        log_calls = []
        log = lambda msg: log_calls.append(msg)

        # Create the expected output file
        out_dir = tmp_path / "texture"
        out_dir.mkdir(parents=True, exist_ok=True)
        glb_file = out_dir / "textured.glb"
        glb_file.touch()
        atlas_file = out_dir / "albedo_atlas.png"
        atlas_file.touch()

        # Mock subprocess result
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "success output"
        mock_proc.stderr = ""
        mock_subprocess.return_value = mock_proc

        result = run(ws, cfg, upstream, log)

        # Verify mesh comes from cleanup
        assert result["mesh"] == str(glb_file)
        assert result["atlas"] == str(atlas_file)

        # Verify comfy.free was called
        mock_comfy_free.assert_called_once()

        # Verify subprocess was called with correct args
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]
        assert cmd[0] == texture_module.COMFY_VENV_PY
        assert "-u" in cmd
        assert "-m" in cmd
        assert "assetforge.texturing.paint" in cmd
        assert "--mesh" in cmd
        assert "--image" in cmd
        assert "--out" in cmd
        assert "--texture-size" in cmd
        assert "--resolution" in cmd
        assert "--steps" in cmd
        assert "--guidance" in cmd

        # Verify log messages
        assert any("freed ComfyUI GPU memory" in msg for msg in log_calls)
        assert any("painting" in msg for msg in log_calls)
        assert any("textured ->" in msg for msg in log_calls)

    @patch("assetforge.stages.texture.subprocess.run")
    @patch("assetforge.stages.texture.comfy.free", return_value=False)
    @patch("assetforge.stages.texture.gpu_env", return_value={"PYTHONPATH": "/existing/path"})
    def test_run_without_cleanup_upstream(
        self, mock_gpu_env, mock_comfy_free, mock_subprocess, tmp_path
    ):
        """Test run() when upstream does NOT have 'cleanup' key."""
        ws = self._make_ws(tmp_path)
        cfg = self._make_cfg()
        upstream = self._make_upstream_without_cleanup(tmp_path)
        log_calls = []
        log = lambda msg: log_calls.append(msg)

        # Create the expected output file
        out_dir = tmp_path / "texture"
        out_dir.mkdir(parents=True, exist_ok=True)
        glb_file = out_dir / "textured.glb"
        glb_file.touch()
        atlas_file = out_dir / "albedo_atlas.png"
        atlas_file.touch()

        # Mock subprocess result
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "success output"
        mock_proc.stderr = ""
        mock_subprocess.return_value = mock_proc

        result = run(ws, cfg, upstream, log)

        # Verify mesh comes from mesh.mesh
        assert result["mesh"] == str(glb_file)
        assert result["atlas"] == str(atlas_file)

        # comfy.free was called but returned False, so no "freed" log
        mock_comfy_free.assert_called_once()
        assert not any("freed ComfyUI GPU memory" in msg for msg in log_calls)

    @patch("assetforge.stages.texture.subprocess.run")
    @patch("assetforge.stages.texture.comfy.free", return_value=True)
    @patch("assetforge.stages.texture.gpu_env", return_value={})
    def test_run_env_pythonpath_no_existing(
        self, mock_gpu_env, mock_comfy_free, mock_subprocess, tmp_path
    ):
        """Test that PYTHONPATH is set correctly when no existing PYTHONPATH."""
        ws = self._make_ws(tmp_path)
        cfg = self._make_cfg()
        upstream = self._make_upstream_with_cleanup(tmp_path)
        log = lambda msg: None

        out_dir = tmp_path / "texture"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "textured.glb").touch()
        (out_dir / "albedo_atlas.png").touch()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_subprocess.return_value = mock_proc

        run(ws, cfg, upstream, log)

        # Check env passed to subprocess
        call_kwargs = mock_subprocess.call_args[1]
        env = call_kwargs["env"]
        # PYTHONPATH should be just PKG_ROOT since no existing
        assert env["PYTHONPATH"] == str(PKG_ROOT)
        assert env["HY_NODE"] == texture_module.HY_NODE
        assert env["COMFY_APP"] == texture_module.COMFY_APP

    @patch("assetforge.stages.texture.subprocess.run")
    @patch("assetforge.stages.texture.comfy.free", return_value=True)
    @patch("assetforge.stages.texture.gpu_env", return_value={"PYTHONPATH": "/existing/path"})
    def test_run_env_pythonpath_with_existing(
        self, mock_gpu_env, mock_comfy_free, mock_subprocess, tmp_path
    ):
        """Test that PYTHONPATH is appended correctly when existing PYTHONPATH."""
        ws = self._make_ws(tmp_path)
        cfg = self._make_cfg()
        upstream = self._make_upstream_with_cleanup(tmp_path)
        log = lambda msg: None

        out_dir = tmp_path / "texture"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "textured.glb").touch()
        (out_dir / "albedo_atlas.png").touch()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_subprocess.return_value = mock_proc

        run(ws, cfg, upstream, log)

        call_kwargs = mock_subprocess.call_args[1]
        env = call_kwargs["env"]
        assert env["PYTHONPATH"] == str(PKG_ROOT) + ":/existing/path"

    @patch("assetforge.stages.texture.subprocess.run")
    @patch("assetforge.stages.texture.comfy.free", return_value=True)
    @patch("assetforge.stages.texture.gpu_env", return_value={})
    def test_run_subprocess_failure_raises(
        self, mock_gpu_env, mock_comfy_free, mock_subprocess, tmp_path
    ):
        """Test that a non-zero return code raises RuntimeError."""
        ws = self._make_ws(tmp_path)
        cfg = self._make_cfg()
        upstream = self._make_upstream_with_cleanup(tmp_path)
        log = lambda msg: None

        out_dir = tmp_path / "texture"
        out_dir.mkdir(parents=True, exist_ok=True)

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = "error output"
        mock_proc.stderr = "error stderr"
        mock_subprocess.return_value = mock_proc

        with pytest.raises(RuntimeError, match="paint failed"):
            run(ws, cfg, upstream, log)

    @patch("assetforge.stages.texture.subprocess.run")
    @patch("assetforge.stages.texture.comfy.free", return_value=True)
    @patch("assetforge.stages.texture.gpu_env", return_value={})
    def test_run_missing_glb_raises(
        self, mock_gpu_env, mock_comfy_free, mock_subprocess, tmp_path
    ):
        """Test that missing textured.glb raises RuntimeError."""
        ws = self._make_ws(tmp_path)
        cfg = self._make_cfg()
        upstream = self._make_upstream_with_cleanup(tmp_path)
        log = lambda msg: None

        out_dir = tmp_path / "texture"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Do NOT create textured.glb

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "success"
        mock_proc.stderr = ""
        mock_subprocess.return_value = mock_proc

        with pytest.raises(RuntimeError, match="textured.glb is missing"):
            run(ws, cfg, upstream, log)

    @patch("assetforge.stages.texture.subprocess.run")
    @patch("assetforge.stages.texture.comfy.free", return_value=True)
    @patch("assetforge.stages.texture.gpu_env", return_value={})
    def test_run_cmd_args_correct(
        self, mock_gpu_env, mock_comfy_free, mock_subprocess, tmp_path
    ):
        """Test that the command line arguments are constructed correctly."""
        ws = self._make_ws(tmp_path)
        cfg = self._make_cfg()
        upstream = self._make_upstream_with_cleanup(tmp_path)
        log = lambda msg: None

        out_dir = tmp_path / "texture"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "textured.glb").touch()
        (out_dir / "albedo_atlas.png").touch()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_subprocess.return_value = mock_proc

        run(ws, cfg, upstream, log)

        cmd = mock_subprocess.call_args[0][0]
        # Verify specific argument values
        mesh_idx = cmd.index("--mesh")
        assert cmd[mesh_idx + 1] == str(tmp_path / "mesh.glb")

        image_idx = cmd.index("--image")
        assert cmd[image_idx + 1] == str(tmp_path / "image.png")

        out_idx = cmd.index("--out")
        assert cmd[out_idx + 1] == str(out_dir)

        ts_idx = cmd.index("--texture-size")
        assert cmd[ts_idx + 1] == "1024"

        res_idx = cmd.index("--resolution")
        assert cmd[res_idx + 1] == "512"

        steps_idx = cmd.index("--steps")
        assert cmd[steps_idx + 1] == "20"

        guid_idx = cmd.index("--guidance")
        assert cmd[guid_idx + 1] == "5.0"

    @patch("assetforge.stages.texture.subprocess.run")
    @patch("assetforge.stages.texture.comfy.free", return_value=True)
    @patch("assetforge.stages.texture.gpu_env", return_value={})
    def test_run_timeout_passed(
        self, mock_gpu_env, mock_comfy_free, mock_subprocess, tmp_path
    ):
        """Test that timeout=3600 is passed to subprocess.run."""
        ws = self._make_ws(tmp_path)
        cfg = self._make_cfg()
        upstream = self._make_upstream_with_cleanup(tmp_path)
        log = lambda msg: None

        out_dir = tmp_path / "texture"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "textured.glb").touch()
        (out_dir / "albedo_atlas.png").touch()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_subprocess.return_value = mock_proc

        run(ws, cfg, upstream, log)

        call_kwargs = mock_subprocess.call_args[1]
        assert call_kwargs["timeout"] == 3600
        assert call_kwargs["capture_output"] is True
        assert call_kwargs["text"] is True

    @patch("assetforge.stages.texture.subprocess.run")
    @patch("assetforge.stages.texture.comfy.free", return_value=True)
    @patch("assetforge.stages.texture.gpu_env", return_value={})
    def test_run_result_dict_keys(
        self, mock_gpu_env, mock_comfy_free, mock_subprocess, tmp_path
    ):
        """Test that the result dict has exactly the expected keys."""
        ws = self._make_ws(tmp_path)
        cfg = self._make_cfg()
        upstream = self._make_upstream_with_cleanup(tmp_path)
        log = lambda msg: None

        out_dir = tmp_path / "texture"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "textured.glb").touch()
        (out_dir / "albedo_atlas.png").touch()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_subprocess.return_value = mock_proc

        result = run(ws, cfg, upstream, log)

        assert set(result.keys()) == {"mesh", "atlas"}
        assert result["mesh"].endswith("textured.glb")
        assert result["atlas"].endswith("albedo_atlas.png")

    @patch("assetforge.stages.texture.subprocess.run")
    @patch("assetforge.stages.texture.comfy.free", return_value=True)
    @patch("assetforge.stages.texture.gpu_env", return_value={})
    def test_run_stdout_written_to_stderr(
        self, mock_gpu_env, mock_comfy_free, mock_subprocess, tmp_path, capsys
    ):
        """Test that stdout from subprocess is written to sys.stderr."""
        ws = self._make_ws(tmp_path)
        cfg = self._make_cfg()
        upstream = self._make_upstream_with_cleanup(tmp_path)
        log = lambda msg: None

        out_dir = tmp_path / "texture"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "textured.glb").touch()
        (out_dir / "albedo_atlas.png").touch()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "some output text"
        mock_proc.stderr = ""
        mock_subprocess.return_value = mock_proc

        run(ws, cfg, upstream, log)

        captured = capsys.readouterr()
        assert "some output text" in captured.err

    @patch("assetforge.stages.texture.subprocess.run")
    @patch("assetforge.stages.texture.comfy.free", return_value=True)
    @patch("assetforge.stages.texture.gpu_env", return_value={})
    def test_run_empty_stdout_no_stderr_write(
        self, mock_gpu_env, mock_comfy_free, mock_subprocess, tmp_path, capsys
    ):
        """Test that empty stdout results in no stderr write."""
        ws = self._make_ws(tmp_path)
        cfg = self._make_cfg()
        upstream = self._make_upstream_with_cleanup(tmp_path)
        log = lambda msg: None

        out_dir = tmp_path / "texture"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "textured.glb").touch()
        (out_dir / "albedo_atlas.png").touch()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_subprocess.return_value = mock_proc

        run(ws, cfg, upstream, log)

        captured = capsys.readouterr()
        assert captured.err == ""