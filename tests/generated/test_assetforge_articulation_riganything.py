import os
import time
from unittest.mock import patch, MagicMock

import pytest

import assetforge.articulation.riganything as ra


def test_module_constants():
    assert ra.RA_DIR == "/home/orbital/projects/orbitis-station/tools/RigAnything"
    assert ra.CKPT == "ckpt/riganything_ckpt.pt"
    assert ra.CONFIG == "config.yaml"


def test_run_riganything_creates_out_dir(tmp_path):
    out_dir = tmp_path / "out"
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("")

    mock_proc = MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(ra, "subprocess") as mock_subprocess, \
         patch.object(ra, "gpu_env", return_value={}):

        def run_side_effect(cmd, **kwargs):
            os.makedirs(out_dir, exist_ok=True)
            (out_dir / "result.npz").touch()
            return mock_proc

        mock_subprocess.run.side_effect = run_side_effect
        ra.run_riganything(str(mesh), str(out_dir))

    assert out_dir.exists()


def test_run_riganything_nonzero_returncode_raises(tmp_path):
    out_dir = tmp_path / "out"
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("")

    mock_proc = MagicMock(returncode=1, stdout="some stdout", stderr="some stderr")

    with patch.object(ra, "subprocess") as mock_subprocess, \
         patch.object(ra, "gpu_env", return_value={}):
        mock_subprocess.run.return_value = mock_proc
        with pytest.raises(RuntimeError, match="RigAnything failed"):
            ra.run_riganything(str(mesh), str(out_dir))


def test_run_riganything_error_includes_output(tmp_path):
    out_dir = tmp_path / "out"
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("")

    mock_proc = MagicMock(returncode=2, stdout="STDOUT_INFO", stderr="STDERR_INFO")

    with patch.object(ra, "subprocess") as mock_subprocess, \
         patch.object(ra, "gpu_env", return_value={}):
        mock_subprocess.run.return_value = mock_proc
        with pytest.raises(RuntimeError) as exc_info:
            ra.run_riganything(str(mesh), str(out_dir))

    assert "STDOUT_INFO" in str(exc_info.value)
    assert "STDERR_INFO" in str(exc_info.value)
    assert "rc=2" in str(exc_info.value)


def test_run_riganything_no_npz_raises(tmp_path):
    out_dir = tmp_path / "out"
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("")

    mock_proc = MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(ra, "subprocess") as mock_subprocess, \
         patch.object(ra, "gpu_env", return_value={}):
        mock_subprocess.run.return_value = mock_proc
        with pytest.raises(RuntimeError, match="produced no npz"):
            ra.run_riganything(str(mesh), str(out_dir))


def test_run_riganything_returns_npz_path(tmp_path):
    out_dir = tmp_path / "out"
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("")

    mock_proc = MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(ra, "subprocess") as mock_subprocess, \
         patch.object(ra, "gpu_env", return_value={}):

        def run_side_effect(cmd, **kwargs):
            os.makedirs(out_dir, exist_ok=True)
            (out_dir / "skeleton.npz").touch()
            return mock_proc

        mock_subprocess.run.side_effect = run_side_effect
        result = ra.run_riganything(str(mesh), str(out_dir))

    assert result == str(out_dir / "skeleton.npz")


def test_run_riganything_returns_latest_npz_by_mtime(tmp_path):
    out_dir = tmp_path / "out"
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("")

    mock_proc = MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(ra, "subprocess") as mock_subprocess, \
         patch.object(ra, "gpu_env", return_value={}):

        def run_side_effect(cmd, **kwargs):
            os.makedirs(out_dir, exist_ok=True)
            old = out_dir / "old.npz"
            new = out_dir / "new.npz"
            old.touch()
            new.touch()
            now = time.time()
            os.utime(old, (now - 100, now - 100))
            os.utime(new, (now, now))
            return mock_proc

        mock_subprocess.run.side_effect = run_side_effect
        result = ra.run_riganything(str(mesh), str(out_dir))

    assert result == str(out_dir / "new.npz")


def test_run_riganything_command_construction(tmp_path):
    out_dir = tmp_path / "out"
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("")

    mock_proc = MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(ra, "subprocess") as mock_subprocess, \
         patch.object(ra, "gpu_env", return_value={}):

        def run_side_effect(cmd, **kwargs):
            os.makedirs(out_dir, exist_ok=True)
            (out_dir / "x.npz").touch()
            return mock_proc

        mock_subprocess.run.side_effect = run_side_effect
        ra.run_riganything(str(mesh), str(out_dir))

    cmd = mock_subprocess.run.call_args[0][0]
    assert cmd[0] == ra.COMFY_VENV_PY
    assert cmd[1] == "inference.py"
    assert "--config" in cmd
    assert ra.CONFIG in cmd
    assert "--load" in cmd
    assert ra.CKPT in cmd
    assert "--mesh_path" in cmd
    assert os.path.abspath(str(mesh)) in cmd
    assert "inference_out_dir" in cmd
    assert os.path.abspath(str(out_dir)) in cmd


def test_run_riganything_passes_correct_kwargs(tmp_path):
    out_dir = tmp_path / "out"
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("")

    mock_proc = MagicMock(returncode=0, stdout="", stderr="")
    fake_env = {"HIP_VISIBLE_DEVICES": "0"}

    with patch.object(ra, "subprocess") as mock_subprocess, \
         patch.object(ra, "gpu_env", return_value=fake_env):

        def run_side_effect(cmd, **kwargs):
            os.makedirs(out_dir, exist_ok=True)
            (out_dir / "x.npz").touch()
            return mock_proc

        mock_subprocess.run.side_effect = run_side_effect
        ra.run_riganything(str(mesh), str(out_dir))

    kwargs = mock_subprocess.run.call_args[1]
    assert kwargs["cwd"] == ra.RA_DIR
    assert kwargs["env"] == fake_env
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["timeout"] == 1800


def test_run_riganything_custom_timeout(tmp_path):
    out_dir = tmp_path / "out"
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("")

    mock_proc = MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(ra, "subprocess") as mock_subprocess, \
         patch.object(ra, "gpu_env", return_value={}):

        def run_side_effect(cmd, **kwargs):
            os.makedirs(out_dir, exist_ok=True)
            (out_dir / "x.npz").touch()
            return mock_proc

        mock_subprocess.run.side_effect = run_side_effect
        ra.run_riganything(str(mesh), str(out_dir), timeout=42)

    kwargs = mock_subprocess.run.call_args[1]
    assert kwargs["timeout"] == 42