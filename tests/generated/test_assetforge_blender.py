import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import assetforge.blender
from assetforge.blender import run, available, SCRIPTS


def test_available_returns_false_when_blender_missing():
    with patch("assetforge.blender.BLENDER", "/nonexistent/blender/path"):
        assert available() is False


def test_available_returns_true_when_blender_exists():
    with patch("assetforge.blender.BLENDER", "/bin/ls"):
        assert available() is True


def test_run_raises_file_not_found_for_missing_relative_script(tmp_path):
    with patch("assetforge.blender.SCRIPTS", tmp_path):
        with pytest.raises(FileNotFoundError):
            run("does_not_exist.py", [])


def test_run_raises_file_not_found_for_missing_absolute_script():
    with pytest.raises(FileNotFoundError):
        run("/nonexistent/absolute/path/script.py", [])


def test_run_builds_correct_command(tmp_path):
    script = tmp_path / "test_script.py"
    script.touch()
    
    mock_completed = MagicMock(spec=subprocess.CompletedProcess)
    mock_completed.returncode = 0
    
    with patch("assetforge.blender.SCRIPTS", tmp_path), \
         patch("assetforge.blender.subprocess.run", return_value=mock_completed) as mock_run, \
         patch("assetforge.blender.gpu_env", return_value={"PATH": "/usr/bin"}):
        result = run("test_script.py", ["arg1", 42, "arg3"], timeout=30)
    
    assert result is mock_completed
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    cmd = call_kwargs[0][0]  # first positional arg
    assert cmd[0] == assetforge.blender.BLENDER
    assert cmd[1] == "-b"
    assert cmd[2] == "-noaudio"
    assert cmd[3] == "-P"
    assert cmd[4] == str(script)
    assert cmd[5] == "--"
    assert cmd[6:] == ["arg1", "42", "arg3"]
    assert call_kwargs[1]["env"] == {"PATH": "/usr/bin"}
    assert call_kwargs[1]["timeout"] == 30
    assert call_kwargs[1]["check"] is True
    assert call_kwargs[1]["capture_output"] is True
    assert call_kwargs[1]["text"] is True


def test_scripts_path_points_to_blender_scripts():
    assert SCRIPTS == Path(assetforge.blender.__file__).parent / "blender_scripts"