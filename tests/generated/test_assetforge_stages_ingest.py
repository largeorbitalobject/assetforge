import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import assetforge.stages.ingest


def _make_ws(tmp_path, input_image):
    ws = MagicMock()
    ws.manifest = {"input_image": str(input_image)}

    def dir_fn(name):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    ws.dir = dir_fn
    return ws


def _make_cfg(asset_type):
    cfg = MagicMock()
    cfg.asset_type = asset_type
    return cfg


def test_missing_input_image_raises_file_not_found(tmp_path):
    ws = _make_ws(tmp_path, tmp_path / "does_not_exist.png")
    cfg = _make_cfg("creature")
    log = MagicMock()
    with pytest.raises(FileNotFoundError):
        assetforge.stages.ingest.run(ws, cfg, None, log)


def test_scene_asset_type_copies_file(tmp_path):
    src = tmp_path / "scene.jpg"
    src.write_bytes(b"scene bytes")
    ws = _make_ws(tmp_path, src)
    cfg = _make_cfg("scene")
    log = MagicMock()

    result = assetforge.stages.ingest.run(ws, cfg, None, log)

    expected_dst = tmp_path / "ingest" / "subject.jpg"
    assert result["image"] == str(expected_dst)
    assert expected_dst.exists()
    assert expected_dst.read_bytes() == b"scene bytes"


def test_scene_asset_type_preserves_png_suffix(tmp_path):
    src = tmp_path / "env.png"
    src.write_bytes(b"png data")
    ws = _make_ws(tmp_path, src)
    cfg = _make_cfg("scene")
    log = MagicMock()

    result = assetforge.stages.ingest.run(ws, cfg, None, log)

    assert result["image"] == str(tmp_path / "ingest" / "subject.png")


def test_non_scene_calls_subprocess_with_rembg(tmp_path):
    src = tmp_path / "obj.png"
    src.write_bytes(b"obj data")
    ws = _make_ws(tmp_path, src)
    cfg = _make_cfg("object")
    log = MagicMock()

    with patch("assetforge.stages.ingest.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = assetforge.stages.ingest.run(ws, cfg, None, log)

    assert mock_run.called
    call_args = mock_run.call_args[0][0]
    assert call_args[1] == "-c"
    assert "rembg" in call_args[2]
    assert call_args[3] == str(src)
    assert call_args[4] == str(tmp_path / "ingest" / "subject.png")
    assert result["image"] == str(tmp_path / "ingest" / "subject.png")
    log.assert_called_once_with("isolated subject via rembg")


def test_non_scene_rembg_failure_falls_back_to_copy(tmp_path):
    src = tmp_path / "creature.png"
    src.write_bytes(b"creature pixels")
    ws = _make_ws(tmp_path, src)
    cfg = _make_cfg("creature")
    log = MagicMock()

    with patch("assetforge.stages.ingest.subprocess.run") as mock_run:
        mock_run.side_effect = Exception("model download failed")
        result = assetforge.stages.ingest.run(ws, cfg, None, log)

    expected_dst = tmp_path / "ingest" / "subject.png"
    assert result["image"] == str(expected_dst)
    assert expected_dst.exists()
    assert expected_dst.read_bytes() == b"creature pixels"
    log.assert_called_once_with("rembg failed (model download failed); using image as-is")


def test_non_scene_rembg_failure_preserves_original_suffix(tmp_path):
    src = tmp_path / "building.webp"
    src.write_bytes(b"webp data")
    ws = _make_ws(tmp_path, src)
    cfg = _make_cfg("building")
    log = MagicMock()

    with patch("assetforge.stages.ingest.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 300)
        result = assetforge.stages.ingest.run(ws, cfg, None, log)

    expected_dst = tmp_path / "ingest" / "subject.webp"
    assert result["image"] == str(expected_dst)
    assert expected_dst.read_bytes() == b"webp data"


def test_uppercase_suffix_is_lowered_on_fallback(tmp_path):
    src = tmp_path / "MECH.PNG"
    src.write_bytes(b"mechanical")
    ws = _make_ws(tmp_path, src)
    cfg = _make_cfg("mechanical")
    log = MagicMock()

    with patch("assetforge.stages.ingest.subprocess.run") as mock_run:
        mock_run.side_effect = Exception("boom")
        result = assetforge.stages.ingest.run(ws, cfg, None, log)

    assert result["image"] == str(tmp_path / "ingest" / "subject.png")
    assert Path(result["image"]).exists()


def test_rembg_success_for_mechanical_type(tmp_path):
    src = tmp_path / "gear.png"
    src.write_bytes(b"gear img")
    ws = _make_ws(tmp_path, src)
    cfg = _make_cfg("mechanical")
    log = MagicMock()

    with patch("assetforge.stages.ingest.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = assetforge.stages.ingest.run(ws, cfg, None, log)

    assert result["image"] == str(tmp_path / "ingest" / "subject.png")
    log.assert_called_once_with("isolated subject via rembg")


def test_subprocess_timeout_kwarg_is_300(tmp_path):
    src = tmp_path / "obj.png"
    src.write_bytes(b"x")
    ws = _make_ws(tmp_path, src)
    cfg = _make_cfg("object")
    log = MagicMock()

    with patch("assetforge.stages.ingest.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        assetforge.stages.ingest.run(ws, cfg, None, log)

    kwargs = mock_run.call_args[1]
    assert kwargs["timeout"] == 300
    assert kwargs["check"] is True
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True