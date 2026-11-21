"""Tests for assetforge.scene.retopo — pure logic only (no GPU, no network, no subprocess)."""
import os
import pytest
import assetforge.scene.retopo as retopo_module


class TestAvailable:
    def test_returns_false_when_dir_does_not_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(retopo_module, "MESHFLOW_DIR", str(tmp_path / "nonexistent"))
        assert retopo_module.available() is False

    def test_returns_false_when_inference_dit_missing(self, tmp_path, monkeypatch):
        # Directory exists but no inference_dit.py inside
        monkeypatch.setattr(retopo_module, "MESHFLOW_DIR", str(tmp_path))
        assert retopo_module.available() is False

    def test_returns_true_when_inference_dit_exists(self, tmp_path, monkeypatch):
        (tmp_path / "inference_dit.py").write_text("# placeholder")
        monkeypatch.setattr(retopo_module, "MESHFLOW_DIR", str(tmp_path))
        assert retopo_module.available() is True

    def test_returns_false_when_messflow_dir_is_empty_string(self, monkeypatch):
        monkeypatch.setattr(retopo_module, "MESHFLOW_DIR", "")
        assert retopo_module.available() is False


class TestRetopo:
    def test_raises_runtime_error_when_not_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr(retopo_module, "MESHFLOW_DIR", str(tmp_path / "nonexistent"))
        with pytest.raises(RuntimeError, match="MeshFlow not set up"):
            retopo_module.retopo(str(tmp_path / "input.glb"), str(tmp_path / "out"))

    def test_error_message_references_docs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(retopo_module, "MESHFLOW_DIR", str(tmp_path / "nonexistent"))
        with pytest.raises(RuntimeError, match="RETOPO-MESHFLOW"):
            retopo_module.retopo(str(tmp_path / "input.glb"), str(tmp_path / "out"))

    def test_error_message_mentions_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setattr(retopo_module, "MESHFLOW_DIR", str(tmp_path / "nonexistent"))
        with pytest.raises(RuntimeError, match="ASSETFORGE_MESHFLOW"):
            retopo_module.retopo(str(tmp_path / "input.glb"), str(tmp_path / "out"))


class TestMESHFLOW_DIR:
    def test_is_nonempty_string(self):
        assert isinstance(retopo_module.MESHFLOW_DIR, str)
        assert len(retopo_module.MESHFLOW_DIR) > 0

    def test_respects_env_var(self, monkeypatch):
        """Verify that MESHFLOW_DIR was originally derived from the env var or default.
        We re-derive the expected value using the same logic as the module."""
        env_val = os.environ.get("ASSETFORGE_MESHFLOW", os.path.expanduser("~/meshflow"))
        # The module captured this at import time, so it should match unless the
        # environment changed between import and now (unlikely in a test run).
        assert retopo_module.MESHFLOW_DIR == env_val