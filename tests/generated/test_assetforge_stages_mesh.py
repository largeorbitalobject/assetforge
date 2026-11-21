import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from assetforge.stages.mesh import (
    _hunyuan3d_graph,
    _hunyuanworld_graph,
    _newest_glb_since,
    run,
    GRAPH_DIR,
    COMFY_INPUT,
    COMFY_OUTPUT,
)


class TestHunyuan3dGraph:
    """Tests for _hunyuan3d_graph function."""

    @pytest.fixture
    def mock_graph_file(self, tmp_path):
        """Create a mock ComfyUI graph JSON file."""
        graph_dir = tmp_path / "comfy_graphs"
        graph_dir.mkdir()
        graph_file = graph_dir / "hunyuan3d-2.1-image-to-glb.json"
        mock_graph = {
            "prompt": {
                "1": {"class_type": "LoadImage", "inputs": {"image": "test.png"}},
                "2": {"class_type": "SomeNode", "inputs": {"image": "test.png"}},
                "7": {"class_type": "KSampler", "inputs": {"steps": 20}},
                "8": {"class_type": "Octree", "inputs": {"octree_resolution": 256}},
                "9": {"class_type": "Voxel", "inputs": {"threshold": 0.5}},
                "10": {"class_type": "SaveGLB", "inputs": {"filename_prefix": "test"}},
            }
        }
        graph_file.write_text(json.dumps(mock_graph))
        return graph_dir

    def test_hunyuan3d_graph_loads_and_modifies(self, mock_graph_file):
        """Test that _hunyuan3d_graph loads the JSON and applies config values."""
        cfg = MagicMock()
        cfg.octree_resolution = 512
        cfg.voxel_threshold = 0.3
        cfg.ksampler_steps = 30

        with patch("assetforge.stages.mesh.GRAPH_DIR", mock_graph_file):
            result = _hunyuan3d_graph("test_image.png", cfg)

        assert result["2"]["inputs"]["image"] == "test_image.png"
        assert result["8"]["inputs"]["octree_resolution"] == 512
        assert result["9"]["inputs"]["threshold"] == 0.3
        assert result["7"]["inputs"]["steps"] == 30
        assert result["10"]["inputs"]["filename_prefix"] == "assetforge/mesh"

    def test_hunyuan3d_graph_preserves_other_nodes(self, mock_graph_file):
        """Test that nodes not modified by config are preserved."""
        cfg = MagicMock()
        cfg.octree_resolution = 256
        cfg.voxel_threshold = 0.5
        cfg.ksampler_steps = 20

        with patch("assetforge.stages.mesh.GRAPH_DIR", mock_graph_file):
            result = _hunyuan3d_graph("img.png", cfg)

        # Node 1 should be unchanged
        assert result["1"]["class_type"] == "LoadImage"
        assert result["1"]["inputs"]["image"] == "test.png"

    def test_hunyuan3d_graph_different_image_name(self, mock_graph_file):
        """Test that different image names are correctly set."""
        cfg = MagicMock()
        cfg.octree_resolution = 128
        cfg.voxel_threshold = 0.1
        cfg.ksampler_steps = 10

        with patch("assetforge.stages.mesh.GRAPH_DIR", mock_graph_file):
            result = _hunyuan3d_graph("different_image.jpg", cfg)

        assert result["2"]["inputs"]["image"] == "different_image.jpg"


class TestHunyuanWorldGraph:
    """Tests for _hunyuanworld_graph function."""

    def test_hunyuanworld_graph_structure(self):
        """Test that _hunyuanworld_graph returns correct structure."""
        result = _hunyuanworld_graph("scene.png")

        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_hunyuanworld_graph_load_image_node(self):
        """Test LoadImage node configuration."""
        result = _hunyuanworld_graph("test_scene.png")

        assert result["1"]["class_type"] == "LoadImage"
        assert result["1"]["inputs"]["image"] == "test_scene.png"

    def test_hunyuanworld_graph_model_loader(self):
        """Test HunyuanWorldMirrorModelLoader node configuration."""
        result = _hunyuanworld_graph("scene.png")

        assert result["2"]["class_type"] == "HunyuanWorldMirrorModelLoader"
        assert result["2"]["inputs"]["model_name"] == "tencent/HunyuanWorld-Mirror"
        assert result["2"]["inputs"]["precision"] == "auto"
        assert result["2"]["inputs"]["offload_to_cpu"] is False

    def test_hunyuanworld_graph_image_to_3d(self):
        """Test HunyuanWorldMirrorImageTo3D node configuration."""
        result = _hunyuanworld_graph("scene.png")

        node = result["3"]
        assert node["class_type"] == "HunyuanWorldMirrorImageTo3D"
        assert node["inputs"]["model"] == ["2", 0]
        assert node["inputs"]["images"] == ["1", 0]
        assert node["inputs"]["preprocessing_mode"] == "crop"
        assert node["inputs"]["save_outputs"] is True
        assert node["inputs"]["filename_prefix"] == "assetforge/scene"
        assert node["inputs"]["preprocessing_resolution"] == 518
        assert node["inputs"]["confidence_filter"] == 5.0
        assert node["inputs"]["edge_normal_threshold"] == 8.0
        assert node["inputs"]["edge_depth_threshold"] == 0.05
        assert node["inputs"]["boundary_erosion"] == 2

    def test_hunyuanworld_graph_different_image(self):
        """Test that different image names are correctly set."""
        result = _hunyuanworld_graph("another_scene.jpg")

        assert result["1"]["inputs"]["image"] == "another_scene.jpg"


class TestNewestGlbSince:
    """Tests for _newest_glb_since function."""

    def test_newest_glb_since_no_files(self, tmp_path):
        """Test that _newest_glb_since returns None when no GLB files exist."""
        with patch("assetforge.stages.mesh.COMFY_OUTPUT", tmp_path):
            result = _newest_glb_since(time.time())
            assert result is None

    def test_newest_glb_since_finds_recent_file(self, tmp_path):
        """Test that _newest_glb_since finds a recently created GLB file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        glb_file = output_dir / "test_mesh.glb"
        glb_file.write_bytes(b"fake glb data")

        t0 = time.time() - 10  # 10 seconds ago
        with patch("assetforge.stages.mesh.COMFY_OUTPUT", output_dir):
            result = _newest_glb_since(t0)
            assert result == glb_file

    def test_newest_glb_since_ignores_old_files(self, tmp_path):
        """Test that _newest_glb_since ignores files older than t0."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        glb_file = output_dir / "old_mesh.glb"
        glb_file.write_bytes(b"fake glb data")

        # Set mtime to be older than t0
        old_time = time.time() - 100
        import os
        os.utime(glb_file, (old_time, old_time))

        t0 = time.time() - 10  # Only 10 seconds ago
        with patch("assetforge.stages.mesh.COMFY_OUTPUT", output_dir):
            result = _newest_glb_since(t0)
            assert result is None

    def test_newest_glb_since_returns_newest(self, tmp_path):
        """Test that _newest_glb_since returns the most recent GLB file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create two GLB files with different mtimes
        glb1 = output_dir / "mesh1.glb"
        glb2 = output_dir / "mesh2.glb"
        glb1.write_bytes(b"glb1")
        glb2.write_bytes(b"glb2")

        import os
        t1 = time.time() - 20
        t2 = time.time() - 5
        os.utime(glb1, (t1, t1))
        os.utime(glb2, (t2, t2))

        t0 = time.time() - 30
        with patch("assetforge.stages.mesh.COMFY_OUTPUT", output_dir):
            result = _newest_glb_since(t0)
            assert result == glb2

    def test_newest_glb_since_recursive_search(self, tmp_path):
        """Test that _newest_glb_since searches recursively."""
        output_dir = tmp_path / "output"
        sub_dir = output_dir / "subdir"
        sub_dir.mkdir(parents=True)

        glb_file = sub_dir / "nested_mesh.glb"
        glb_file.write_bytes(b"nested glb")

        t0 = time.time() - 10
        with patch("assetforge.stages.mesh.COMFY_OUTPUT", output_dir):
            result = _newest_glb_since(t0)
            assert result == glb_file

    def test_newest_glb_since_ignores_non_glb(self, tmp_path):
        """Test that _newest_glb_since ignores non-GLB files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a non-GLB file
        txt_file = output_dir / "notes.txt"
        txt_file.write_text("not a glb")

        t0 = time.time() - 10
        with patch("assetforge.stages.mesh.COMFY_OUTPUT", output_dir):
            result = _newest_glb_since(t0)
            assert result is None


class TestRun:
    """Tests for run function."""

    def test_run_hunyuan3d_backend_success(self, tmp_path):
        """Test run with hunyuan3d backend and successful GLB generation."""
        # Setup workspace
        ws = MagicMock()
        ws.manifest = {"input_image": "test.png"}
        ws.dir.return_value = tmp_path / "mesh_dir"
        ws.dir.return_value.mkdir(parents=True)

        # Setup config
        cfg = MagicMock()
        cfg.mesh_backend = "hunyuan3d"
        cfg.asset_type = "creature"
        cfg.octree_resolution = 256
        cfg.voxel_threshold = 0.5
        cfg.ksampler_steps = 20

        # Setup upstream
        upstream = {"ingest": {"image": str(tmp_path / "test.png")}}
        (tmp_path / "test.png").write_bytes(b"fake image")

        # Mock comfy.run_graph to return a GLB path
        glb_output = tmp_path / "output" / "mesh.glb"
        glb_output.parent.mkdir(parents=True)
        glb_output.write_bytes(b"fake glb")

        with patch("assetforge.stages.mesh.comfy.run_graph") as mock_run_graph:
            mock_run_graph.return_value = [glb_output]
            with patch("assetforge.stages.mesh.COMFY_INPUT", tmp_path / "comfy_input"):
                (tmp_path / "comfy_input").mkdir()
                result = run(ws, cfg, upstream, MagicMock())

        assert "mesh" in result
        assert result["mesh"].endswith("mesh.glb")
        assert Path(result["mesh"]).exists()

    def test_run_hunyuanworld_backend_success(self, tmp_path):
        """Test run with hunyuanworld backend and successful GLB generation."""
        ws = MagicMock()
        ws.manifest = {"input_image": "scene.png"}
        ws.dir.return_value = tmp_path / "mesh_dir"
        ws.dir.return_value.mkdir(parents=True)

        cfg = MagicMock()
        cfg.mesh_backend = "hunyuanworld"
        cfg.asset_type = "scene"

        upstream = {"ingest": {"image": str(tmp_path / "scene.png")}}
        (tmp_path / "scene.png").write_bytes(b"fake image")

        # Create a GLB file that will be found by _newest_glb_since
        output_dir = tmp_path / "comfy_output"
        output_dir.mkdir()
        glb_file = output_dir / "scene_mesh.glb"
        glb_file.write_bytes(b"fake glb")

        with patch("assetforge.stages.mesh.comfy.run_graph") as mock_run_graph:
            mock_run_graph.return_value = None
            with patch("assetforge.stages.mesh.COMFY_INPUT", tmp_path / "comfy_input"):
                (tmp_path / "comfy_input").mkdir()
                with patch("assetforge.stages.mesh.COMFY_OUTPUT", output_dir):
                    result = run(ws, cfg, upstream, MagicMock())

        assert "mesh" in result
        assert result["mesh"].endswith("mesh.glb")

    def test_run_scene_asset_type_uses_hunyuanworld(self, tmp_path):
        """Test that scene asset type triggers hunyuanworld backend."""
        ws = MagicMock()
        ws.manifest = {"input_image": "scene.png"}
        ws.dir.return_value = tmp_path / "mesh_dir"
        ws.dir.return_value.mkdir(parents=True)

        cfg = MagicMock()
        cfg.mesh_backend = "hunyuan3d"  # Even if backend is hunyuan3d
        cfg.asset_type = "scene"  # But asset type is scene

        upstream = {"ingest": {"image": str(tmp_path / "scene.png")}}
        (tmp_path / "scene.png").write_bytes(b"fake image")

        output_dir = tmp_path / "comfy_output"
        output_dir.mkdir()
        glb_file = output_dir / "scene.glb"
        glb_file.write_bytes(b"fake glb")

        with patch("assetforge.stages.mesh.comfy.run_graph") as mock_run_graph:
            mock_run_graph.return_value = None
            with patch("assetforge.stages.mesh.COMFY_INPUT", tmp_path / "comfy_input"):
                (tmp_path / "comfy_input").mkdir()
                with patch("assetforge.stages.mesh.COMFY_OUTPUT", output_dir):
                    result = run(ws, cfg, upstream, MagicMock())

        # Should have called hunyuanworld graph
        assert mock_run_graph.called
        assert "mesh" in result

    def test_run_hunyuan3d_no_glb_raises(self, tmp_path):
        """Test that run raises RuntimeError when hunyuan3d produces no GLB."""
        ws = MagicMock()
        ws.manifest = {"input_image": "test.png"}
        ws.dir.return_value = tmp_path / "mesh_dir"
        ws.dir.return_value.mkdir(parents=True)

        cfg = MagicMock()
        cfg.mesh_backend = "hunyuan3d"
        cfg.asset_type = "creature"
        cfg.octree_resolution = 256
        cfg.voxel_threshold = 0.5
        cfg.ksampler_steps = 20

        upstream = {"ingest": {"image": str(tmp_path / "test.png")}}
        (tmp_path / "test.png").write_bytes(b"fake image")

        with patch("assetforge.stages.mesh.comfy.run_graph") as mock_run_graph:
            mock_run_graph.return_value = []  # No GLB produced
            with patch("assetforge.stages.mesh.COMFY_INPUT", tmp_path / "comfy_input"):
                (tmp_path / "comfy_input").mkdir()
                with pytest.raises(RuntimeError, match="ComfyUI produced no GLB"):
                    run(ws, cfg, upstream, MagicMock())

    def test_run_hunyuanworld_no_glb_raises(self, tmp_path):
        """Test that run raises RuntimeError when hunyuanworld produces no GLB."""
        ws = MagicMock()
        ws.manifest = {"input_image": "scene.png"}
        ws.dir.return_value = tmp_path / "mesh_dir"
        ws.dir.return_value.mkdir(parents=True)

        cfg = MagicMock()
        cfg.mesh_backend = "hunyuanworld"
        cfg.asset_type = "scene"

        upstream = {"ingest": {"image": str(tmp_path / "scene.png")}}
        (tmp_path / "scene.png").write_bytes(b"fake image")

        with patch("assetforge.stages.mesh.comfy.run_graph") as mock_run_graph:
            mock_run_graph.return_value = None
            with patch("assetforge.stages.mesh.COMFY_INPUT", tmp_path / "comfy_input"):
                (tmp_path / "comfy_input").mkdir()
                with patch("assetforge.stages.mesh.COMFY_OUTPUT", tmp_path / "empty_output"):
                    (tmp_path / "empty_output").mkdir()
                    with pytest.raises(RuntimeError, match="HunyuanWorld produced no GLB"):
                        run(ws, cfg, upstream, MagicMock())

    def test_run_copies_image_to_comfy_input(self, tmp_path):
        """Test that run copies the input image to COMFY_INPUT."""
        ws = MagicMock()
        ws.manifest = {"input_image": "test.png"}
        ws.dir.return_value = tmp_path / "mesh_dir"
        ws.dir.return_value.mkdir(parents=True)

        cfg = MagicMock()