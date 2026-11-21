"""Tests for assetforge.stages.rig — pure logic only (no GPU, Blender, or network)."""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import assetforge.stages.rig as rigmod


# ---------------------------------------------------------------------------
# _input_mesh
# ---------------------------------------------------------------------------

class TestInputMesh:
    def test_prefers_texture_over_cleanup_and_mesh(self):
        upstream = {
            "texture": {"mesh": "/path/to/textured.glb"},
            "cleanup": {"mesh": "/path/to/cleaned.glb"},
            "mesh": {"mesh": "/path/to/raw.glb"},
        }
        ws = SimpleNamespace(manifest={"input_image": "/path/to/input.png"})
        assert rigmod._input_mesh(upstream, ws) == "/path/to/textured.glb"

    def test_prefers_cleanup_over_mesh(self):
        upstream = {
            "cleanup": {"mesh": "/path/to/cleaned.glb"},
            "mesh": {"mesh": "/path/to/raw.glb"},
        }
        ws = SimpleNamespace(manifest={"input_image": "/path/to/input.png"})
        assert rigmod._input_mesh(upstream, ws) == "/path/to/cleaned.glb"

    def test_falls_back_to_mesh_stage(self):
        upstream = {"mesh": {"mesh": "/path/to/raw.glb"}}
        ws = SimpleNamespace(manifest={"input_image": "/path/to/input.png"})
        assert rigmod._input_mesh(upstream, ws) == "/path/to/raw.glb"

    def test_falls_back_to_input_image_when_no_stage_has_mesh(self):
        upstream = {"texture": {"something_else": "x"}, "mesh": {}}
        ws = SimpleNamespace(manifest={"input_image": "/path/to/input.png"})
        assert rigmod._input_mesh(upstream, ws) == "/path/to/input.png"

    def test_empty_upstream_falls_back_to_input_image(self):
        ws = SimpleNamespace(manifest={"input_image": "img.png"})
        assert rigmod._input_mesh({}, ws) == "img.png"

    def test_ignores_falsy_mesh_values(self):
        upstream = {
            "texture": {"mesh": None},
            "cleanup": {"mesh": ""},
            "mesh": {"mesh": "/path/to/raw.glb"},
        }
        ws = SimpleNamespace(manifest={"input_image": "/path/to/input.png"})
        assert rigmod._input_mesh(upstream, ws) == "/path/to/raw.glb"

    def test_all_falsy_mesh_falls_back_to_input_image(self):
        upstream = {
            "texture": {"mesh": None},
            "cleanup": {"mesh": ""},
            "mesh": {"mesh": None},
        }
        ws = SimpleNamespace(manifest={"input_image": "/path/to/input.png"})
        assert rigmod._input_mesh(upstream, ws) == "/path/to/input.png"


# ---------------------------------------------------------------------------
# run — dispatch logic
# ---------------------------------------------------------------------------

class TestRunDispatch:
    def test_unknown_kind_raises_not_implemented(self):
        ws = SimpleNamespace(manifest={"input_image": "img.png"})
        cfg = SimpleNamespace(kind="aliens", asset_type="alien_creature")
        upstream = {"mesh": {"mesh": "m.glb"}}
        log = MagicMock()
        with pytest.raises(NotImplementedError, match="alien_creature"):
            rigmod.run(ws, cfg, upstream, log)

    def test_organic_kind_calls_organic_branch(self):
        ws = SimpleNamespace(manifest={"input_image": "img.png"})
        cfg = SimpleNamespace(kind="organic", asset_type="dragon")
        upstream = {"mesh": {"mesh": "dragon_raw.glb"}}
        log = MagicMock()
        expected = {"mesh": "rigged.glb", "graph": "graph.json", "npz": "skel.npz"}

        with patch.object(rigmod, "_organic", return_value=expected) as mock_org:
            result = rigmod.run(ws, cfg, upstream, log)

        assert result == expected
        mock_org.assert_called_once_with(ws, cfg, "dragon_raw.glb", log)

    def test_mechanical_kind_calls_mechanical_rig(self):
        ws = SimpleNamespace(manifest={"input_image": "img.png"})
        cfg = SimpleNamespace(kind="mechanical", asset_type="robot_arm")
        upstream = {"texture": {"mesh": "textured_arm.glb"}}
        log = MagicMock()
        expected = {"mesh": "rigged_arm.glb", "graph": "arm_graph.json"}

        fake_mech_mod = MagicMock()
        fake_mech_mod.mechanical_rig = MagicMock(return_value=expected)

        with patch.dict(sys.modules, {"assetforge.stages.mech_rig": fake_mech_mod}):
            result = rigmod.run(ws, cfg, upstream, log)

        assert result == expected
        fake_mech_mod.mechanical_rig.assert_called_once_with(
            ws, cfg, "textured_arm.glb", log
        )

    def test_run_passes_correct_mesh_from_upstream_to_organic(self):
        """Verify the mesh resolved by _input_mesh is what gets passed to _organic."""
        ws = SimpleNamespace(manifest={"input_image": "img.png"})
        cfg = SimpleNamespace(kind="organic", asset_type="cat")
        upstream = {
            "cleanup": {"mesh": "cleaned_cat.glb"},
            "mesh": {"mesh": "raw_cat.glb"},
        }
        log = MagicMock()
        expected = {"mesh": "out.glb", "graph": "g.json", "npz": "s.npz"}

        with patch.object(rigmod, "_organic", return_value=expected) as mock_org:
            rigmod.run(ws, cfg, upstream, log)

        # Should pick "cleanup" over "mesh"
        mock_org.assert_called_once_with(ws, cfg, "cleaned_cat.glb", log)