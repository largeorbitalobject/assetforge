import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import assetforge.scene.equipment as eq


class TestDefaultEquipment:
    def test_default_equipment_contains_expected_items(self):
        assert "sword" in eq.DEFAULT_EQUIPMENT
        assert "shield" in eq.DEFAULT_EQUIPMENT
        assert "armor" in eq.DEFAULT_EQUIPMENT
        assert "helmet" in eq.DEFAULT_EQUIPMENT
        assert "cape" in eq.DEFAULT_EQUIPMENT
        assert "weapon" in eq.DEFAULT_EQUIPMENT
        assert "staff" in eq.DEFAULT_EQUIPMENT

    def test_default_equipment_is_a_list(self):
        assert isinstance(eq.DEFAULT_EQUIPMENT, list)

    def test_default_equipment_has_seven_items(self):
        assert len(eq.DEFAULT_EQUIPMENT) == 7


class TestSeparate:
    @patch("assetforge.scene.equipment.P.split_parts_blender")
    @patch("assetforge.scene.equipment.P.segment_parts")
    def test_separate_uses_default_concepts_when_none(self, mock_seg, mock_split, tmp_path):
        mock_seg.return_value = {"sword": [0, 1], "body": [2]}
        mock_split.return_value = {"sword": "sword.glb", "body": "body.glb"}

        result = eq.separate("mesh.glb", str(tmp_path / "work"))

        mock_seg.assert_called_once_with(
            "mesh.glb", eq.DEFAULT_EQUIPMENT, str(tmp_path / "work" / "seg")
        )
        assert result == {"sword": "sword.glb", "body": "body.glb"}

    @patch("assetforge.scene.equipment.P.split_parts_blender")
    @patch("assetforge.scene.equipment.P.segment_parts")
    def test_separate_passes_custom_concepts(self, mock_seg, mock_split, tmp_path):
        mock_seg.return_value = {"sword": [0]}
        mock_split.return_value = {"sword": "sword.glb"}

        custom = ["sword", "shield"]
        eq.separate("mesh.glb", str(tmp_path / "work"), concepts=custom)

        mock_seg.assert_called_once_with(
            "mesh.glb", custom, str(tmp_path / "work" / "seg")
        )

    @patch("assetforge.scene.equipment.P.split_parts_blender")
    @patch("assetforge.scene.equipment.P.segment_parts")
    def test_separate_creates_work_dir(self, mock_seg, mock_split, tmp_path):
        mock_seg.return_value = {}
        mock_split.return_value = {}

        work = tmp_path / "new" / "work"
        eq.separate("mesh.glb", str(work))

        assert work.exists()

    @patch("assetforge.scene.equipment.P.split_parts_blender")
    @patch("assetforge.scene.equipment.P.segment_parts")
    def test_separate_passes_min_comp_frac(self, mock_seg, mock_split, tmp_path):
        mock_seg.return_value = {}
        mock_split.return_value = {}

        eq.separate("mesh.glb", str(tmp_path / "work"))

        mock_split.assert_called_once_with(
            "mesh.glb", {}, str(tmp_path / "work" / "parts"), min_comp_frac=0.10
        )

    @patch("assetforge.scene.equipment.P.split_parts_blender")
    @patch("assetforge.scene.equipment.P.segment_parts")
    def test_separate_returns_split_result(self, mock_seg, mock_split, tmp_path):
        expected = {"body": "body.glb", "sword": "sword.glb"}
        mock_seg.return_value = {"sword": [0], "body": [1]}
        mock_split.return_value = expected

        result = eq.separate("m.glb", str(tmp_path / "w"))
        assert result is expected


class TestSocketToWrist:
    def _make_npz(self, path, joints):
        """Create an npz file with a 'joints' key."""
        np.savez(str(path), joints=np.array(joints, dtype=np.float32))
        return str(path)

    def test_finds_wrist_by_side(self, tmp_path):
        npz_path = self._make_npz(
            tmp_path / "j.npz",
            joints=[[10, 20, 30], [100, 200, 300], [5, 5, 5]],
        )
        roles = {
            "arms": [
                {"side": "L", "wrist": 0},
                {"side": "R", "wrist": 1},
            ]
        }
        result = eq.socket_to_wrist("part.glb", npz_path, roles, "R", "out.glb")

        assert result["wrist_index"] == 1
        # J[1] = [100, 200, 300] * [1, -1, -1] = [100, -200, -300]
        assert result["wrist_pos"] == [100.0, -200.0, -300.0]
        assert result["part"] == "part.glb"

    def test_finds_wrist_left_side(self, tmp_path):
        npz_path = self._make_npz(
            tmp_path / "j.npz",
            joints=[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
        )
        roles = {
            "arms": [
                {"side": "L", "wrist": 0},
                {"side": "R", "wrist": 1},
            ]
        }
        result = eq.socket_to_wrist("p.glb", npz_path, roles, "L", "out.glb")

        assert result["wrist_index"] == 0
        # J[0] = [1, 2, 3] * [1, -1, -1] = [1, -2, -3]
        assert result["wrist_pos"] == [1.0, -2.0, -3.0]

    def test_fallback_to_first_arm_when_side_not_found(self, tmp_path):
        npz_path = self._make_npz(
            tmp_path / "j.npz",
            joints=[[7.0, 8.0, 9.0], [100.0, 200.0, 300.0]],
        )
        roles = {
            "arms": [
                {"side": "L", "wrist": 1},
            ]
        }
        # Request side "R" which doesn't exist; should fall back to arms[0]
        result = eq.socket_to_wrist("p.glb", npz_path, roles, "R", "out.glb")

        assert result["wrist_index"] == 1
        # J[1] = [100, 200, 300] * [1, -1, -1] = [100, -200, -300]
        assert result["wrist_pos"] == [100.0, -200.0, -300.0]

    def test_empty_arms_gives_origin(self, tmp_path):
        npz_path = self._make_npz(
            tmp_path / "j.npz",
            joints=[[1.0, 2.0, 3.0]],
        )
        roles = {"arms": []}
        result = eq.socket_to_wrist("p.glb", npz_path, roles, "L", "out.glb")

        assert result["wrist_index"] is None
        assert result["wrist_pos"] == [0, 0, 0]

    def test_missing_arms_key_gives_origin(self, tmp_path):
        npz_path = self._make_npz(
            tmp_path / "j.npz",
            joints=[[1.0, 2.0, 3.0]],
        )
        roles = {}
        result = eq.socket_to_wrist("p.glb", npz_path, roles, "L", "out.glb")

        assert result["wrist_index"] is None
        assert result["wrist_pos"] == [0, 0, 0]

    def test_coordinate_flip_applied(self, tmp_path):
        """Verify the [1, -1, -1] axis flip is applied to joint positions."""
        npz_path = self._make_npz(
            tmp_path / "j.npz",
            joints=[[3.0, -4.0, 5.0]],
        )
        roles = {"arms": [{"side": "L", "wrist": 0}]}
        result = eq.socket_to_wrist("p.glb", npz_path, roles, "L", "out.glb")

        # [3, -4, 5] * [1, -1, -1] = [3, 4, -5]
        assert result["wrist_pos"] == [3.0, 4.0, -5.0]

    def test_part_glb_passed_through(self, tmp_path):
        npz_path = self._make_npz(
            tmp_path / "j.npz",
            joints=[[0.0, 0.0, 0.0]],
        )
        roles = {"arms": [{"side": "L", "wrist": 0}]}
        result = eq.socket_to_wrist("/path/to/sword.glb", npz_path, roles, "L", "out.glb")

        assert result["part"] == "/path/to/sword.glb"

    def test_wrist_arm_with_none_wrist_falls_back(self, tmp_path):
        """If the matching arm has wrist=None, wrist_idx stays None → origin."""
        npz_path = self._make_npz(
            tmp_path / "j.npz",
            joints=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )
        roles = {
            "arms": [
                {"side": "L", "wrist": None},
                {"side": "R", "wrist": 1},
            ]
        }
        # side "L" matches but wrist is None
        result = eq.socket_to_wrist("p.glb", npz_path, roles, "L", "out.glb")

        assert result["wrist_index"] is None
        assert result["wrist_pos"] == [0, 0, 0]