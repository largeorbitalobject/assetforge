"""Tests for assetforge.articulation.graph"""
import json
import pytest
from assetforge.articulation.graph import (
    JOINT_TYPES,
    Joint,
    Part,
    ArticulationGraph,
)


# ---------- Joint ----------

def test_joint_valid_types():
    for t in JOINT_TYPES:
        j = Joint(id="j1", parent="world", child="p1", type=t,
                  origin=[0, 0, 0], axis=[0, 0, 1])
        assert j.type == t


def test_joint_invalid_type_raises():
    with pytest.raises(ValueError, match="bad joint type"):
        Joint(id="j1", parent="world", child="p1", type="hinge",
              origin=[0, 0, 0], axis=[0, 0, 1])


def test_joint_defaults():
    j = Joint(id="j1", parent="world", child="p1", type="fixed",
              origin=[0, 0, 0], axis=[0, 0, 1])
    assert j.limits is None
    assert j.rest == 0.0
    assert j.motion is None


def test_joint_with_limits_and_motion():
    j = Joint(id="j1", parent="world", child="p1", type="revolute",
              origin=[1, 2, 3], axis=[0, 0, 1],
              limits={"min": -1.0, "max": 1.0}, rest=0.5,
              motion={"preset": "elevate"})
    assert j.limits == {"min": -1.0, "max": 1.0}
    assert j.rest == 0.5
    assert j.motion == {"preset": "elevate"}


# ---------- Part ----------

def test_part_defaults():
    p = Part(id="p1")
    assert p.object is None
    assert p.vgroup is None
    assert p.face_ids is None
    assert p.centroid is None
    assert p.bbox is None
    assert p.principal_axes is None


def test_part_full():
    p = Part(id="p1", object="mesh_a", vgroup="grp1",
             face_ids=[0, 1, 2], centroid=[1, 2, 3],
             bbox=[[0, 0, 0], [1, 1, 1]],
             principal_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert p.object == "mesh_a"
    assert p.face_ids == [0, 1, 2]


# ---------- ArticulationGraph: defaults ----------

def test_up_axis_default():
    g = ArticulationGraph(kind="mechanical", mesh="model.glb")
    assert g.up_axis == [0, 0, 1]
    assert g.parts == []
    assert g.joints == []
    assert g.skin is None
    assert g.roles is None


# ---------- to_dict ----------

def test_to_dict_without_skin_roles():
    g = ArticulationGraph(kind="mechanical", mesh="model.glb",
                          parts=[Part(id="p1")],
                          joints=[Joint(id="j1", parent="world", child="p1",
                                        type="fixed", origin=[0, 0, 0], axis=[0, 0, 1])])
    d = g.to_dict()
    assert d["kind"] == "mechanical"
    assert d["mesh"] == "model.glb"
    assert "skin" not in d
    assert "roles" not in d
    assert isinstance(d["parts"], list)
    assert d["parts"][0]["id"] == "p1"
    assert d["joints"][0]["type"] == "fixed"


def test_to_dict_with_skin_and_roles():
    g = ArticulationGraph(kind="organic", mesh="body.glb",
                          skin={"joints_xyz": [[0, 0, 0]], "parents": [0], "weights_npz": "w.npz"},
                          roles={"pelvis": "p0", "spine": ["p1", "p2"]})
    d = g.to_dict()
    assert d["skin"] == {"joints_xyz": [[0, 0, 0]], "parents": [0], "weights_npz": "w.npz"}
    assert d["roles"] == {"pelvis": "p0", "spine": ["p1", "p2"]}


# ---------- save / load roundtrip ----------

def test_save_and_load_roundtrip(tmp_path):
    g = ArticulationGraph(
        kind="mechanical", mesh="robot.glb",
        parts=[Part(id="base"), Part(id="arm", object="arm_mesh")],
        joints=[Joint(id="j1", parent="world", child="base", type="fixed",
                      origin=[0, 0, 0], axis=[0, 0, 1]),
                Joint(id="j2", parent="base", child="arm", type="revolute",
                      origin=[0, 0, 1], axis=[0, 1, 0],
                      limits={"min": -3.14, "max": 3.14})],
    )
    path = str(tmp_path / "graph.json")
    g.save(path)

    loaded = ArticulationGraph.load(path)
    assert loaded.kind == "mechanical"
    assert loaded.mesh == "robot.glb"
    assert len(loaded.parts) == 2
    assert loaded.parts[0].id == "base"
    assert loaded.parts[1].object == "arm_mesh"
    assert len(loaded.joints) == 2
    assert loaded.joints[1].type == "revolute"
    assert loaded.joints[1].limits == {"min": -3.14, "max": 3.14}


def test_save_returns_path(tmp_path):
    g = ArticulationGraph(kind="mechanical", mesh="x.glb")
    path = str(tmp_path / "out.json")
    result = g.save(path)
    assert result == path


def test_load_with_skin_and_roles(tmp_path):
    g = ArticulationGraph(kind="organic", mesh="body.glb",
                          skin={"joints_xyz": [[0, 0, 1]]},
                          roles={"pelvis": "p0"})
    path = str(tmp_path / "organic.json")
    g.save(path)

    loaded = ArticulationGraph.load(path)
    assert loaded.skin == {"joints_xyz": [[0, 0, 1]]}
    assert loaded.roles == {"pelvis": "p0"}


def test_load_missing_mesh_defaults_empty(tmp_path):
    d = {"kind": "mechanical", "parts": [], "joints": []}
    path = tmp_path / "minimal.json"
    path.write_text(json.dumps(d))
    loaded = ArticulationGraph.load(str(path))
    assert loaded.mesh == ""
    assert loaded.kind == "mechanical"


def test_load_missing_up_axis_defaults(tmp_path):
    d = {"kind": "mechanical", "mesh": "m.glb", "parts": [], "joints": []}
    path = tmp_path / "no_up.json"
    path.write_text(json.dumps(d))
    loaded = ArticulationGraph.load(str(path))
    assert loaded.up_axis == [0, 0, 1]


# ---------- validate ----------

def test_validate_valid_mechanical_graph():
    g = ArticulationGraph(
        kind="mechanical", mesh="robot.glb",
        parts=[Part(id="base"), Part(id="arm")],
        joints=[Joint(id="j1", parent="world", child="base", type="fixed",
                      origin=[0, 0, 0], axis=[0, 0, 1]),
                Joint(id="j2", parent="base", child="arm", type="revolute",
                      origin=[0, 0, 1], axis=[0, 1, 0])],
    )
    assert g.validate() == []


def test_validate_bad_kind():
    g = ArticulationGraph(kind="hybrid", mesh="x.glb")
    errs = g.validate()
    assert any("bad kind" in e for e in errs)


def test_validate_parent_not_a_part():
    g = ArticulationGraph(
        kind="mechanical", mesh="x.glb",
        parts=[Part(id="p1")],
        joints=[Joint(id="j1", parent="nonexistent", child="p1", type="fixed",
                      origin=[0, 0, 0], axis=[0, 0, 1])],
    )
    errs = g.validate()
    assert any("parent" in e and "nonexistent" in e for e in errs)


def test_validate_child_not_a_part():
    g = ArticulationGraph(
        kind="mechanical", mesh="x.glb",
        parts=[Part(id="p1")],
        joints=[Joint(id="j1", parent="world", child="ghost", type="fixed",
                      origin=[0, 0, 0], axis=[0, 0, 1])],
    )
    errs = g.validate()
    assert any("child" in e and "ghost" in e for e in errs)


def test_validate_world_parent_is_valid():
    g = ArticulationGraph(
        kind="mechanical", mesh="x.glb",
        parts=[Part(id="p1")],
        joints=[Joint(id="j1", parent="world", child="p1", type="fixed",
                      origin=[0, 0, 0], axis=[0, 0, 1])],
    )
    errs = g.validate()
    assert errs == []


def test_validate_organic_missing_skin():
    g = ArticulationGraph(kind="organic", mesh="body.glb",
                          parts=[Part(id="pelvis")])
    errs = g.validate()
    assert any("skin" in e for e in errs)


def test_validate_organic_with_skin_ok():
    g = ArticulationGraph(kind="organic", mesh="body.glb",
                          parts=[Part(id="pelvis")],
                          skin={"joints_xyz": [[0, 0, 0]], "parents": [0], "weights_npz": "w.npz"})
    errs = g.validate()
    assert errs == []


def test_validate_mechanical_missing_skin_ok():
    """Mechanical graphs do not require a skin block."""
    g = ArticulationGraph(kind="mechanical", mesh="robot.glb",
                          parts=[Part(id="base")])
    errs = g.validate()
    assert errs == []


# ---------- validate with dict parts/joints (mixed) ----------

def test_validate_with_dict_parts_and_joints():
    g = ArticulationGraph(
        kind="mechanical", mesh="x.glb",
        parts=[{"id": "p1"}, {"id": "p2"}],
        joints=[{"id": "j1", "parent": "p1", "child": "p2", "type": "fixed",
                 "origin": [0, 0, 0], "axis": [0, 0, 1]}],
    )
    assert g.validate() == []


def test_validate_with_dict_parts_bad_child():
    g = ArticulationGraph(
        kind="mechanical", mesh="x.glb",
        parts=[{"id": "p1"}],
        joints=[{"id": "j1", "parent": "p1", "child": "p_missing", "type": "fixed",
                 "origin": [0, 0, 0], "axis": [0, 0, 1]}],
    )
    errs = g.validate()
    assert any("p_missing" in e for e in errs)


# ---------- JOINT_TYPES constant ----------

def test_joint_types_content():
    assert JOINT_TYPES == ("fixed", "revolute", "prismatic", "ball")