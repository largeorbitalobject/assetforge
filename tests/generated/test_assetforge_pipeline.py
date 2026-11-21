"""Tests for assetforge.pipeline — pure logic, no GPU/Blender/network."""
from unittest.mock import MagicMock, patch
import pytest

import assetforge.pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(stages):
    """Build a minimal config mock with the .stages attribute."""
    cfg = MagicMock()
    cfg.stages = list(stages)
    return cfg


def _make_ws(outputs_map=None, is_fresh_map=None, root="/tmp/fake_ws"):
    """Build a minimal workspace mock."""
    ws = MagicMock()
    ws.root = root
    _outputs = outputs_map or {}
    _fresh = is_fresh_map or {}
    ws.outputs.side_effect = lambda stage: _outputs.get(stage, {})
    ws.is_fresh.side_effect = lambda stage, fp: _fresh.get(stage, False)
    ws.manifest = {"stages": {}}
    return ws


def _run(stages, *, only=None, until=None, resume=True, outputs_map=None,
         is_fresh_map=None, registry_map=None):
    """Convenience wrapper that patches all pipeline dependencies and calls run()."""
    cfg = _make_cfg(stages)
    ws = _make_ws(outputs_map=outputs_map, is_fresh_map=is_fresh_map)
    registry = registry_map if registry_map is not None else {}

    with patch.object(assetforge.pipeline.cfgmod, "load", return_value=cfg), \
         patch.object(assetforge.pipeline.cfgmod, "as_dict", return_value={"stages": stages}), \
         patch.object(assetforge.pipeline.wsmod, "Workspace") as MockWS, \
         patch.object(assetforge.pipeline.wsmod, "fingerprint", return_value="fp123"), \
         patch.object(assetforge.pipeline, "REGISTRY", registry):
        MockWS.create.return_value = ws
        result = assetforge.pipeline.run(
            "test.png", "model",
            overrides=None, resume=resume, until=until, only=only,
        )
    return ws, result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_until_truncates_stages():
    """'until' should limit execution to stages up to and including the named stage."""
    stages = ["a", "b", "c"]
    call_order = []
    registry = {
        "a": MagicMock(side_effect=lambda *a: (call_order.append("a"), ["out_a"])[1]),
        "b": MagicMock(side_effect=lambda *a: (call_order.append("b"), ["out_b"])[1]),
        "c": MagicMock(side_effect=lambda *a: (call_order.append("c"), ["out_c"])[1]),
    }

    with patch.object(assetforge.pipeline.cfgmod, "load", return_value=_make_cfg(stages)), \
         patch.object(assetforge.pipeline.cfgmod, "as_dict", return_value={"stages": stages}), \
         patch.object(assetforge.pipeline.wsmod, "Workspace") as MockWS, \
         patch.object(assetforge.pipeline.wsmod, "fingerprint", return_value="fp"), \
         patch.object(assetforge.pipeline, "REGISTRY", registry):
        ws = _make_ws()
        MockWS.create.return_value = ws
        assetforge.pipeline.run("img.png", "model", until="b")

    assert call_order == ["a", "b"]
    assert "c" not in call_order


def test_until_invalid_stage_raises_valueerror():
    """Passing an unknown stage name to 'until' must raise ValueError."""
    stages = ["a", "b"]
    with patch.object(assetforge.pipeline.cfgmod, "load", return_value=_make_cfg(stages)), \
         patch.object(assetforge.pipeline.wsmod, "Workspace") as MockWS:
        ws = _make_ws()
        MockWS.create.return_value = ws
        with pytest.raises(ValueError, match="not in"):
            assetforge.pipeline.run("img.png", "model", until="nonexistent")


def test_only_runs_single_stage():
    """When 'only' is set, exactly that one stage should be executed."""
    stages = ["a", "b", "c"]
    call_order = []
    registry = {
        "a": MagicMock(side_effect=lambda *a: (call_order.append("a"), ["oa"])[1]),
        "b": MagicMock(side_effect=lambda *a: (call_order.append("b"), ["ob"])[1]),
        "c": MagicMock(side_effect=lambda *a: (call_order.append("c"), ["oc"])[1]),
    }

    with patch.object(assetforge.pipeline.cfgmod, "load", return_value=_make_cfg(stages)), \
         patch.object(assetforge.pipeline.cfgmod, "as_dict", return_value={"stages": stages}), \
         patch.object(assetforge.pipeline.wsmod, "Workspace") as MockWS, \
         patch.object(assetforge.pipeline.wsmod, "fingerprint", return_value="fp"), \
         patch.object(assetforge.pipeline, "REGISTRY", registry):
        ws = _make_ws()
        MockWS.create.return_value = ws
        assetforge.pipeline.run("img.png", "model", only="b")

    assert call_order == ["b"]


def test_resume_skips_fresh_stages():
    """With resume=True, stages where is_fresh() returns True should be skipped."""
    stages = ["a", "b"]
    registry = {
        "a": MagicMock(return_value=["oa"]),
        "b": MagicMock(return_value=["ob"]),
    }

    with patch.object(assetforge.pipeline.cfgmod, "load", return_value=_make_cfg(stages)), \
         patch.object(assetforge.pipeline.cfgmod, "as_dict", return_value={"stages": stages}), \
         patch.object(assetforge.pipeline.wsmod, "Workspace") as MockWS, \
         patch.object(assetforge.pipeline.wsmod, "fingerprint", return_value="fp"), \
         patch.object(assetforge.pipeline, "REGISTRY", registry):
        ws = _make_ws(is_fresh_map={"a": True, "b": False})
        MockWS.create.return_value = ws
        assetforge.pipeline.run("img.png", "model", resume=True)

    # Stage 'a' is fresh → should NOT be called
    assert registry["a"].call_count == 0
    # Stage 'b' is not fresh → should be called
    assert registry["b"].call_count == 1


def test_resume_false_ignores_fresh():
    """With resume=False, even fresh stages should be re-executed."""
    stages = ["a"]
    registry = {"a": MagicMock(return_value=["oa"])}

    with patch.object(assetforge.pipeline.cfgmod, "load", return_value=_make_cfg(stages)), \
         patch.object(assetforge.pipeline.cfgmod, "as_dict", return_value={"stages": stages}), \
         patch.object(assetforge.pipeline.wsmod, "Workspace") as MockWS, \
         patch.object(assetforge.pipeline.wsmod, "fingerprint", return_value="fp"), \
         patch.object(assetforge.pipeline, "REGISTRY", registry):
        ws = _make_ws(is_fresh_map={"a": True})
        MockWS.create.return_value = ws
        assetforge.pipeline.run("img.png", "model", resume=False)

    assert registry["a"].call_count == 1


def test_not_implemented_breaks_pipeline():
    """A stage raising NotImplementedError should stop the pipeline and record 'skipped'."""
    stages = ["a", "b", "c"]
    registry = {
        "a": MagicMock(side_effect=NotImplementedError("stub")),
        "b": MagicMock(return_value=["ob"]),
        "c": MagicMock(return_value=["oc"]),
    }

    with patch.object(assetforge.pipeline.cfgmod, "load", return_value=_make_cfg(stages)), \
         patch.object(assetforge.pipeline.cfgmod, "as_dict", return_value={"stages": stages}), \
         patch.object(assetforge.pipeline.wsmod, "Workspace") as MockWS, \
         patch.object(assetforge.pipeline.wsmod, "fingerprint", return_value="fp"), \
         patch.object(assetforge.pipeline, "REGISTRY", registry):
        ws = _make_ws()
        MockWS.create.return_value = ws
        assetforge.pipeline.run("img.png", "model")

    # 'a' raised → 'b' and 'c' should never be called
    assert registry["a"].call_count == 1
    assert registry["b"].call_count == 0
    assert registry["c"].call_count == 0
    # 'a' should have been recorded with status="skipped"
    ws.record.assert_any_call("a", "fp", {}, status="skipped")


def test_normal_execution_records_outputs():
    """Stages should be recorded via ws.record with their output list."""
    stages = ["a", "b"]
    registry = {
        "a": MagicMock(return_value=["file_a"]),
        "b": MagicMock(return_value=["file_b"]),
    }

    with patch.object(assetforge.pipeline.cfgmod, "load", return_value=_make_cfg(stages)), \
         patch.object(assetforge.pipeline.cfgmod, "as_dict", return_value={"stages": stages}), \
         patch.object(assetforge.pipeline.wsmod, "Workspace") as MockWS, \
         patch.object(assetforge.pipeline.wsmod, "fingerprint", return_value="fp"), \
         patch.object(assetforge.pipeline, "REGISTRY", registry):
        ws = _make_ws()
        MockWS.create.return_value = ws
        assetforge.pipeline.run("img.png", "model")

    ws.record.assert_any_call("a", "fp", ["file_a"])
    ws.record.assert_any_call("b", "fp", ["file_b"])


def test_stages_receive_upstream_outputs():
    """Each stage callable should receive the accumulated upstream dict."""
    stages = ["a", "b"]
    captured = {}

    def stage_a(ws, cfg, upstream, log):
        captured["a_upstream"] = dict(upstream)
        return ["oa"]

    def stage_b(ws, cfg, upstream, log):
        captured["b_upstream"] = dict(upstream)
        return ["ob"]

    registry = {"a": stage_a, "b": stage_b}

    with patch.object(assetforge.pipeline.cfgmod, "load", return_value=_make_cfg(stages)), \
         patch.object(assetforge.pipeline.cfgmod, "as_dict", return_value={"stages": stages}), \
         patch.object(assetforge.pipeline.wsmod, "Workspace") as MockWS, \
         patch.object(assetforge.pipeline.wsmod, "fingerprint", return_value="fp"), \
         patch.object(assetforge.pipeline, "REGISTRY", registry):
        ws = _make_ws()
        MockWS.create.return_value = ws
        assetforge.pipeline.run("img.png", "model")

    # Stage 'a' runs first; upstream should have 'a' mapped to its (empty) prior outputs
    # Since ws.outputs("a") returned {} (falsy), upstream starts empty for 'a'
    assert "a" not in captured["a_upstream"] or captured["a_upstream"].get("a") is None
    # Stage 'b' should see 'a's outputs in upstream
    assert captured["b_upstream"]["a"] == ["oa"]


def test_return_value_is_manifest_stages():
    """run() should return ws.manifest['stages']."""
    stages = ["a"]
    registry = {"a": MagicMock(return_value=["out"])}

    with patch.object(assetforge.pipeline.cfgmod, "load", return_value=_make_cfg(stages)), \
         patch.object(assetforge.pipeline.cfgmod, "as_dict", return_value={"stages": stages}), \
         patch.object(assetforge.pipeline.wsmod, "Workspace") as MockWS, \
         patch.object(assetforge.pipeline.wsmod, "fingerprint", return_value="fp"), \
         patch.object(assetforge.pipeline, "REGISTRY", registry):
        ws = _make_ws()
        ws.manifest = {"stages": {"a": {"outputs": ["out"]}}}
        MockWS.create.return_value = ws
        result = assetforge.pipeline.run("img.png", "model")

    assert result == {"a": {"outputs": ["out"]}}


def test_fingerprint_uses_only_prior_stages():
    """The fingerprint for a stage should be computed from outputs of stages strictly before it."""
    stages = ["a", "b", "c"]
    fp_calls = []

    def fake_fingerprint(cfg_dict, prior_outputs):
        fp_calls.append(dict(prior_outputs))
        return f"fp_{len(fp_calls)}"

    registry = {
        "a": MagicMock(return_value=["oa"]),
        "b": MagicMock(return_value=["ob"]),
        "c": MagicMock(return_value=["oc"]),
    }

    with patch.object(assetforge.pipeline.cfgmod, "load", return_value=_make_cfg(stages)), \
         patch.object(assetforge.pipeline.cfgmod, "as_dict", return_value={"stages": stages}), \
         patch.object(assetforge.pipeline.wsmod, "Workspace") as MockWS, \
         patch.object(assetforge.pipeline.wsmod, "fingerprint", side_effect=fake_fingerprint), \
         patch.object(assetforge.pipeline, "REGISTRY", registry):
        ws = _make_ws()
        MockWS.create.return_value = ws
        assetforge.pipeline.run("img.png", "model")

    # For stage 'a': prior = [] → prior_outputs = {}
    assert fp_calls[0] == {}
    # For stage 'b': prior = ['a'] → prior_outputs = {'a': upstream['a']}
    # upstream['a'] was set to ws.outputs('a') which is {} (falsy), so it's None via .get()
    # Actually: prior_outputs = {s: upstream.get(s) for s in prior}
    # upstream['a'] = {} (from initial build, since ws.outputs('a') = {} which is falsy, it's excluded)
    # Wait: upstream is built as {s: ws.outputs(s) for s in cfg.stages if ws.outputs(s)}
    # ws.outputs('a') returns {} which is falsy, so 'a' is NOT in initial upstream
    # After stage 'a' runs: upstream['a'] = ['oa']
    # So for stage 'b': prior_outputs = {'a': ['oa']}
    assert fp_calls[1] == {"a": ["oa"]}
    # For stage 'c': prior = ['a', 'b'] → prior_outputs = {'a': ['oa'], 'b': ['ob']}
    assert fp_calls[2] == {"a": ["oa"], "b": ["ob"]}