"""Tests for assetforge.rocm_env — pure-logic, no GPU/Blender/network required."""

import os

import pytest

from assetforge.rocm_env import (
    BLENDER,
    COMFY_APP,
    COMFY_ROOT,
    COMFY_VENV_PY,
    GPU_ENV,
    HY_NODE,
    HY_PAINT,
    apply_gpu_env,
    gpu_env,
)


# ─── Constants ───────────────────────────────────────────────────────────────


class TestConstants:
    def test_comfy_root(self):
        assert COMFY_ROOT == "/home/orbital/ComfyUI"

    def test_comfy_app(self):
        assert COMFY_APP == "/home/orbital/ComfyUI/app"

    def test_comfy_venv_py(self):
        assert COMFY_VENV_PY == "/home/orbital/ComfyUI/venv/bin/python"

    def test_hy_node(self):
        assert HY_NODE == (
            "/home/orbital/ComfyUI/app/custom_nodes/ComfyUI-Hunyuan3d-2-1.broken"
        )

    def test_hy_paint(self):
        assert HY_PAINT == (
            "/home/orbital/ComfyUI/app/custom_nodes/ComfyUI-Hunyuan3d-2-1.broken/hy3dpaint"
        )

    def test_blender(self):
        assert BLENDER == "/home/orbital/blender/blender"


# ─── GPU_ENV dict ────────────────────────────────────────────────────────────


class TestGpuEnvDict:
    def test_expected_keys(self):
        expected = {
            "HSA_OVERRIDE_GFX_VERSION",
            "HIP_VISIBLE_DEVICES",
            "PYTORCH_HIP_ALLOC_CONF",
            "HSA_ENABLE_SDMA",
            "HSA_USE_SVM",
            "HSA_FORCE_FINE_GRAIN_PCIE",
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
            "PYTORCH_TUNABLEOP_ENABLED",
            "MIOPEN_FIND_MODE",
        }
        assert set(GPU_ENV.keys()) == expected

    def test_hsa_override_gfx_version(self):
        assert GPU_ENV["HSA_OVERRIDE_GFX_VERSION"] == "11.5.1"

    def test_hip_visible_devices(self):
        assert GPU_ENV["HIP_VISIBLE_DEVICES"] == "0"

    def test_hsa_enable_sdma(self):
        assert GPU_ENV["HSA_ENABLE_SDMA"] == "0"

    def test_hsa_use_svm(self):
        assert GPU_ENV["HSA_USE_SVM"] == "0"

    def test_hsa_force_fine_grain_pcie(self):
        assert GPU_ENV["HSA_FORCE_FINE_GRAIN_PCIE"] == "1"

    def test_miope_find_mode(self):
        assert GPU_ENV["MIOPEN_FIND_MODE"] == "FAST"

    def test_pytorch_tunableop_disabled(self):
        assert GPU_ENV["PYTORCH_TUNABLEOP_ENABLED"] == "0"

    def test_pytorch_hip_alloc_conf(self):
        conf = GPU_ENV["PYTORCH_HIP_ALLOC_CONF"]
        assert "expandable_segments:True" in conf
        assert "garbage_collection_threshold:0.6" in conf
        assert "max_split_size_mb:8192" in conf

    def test_no_ld_library_path_key(self):
        assert "LD_LIBRARY_PATH" not in GPU_ENV


# ─── gpu_env() function ──────────────────────────────────────────────────────


class TestGpuEnvFunction:
    def test_returns_dict(self):
        result = gpu_env(base={})
        assert isinstance(result, dict)

    def test_all_gpu_env_keys_present_with_custom_base(self):
        base = {"SOME_VAR": "hello"}
        result = gpu_env(base=base)
        for key, value in GPU_ENV.items():
            assert result[key] == value

    def test_preserves_base_keys(self):
        base = {"SOME_VAR": "hello", "ANOTHER": "world"}
        result = gpu_env(base=base)
        assert result["SOME_VAR"] == "hello"
        assert result["ANOTHER"] == "world"

    def test_gpu_env_overrides_base(self):
        base = {"HSA_ENABLE_SDMA": "1", "HIP_VISIBLE_DEVICES": "1,2,3"}
        result = gpu_env(base=base)
        assert result["HSA_ENABLE_SDMA"] == "0"
        assert result["HIP_VISIBLE_DEVICES"] == "0"

    def test_does_not_mutate_input_base(self):
        base = {"SOME_VAR": "hello"}
        snapshot = dict(base)
        gpu_env(base=base)
        assert base == snapshot

    def test_does_not_add_ld_library_path(self):
        base = {}
        result = gpu_env(base=base)
        assert "LD_LIBRARY_PATH" not in result

    def test_preserves_existing_ld_library_path_untouched(self):
        base = {"LD_LIBRARY_PATH": "/some/keep/path"}
        result = gpu_env(base=base)
        assert result["LD_LIBRARY_PATH"] == "/some/keep/path"

    def test_default_base_uses_os_environ(self):
        # Inject a sentinel into os.environ, verify it appears in result
        os.environ["__ROCM_ENV_TEST_SENTINEL__"] = "sentinel_value"
        try:
            result = gpu_env()
            assert result["__ROCM_ENV_TEST_SENTINEL__"] == "sentinel_value"
            # And GPU_ENV keys are still applied
            assert result["HSA_OVERRIDE_GFX_VERSION"] == "11.5.1"
        finally:
            del os.environ["__ROCM_ENV_TEST_SENTINEL__"]

    def test_result_is_new_dict_not_os_environ(self):
        result = gpu_env()
        assert result is not os.environ
        # Mutating result should not affect os.environ
        result["__SHOULD_NOT_LEAK__"] = "x"
        assert "__SHOULD_NOT_LEAK__" not in os.environ


# ─── apply_gpu_env() function ────────────────────────────────────────────────


class TestApplyGpuEnv:
    def test_applies_all_gpu_env_vars_to_os_environ(self, monkeypatch):
        # Clear the GPU vars from os.environ first to have a clean slate
        for key in GPU_ENV:
            monkeypatch.delenv(key, raising=False)

        apply_gpu_env()

        for key, value in GPU_ENV.items():
            assert os.environ[key] == value

    def test_does_not_add_ld_library_path(self, monkeypatch):
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        apply_gpu_env()
        assert "LD_LIBRARY_PATH" not in os.environ

    def test_overwrites_existing_values(self, monkeypatch):
        monkeypatch.setenv("HSA_ENABLE_SDMA", "1")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,2,3")
        apply_gpu_env()
        assert os.environ["HSA_ENABLE_SDMA"] == "0"
        assert os.environ["HIP_VISIBLE_DEVICES"] == "0"