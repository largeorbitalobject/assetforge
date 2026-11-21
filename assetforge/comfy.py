"""Minimal ComfyUI HTTP client: submit an API-format graph, wait, locate output files.

Verified against ComfyUI 0.9.2 on :8188. Outputs land under ComfyUI/app/output/.
"""
from __future__ import annotations
from pathlib import Path
import json
import time
import uuid
import urllib.request
import urllib.error

from .rocm_env import COMFY_APP

DEFAULT_HOST = "http://localhost:8188"
OUTPUT_DIR = Path(COMFY_APP) / "output"


class ComfyError(RuntimeError):
    pass


def _get(host: str, path: str) -> dict:
    with urllib.request.urlopen(f"{host}{path}", timeout=30) as r:
        return json.loads(r.read())


def _post(host: str, path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{host}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def is_live(host: str = DEFAULT_HOST) -> bool:
    try:
        _get(host, "/system_stats")
        return True
    except (urllib.error.URLError, OSError):
        return False


def free(host: str = DEFAULT_HOST) -> bool:
    """Ask ComfyUI to unload models + free GPU memory.

    Important on this APU: ComfyUI and the painter subprocess share ONE unified memory pool,
    so a mesh model left resident after the mesh stage can starve the painter. Call this before
    the texture stage. Best-effort — returns False if the server isn't reachable.
    """
    # /free returns an empty body (not JSON), so post raw and ignore the response.
    try:
        data = json.dumps({"unload_models": True, "free_memory": True}).encode()
        req = urllib.request.Request(f"{host}/free", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30):
            return True
    except (urllib.error.URLError, OSError):
        return False


def submit(graph: dict, host: str = DEFAULT_HOST) -> str:
    """Queue an API-format graph; return its prompt_id."""
    client_id = uuid.uuid4().hex
    resp = _post(host, "/prompt", {"prompt": graph, "client_id": client_id})
    if resp.get("node_errors"):
        raise ComfyError(f"node errors: {resp['node_errors']}")
    return resp["prompt_id"]


def wait(prompt_id: str, host: str = DEFAULT_HOST, timeout: float = 1200, poll: float = 1.0) -> dict:
    """Poll /history until the prompt completes; return its history entry."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = _get(host, f"/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise ComfyError(f"execution error: {status}")
            return entry
        time.sleep(poll)
    raise ComfyError(f"timed out after {timeout}s waiting for {prompt_id}")


def output_files(history_entry: dict, ext: str = ".glb") -> list[Path]:
    """Resolve absolute paths of saved outputs (GLB/PLY/PNG) from a history entry."""
    found: list[Path] = []
    for node_out in history_entry.get("outputs", {}).values():
        for key in ("3d", "gltf", "images", "files", "meshes"):
            for item in node_out.get(key, []) or []:
                fn = item.get("filename")
                if not fn or not fn.lower().endswith(ext):
                    continue
                found.append(OUTPUT_DIR / item.get("subfolder", "") / fn)
    return found


def run_graph(graph: dict, host: str = DEFAULT_HOST, ext: str = ".glb", timeout: float = 1200) -> list[Path]:
    """Submit + wait + return output paths. Convenience wrapper."""
    if not is_live(host):
        raise ComfyError(f"ComfyUI not reachable at {host}")
    pid = submit(graph, host)
    entry = wait(pid, host, timeout=timeout)
    return output_files(entry, ext=ext)
