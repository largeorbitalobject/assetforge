# Generated tests

These tests were written by a 29-agent **Qwen3.8-27B** swarm (bf16, vLLM on the
Strix Halo 8060S) on 2026-08-14, one agent per importable `assetforge` module.

**Every file here was executed and passed before being committed.** Files that
failed, errored, or produced no parseable code were dropped — 17 of 26 candidates
were kept (336 tests). They run in ~0.3s:

```bash
pytest tests
```

## Provenance / caveats

- Generated with `reasoning_effort=low` and Qwen's documented thinking-mode
  sampling (`temperature=1.0, top_p=0.95, top_k=20`). An earlier run using
  defaults (`reasoning_effort=xhigh`) produced ~8x more failures — the config
  matters far more than the prompt.
- Tests were validated inside `docker --network none` with read-only mounts,
  since they are model-authored code. They mock `subprocess`/`requests` rather
  than calling them, but they have not been line-by-line human reviewed.
- Coverage is skewed toward pure logic. Modules needing `bpy`, `torch` or `cv2`
  were excluded because they cannot be imported outside Blender/GPU contexts.
- Not covered (no all-green version): `articulation.roles`, `comfy`,
  `scene.compose`, `scene.decompose`, `scene.depth`, `scene.layout`,
  `scene.point_mesh`, `stages.cleanup`, `stages.mech_rig`.

## Bug this run found

`scene/point_mesh.py:29` raised `ZeroDivisionError` for any single-row or
single-column grid (`i/(w-1)`, `j/(h-1)`). Fixed 2026-08-14 by guarding both
divisions; degenerate grids now yield 0 faces instead of crashing.
