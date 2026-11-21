# AssetForge

AssetForge turns one image into a production 3D asset on a single AMD Strix Halo
box (gfx1151, ROCm). Drop in a picture of a creature, a person, an object, a
building or a scene and get back a clean, textured and, where the type supports
it, rigged and animated GLB. Nothing leaves the machine.

```
assetforge run concept.png --type creature
```

## Status

The pipeline runs end to end on gfx1151 under ROCm. [`docs/STATUS.md`](docs/STATUS.md)
carries the current detail and [`docs/PLAN.md`](docs/PLAN.md) the design.

| Capability | State |
|---|---|
| Image to mesh (creature, object, mechanical, building) | working, Hunyuan3D-2.1 through the ComfyUI API |
| Image to scene or environment | working, HunyuanWorld (2.5D) plus Track B compositional, editable |
| **Texturing (PBR)** | working, Hunyuan3D-2.1 native painter on ROCm |
| Mesh cleanup | working, decimate to roughly 200k as the production default; retopo opt-in through MeshFlow |
| Auto-rig, creature | working, RigAnything into adaptive auto-roles |
| Auto-rig, mechanical | working, part segmentation plus Qwen3-VL topology plus geometric axes into an articulation graph |
| Animation | working, roles-driven walk for organic, procedural degrees of freedom for mechanical |
| Equipment separation | working, multiview SAM3 part split; quality varies with the input |
| CLI orchestrator | working, resumable, every stage wired |

## Texturing on ROCm

The Hunyuan3D-2.1 native PBR painter was treated as unavailable on AMD because
of ROCm issue #5981, which corrupts `custom_rasterizer` output. On this box it
runs end to end. **The blocker was import order: `import torch` has to come
before `import custom_rasterizer`.** The #5981 corruption does not appear on
this venv's torch nightly, `2.8.0+rocm7.12`, a TheRock gfx12 build from the
family that fixes it.

The recipe and the on-shelf hybrid fallback are in
[`docs/TEXTURING-ROCM.md`](docs/TEXTURING-ROCM.md). The artifact behind the
claim is [`examples/minotaur/minotaur_textured_native.glb`](examples/minotaur/),
with its renders beside it. Texture quality is judged by eye against the input
image; there is no numeric score for it yet, and adding one is the next thing I
want from this stage.

## Layout

```
assetforge/
  cli.py            CLI entrypoint:  assetforge run IMG --type ...
  config.py         per-type presets (presets/*.yaml) + dataclasses
  workspace.py      run-folder schema + manifest.json (resumable)
  pipeline.py       stage orchestrator
  comfy.py          ComfyUI HTTP client (submit graph / poll / fetch GLB)
  blender.py        headless Blender runner
  rocm_env.py       the proven GPU env block (gfx1151)
  stages/           ingest · mesh · cleanup · texture · rig · animate · export
  texturing/        paint.py (native Hunyuan painter, generalized) + hybrid/ fallback
  blender_scripts/  rescued headless scripts (rig / anim / bake / render)
  comfy_graphs/     parameterized ComfyUI workflow templates
  presets/          creature · object · building · scene
  _prototypes/      segmentation/PBR building blocks (future work)
docs/               PLAN.md · TEXTURING-ROCM.md
examples/minotaur/  proven textured GLB + renders
```

## Environment

- ComfyUI 0.9.2 on `:8188`, venv `/home/orbital/ComfyUI/venv` (torch 2.8.0+rocm7.12).
- Blender 4.5.3 LTS headless at `/home/orbital/blender/blender`.
- Hunyuan paint weights, roughly 11.5 GB, cached under `hy3dpaint/ckpt/` and the
  HF cache. They download once.
- The GPU environment lives in one place, `assetforge/rocm_env.py`.
  **Keep `LD_LIBRARY_PATH` clean:** injecting `_rocm_sdk` or `/opt/rocm-7.1.0`
  shadows the venv's ROCm 7.12 and breaks linking.

## Third-party models

AssetForge calls models it does not ship. Each one has its own license, and you must accept those terms yourself:
Hunyuan3D-2.1 and HunyuanWorld (Tencent Hunyuan community license, which excludes the EU, UK and South Korea),
SAM 3 (Meta SAM license), RigAnything, MeshFlow (Meta, its own license), Qwen3-VL (Apache-2.0) and Blender (GPL, run as an external program).
No weights are included here.

## License

Apache License 2.0 for the code in this repository. See `LICENSE`.
