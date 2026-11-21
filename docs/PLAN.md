# AssetForge — production image→3D asset pipeline

## Context

Today the "3D pipeline" is a working proof but **not a product**: ~40 bespoke scripts living in an
*ephemeral* job tmp dir (`/home/orbital/.claude/jobs/fde14dc0/tmp/`), every one hand-tuned to a single
minotaur model, none version-controlled. Mesh generation, rigging, and animation all work; texturing was
abandoned as "too low-quality locally."

Goal (decided with the user): turn this into a **legit production pipeline** — a CLI asset generator that
takes an image of *anything* (creature/person, object/prop, building, or full scene) and produces a clean,
textured, and (where applicable) rigged + animated GLB. Constraints locked in:

- **100% local / on-box.** Nothing leaves the machine. (Strix Halo gfx1151 / ROCm — no CUDA.)
- **Push texturing quality hard locally** — revive the abandoned texturing chain, don't skip it.
- **All four input types** supported, branching as needed.
- **Form factor = CLI / scripted orchestrator** — a versioned, installable Python package, reproducible & logged.

The good news from exploration: the texturing chain is **~90% already built** (just minotaur-hardcoded), the
scene branch is viable (HunyuanWorld nodes installed, no CUDA blockers, already emitting GLB/PLY in
`ComfyUI/app/output/`), and the ComfyUI HTTP API + headless Blender mechanics are confirmed. This is mostly a
**consolidation + generalization** effort, not green-field.

## Where the pieces are now

- ComfyUI 0.9.2 running on :8188. Driveable via `POST /prompt` → poll `GET /history/<id>` → read GLB from
  `/home/orbital/ComfyUI/app/output/`. Native Hunyuan3D-2.1 graph already saved at
  `comfyui_workflows/hunyuan3d-2.1-image-to-glb.json`.
- Scene branch: `ComfyUI-HunyuanWorld-Mirror` + `ComfyUI_HunyuanWorldnode` custom nodes installed, ROCm-clean
  (no spconv/nvdiffrast/pytorch3d/flash-attn). Weights auto-download (~10–30 GB) on first use. Output = GLB mesh
  (+ optional Gaussian PLY).
- RigAnything cloned + ROCm-working at `tools/RigAnything/` (weights present, `inference.py` patched per memory).
- Keeper scripts in the job tmp (to be rescued into the package): `build_rig.py`, `skel_dump.py`, `tex_prep.py`,
  `bake_maps.py`, `bake_cov.py`, `gather_bake.py`, `lift_labels.py`, `seg_sam3.py`, `make_pbr.py`,
  `apply_pbr.py`, `apply_tex.py`, `submit_tex.py`, `walk_pro.py`, `ik_walk3.py`.
- Local VLM available (MiniCPM-o, per memory) — usable for auto-captioning inputs to drive texture prompts.

## Target architecture

New installable package living at **`orbitis-station/pipeline/`** (co-located with assets, RigAnything, and
comfyui_workflows; can graduate to its own repo later). Package name `assetforge`, CLI entry `assetforge`.

```
pipeline/
  pyproject.toml                 # installable; console_scripts: assetforge = assetforge.cli:main
  README.md
  assetforge/
    cli.py            # `assetforge run IMG --type creature|object|building|scene [--resume] [--only STAGE]`
    config.py         # dataclasses + per-type YAML presets (presets/*.yaml)
    workspace.py      # run-folder schema + manifest.json (stage status, input hash, params, outputs)
    pipeline.py       # stage orchestrator: resumable, idempotent, logged, per-stage preview render
    comfy.py          # ComfyUI client: submit graph JSON, poll, locate output GLB
    blender.py        # headless Blender runner (subprocess + arg passing to blender_scripts/*)
    stages/
      ingest.py       # bg-removal / subject isolation (rembg/BiRefNet); skip for scene
      mesh.py         # branch: hunyuan3d (creature/object/building) | hunyuanworld (scene)
      cleanup.py      # Blender: drop floaters, voxel/Quadriflow remesh, decimate, Smart-UV unwrap
      texture.py      # drives hy3dpaint (Hunyuan native painter); hybrid bake fallback (see Phase 2)
      rig.py          # RigAnything -> build_rig -> NN skin transfer   [creature/person only]
      animate.py      # skeleton-role detect -> parametric motion       [creature/person only]
      export.py       # final GLB + preview montage + manifest finalize
    blender_scripts/  # rescued, generalized headless scripts (no hardcoded minotaur)
    comfy_graphs/     # parameterized workflow templates
    presets/          # creature.yaml object.yaml building.yaml scene.yaml
```

**Run workspace** (replaces the flat `assets/0x_*` dirs): `assets/runs/<slug>/{00_input,01_mesh,02_clean,
03_texture,04_rig,05_anim,06_final}/` + `manifest.json`. Each stage records input hash + config; a stage is
skipped on `--resume` when nothing it depends on changed.

**Type → stage matrix**

| type | ingest | mesh | cleanup | texture | rig | animate |
|---|---|---|---|---|---|---|
| creature / person | isolate | hunyuan3d | remesh+UV | ✓ | ✓ | ✓ |
| object / prop | isolate | hunyuan3d | remesh+UV | ✓ | — | — |
| building | isolate | hunyuan3d (hi-res octree) | remesh+UV | ✓ | — | — |
| scene | — | hunyuanworld | light clean | bake/native | — | — |

## Key generalizations (the actual work)

1. **`SkeletonConfig`** (new, in `stages/animate.py` + `blender_scripts/skel_dump.py`): parse the dumped skeleton
   and auto-map roles (pelvis / spine / neck / arms / legs / tail) from geometry (height, laterality, parent
   chains) + optional preset hints. Replaces every hardcoded `PELVIS=0 / SPINE=[1,2,3] / H=1.21` index in the
   animation scripts. This is what unblocks animating *arbitrary* creatures, not just the minotaur. Generalize
   `walk_pro.py` (best parameterized walk) + `ik_walk3.py` (tail dynamics) to read roles from this config.

2. **Texturing method = revive Hunyuan3D-2.1's native painter (hy3dpaint)** — DECIDED with user. The old
   "generate 6 independent SDXL views → average onto mesh" paradigm is abandoned entirely: independent views
   disagree, and back-projection averaged the disagreements into washout. Hunyuan's painter is purpose-built and
   **multiview-consistent** (cross-view attention), which is the exact property the old method lacked.
   The install lives at `ComfyUI/app/custom_nodes/ComfyUI-Hunyuan3d-2-1.broken/hy3dpaint/`. Breakdown of parts:
   - `hunyuanpaintpbr/` (UNet + `pipeline.py`) — the multiview-consistent **PBR** diffusion model (albedo + metal/
     rough). **Pure PyTorch → runs on ROCm as-is.** This is the quality engine and the part we want.
   - `textureGenPipeline.py` + `DifferentiableRenderer/MeshRender.py` — orchestration: generate views → rasterize
     → bake to UV atlas → inpaint seams.
   - `custom_rasterizer/` — **the AMD wall.** A `CUDAExtension` (`rasterizer.cpp` + `rasterizer_gpu.cu`); the
     shipped `.so` was built against a stale torch ABI (libc10 not found) and ROCm issue #5981 reports corrupt
     texture output. Used at two call sites in `MeshRender.py`: **view rasterization** (visibility per camera,
     lines 428/453/477) and **UV rasterization** (texel→face for the atlas, lines 935/1059).
   - `DifferentiableRenderer/mesh_inpaint_processor` — a separate **CPU** C++/pybind11 ext; trivially rebuilt via
     the one-line `compile_mesh_painter.sh`. Not a real blocker.

   **Strategy — rebuild first, hybrid fallback if it corrupts:**
   - **Path A (native):** Rebuild `custom_rasterizer` against the current ComfyUI ROCm venv (torch 2.8+rocm7.12;
     torch's hipify translates the `.cu`) to fix the ABI break; rebuild `mesh_inpaint_processor`. Run
     `textureGenPipeline.py` on the minotaur and **inspect for the #5981 corruption.** If clean, this is the
     full native painter — best quality, done.
   - **Path B (hybrid, if rasterizer corrupts/won't build — likely):** Keep Hunyuan's multiview-consistent PBR
     diffusion output, but **swap the broken rasterizer for the proven Blender/numpy projector we already own** —
     `bake_maps.py` does UV-space rasterization (world-pos/normal per texel) and `gather_bake.py` does per-camera
     visibility projection with z-buffer. Wrap them behind a `raster_rasterize`-compatible shim so `MeshRender`
     calls our backend. Because Hunyuan's views now *agree*, back-projection no longer washes out — the original
     failure mode is gone regardless of projector.
   - Optional polish: local MiniCPM-o VLM caption of the input to enrich the paint conditioning prompt.

3. **Config-drive everything that was hardcoded:** view lists, atlas size, occlusion epsilon, concept→PBR
   material table, diffusion model/ControlNet/IP-Adapter names + strengths, decimation/octree targets — all move
   into per-type preset YAML.

## Phased execution

- **Phase 0 — Scaffold & rescue (spine).** Create package, `pyproject`, `workspace`/manifest, `comfy.py`,
  `blender.py`. Rescue keeper scripts into `blender_scripts/`. Wire `cli → pipeline` for the minimal path
  `ingest → mesh → cleanup → export`. Acceptance: `assetforge run <object.png> --type object` yields a clean
  untextured GLB + preview, fully via CLI, resumable.
- **Phase 1 — Geometry, all types.** Add ingest/segmentation (rembg/BiRefNet), the HunyuanWorld **scene** branch,
  building hi-res settings. Acceptance: each of the 4 types produces a clean GLB end-to-end.
- **Phase 2 — Texturing via Hunyuan native painter (local, hard push).** Per "Key generalization #2": (a) stand
  up `hunyuanpaintpbr` multiview diffusion on the ROCm venv; (b) rebuild `custom_rasterizer` + `mesh_inpaint_
  processor`, run `textureGenPipeline.py` on the minotaur, inspect for #5981 corruption; (c) if corrupt, build
  the hybrid: `raster_rasterize` shim backed by `bake_maps.py` + `gather_bake.py`; (d) wrap as `stages/texture.py`.
  Acceptance: minotaur + one new asset come out with clean, **seam-free, identity-matching** PBR texture that
  clearly beats the old washed-out attempt (side-by-side render vs the saved `minotaur_textured_preview.png`).
- **Phase 3 — Rig + animate (creature/person).** Build `SkeletonConfig` auto-detection; integrate RigAnything
  run + generalized `build_rig`; generalized parametric walk/idle from detected roles; optional Mixamo retarget
  for clean bipeds. Acceptance: re-rig minotaur (matches known-good `minotaur_rigged.glb`) **and** rig+walk a
  different biped/creature with no per-model code edits.
- **Phase 4 — Polish.** Batch mode (`assetforge batch dir/`), preview gallery, docs/README, prune the redundant
  ~8 GB original-format Hunyuan weights once native-only is confirmed.

## Verification

- **Per stage:** every stage writes a preview render; `manifest.json` is the source of truth for resume.
- **Golden runs:** keep `assets/00_concepts/orbitis_minotaur.png` as a regression fixture — its mesh/rig/anim
  outputs are already known-good, so each phase re-runs it and diffs renders against the saved minotaur assets.
- **Generalization test:** Phase 3 must rig+animate a *second*, structurally different subject (a clean biped)
  with zero code changes — only a preset — to prove the minotaur-hardcoding is gone.
- **ComfyUI/Blender smoke:** `comfy.py` self-test hits `/system_stats`; `blender.py` self-test imports+renders a
  GLB headless. Both confirmed working in exploration.

## Notes / risks

- Work happens in a git worktree of `orbitis-station` (it's a git repo); ComfyUI stays running on :8188.
- HunyuanWorld weights (~10–30 GB) download on first scene run — flag to user before pulling (disk-conscious).
- Scene-branch texturing differs from per-object (HunyuanWorld emits its own textured mesh / splat) — Phase 1
  will confirm its native output quality before deciding whether the Phase-2 chain even applies to scenes.
- This plan consolidates and supersedes the scattered tmp scripts; `3d-pipeline-LOCAL-plan.md` becomes history.
- **Texturing is the highest-risk phase** (user chose the high-ceiling Hunyuan-native path). The rebuild may not
  clear ROCm #5981 corruption — but the hybrid fallback reuses already-working code (`bake_maps`/`gather_bake`)
  and only depends on the pure-torch multiview model, so a quality result is reachable either way. The old
  SDXL-multiview-average chain is retired, not patched.
