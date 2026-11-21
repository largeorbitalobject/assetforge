# AssetForge — current status (2026-06-20)

Local image→3D production pipeline on AMD Strix Halo gfx1151/ROCm. Everything runs on-box; CLI:
`assetforge run IMG --type creature|mechanical|object|building|scene [--until S] [--retopo] [--draft]`.

## Working end-to-end (verified)
| Stage | State | Notes |
|---|---|---|
| ingest | ✅ | rembg subject isolation (skip for scene) |
| mesh | ✅ | Hunyuan3D-2.1 (object/creature/mechanical/building) · HunyuanWorld (scene, 2.5D) via ComfyUI |
| cleanup | ✅ | **decimate to ~200k = production default** (clean texture + animates); opt-in `--retopo` (MeshFlow) |
| texture | ✅ | native Hunyuan PBR painter on ROCm (import-torch-first fix); ~100s; unwrap cached by mesh hash |
| rig (creature) | ✅ | RigAnything → adaptive auto-roles → build_rig; minotaur + swordsman, no per-model code |
| rig (mechanical) | ✅ | SAM3 part-seg + Qwen3-VL topology + geometric axis/pivot → build_artic (turret, pedal) |
| animate (creature) | ✅ | roles-driven FK walk, any limb count |
| animate (mechanical) | ✅ | procedural DOF driver (turret scan/elevate, pedal knobs spin) |
| export | ✅ | final GLB + preview render |

All 6 example inputs processed. Reasoning (scene hero-naming, mechanical joint topology) uses local
**Qwen3-VL** (Ollama :11434) — vision-based (renders fed to the VLM).

## Capabilities
- **Texturing**: native Hunyuan painter, ROCm-clean (#5981 absent on the rocm7.12 nightly). docs/TEXTURING-ROCM.md
- **Open-ended rigging**: articulation graph (parts + typed joints) unifies organic skeletons AND
  mechanical objects (turret/pedal). docs/PHASE3-RIGGING.md
- **Compositional environments (Track B)**: decompose (SAM3) → MoGe metric layout → per-hero gen →
  Blender assembly → editable scene GLB. docs/ENVIRONMENTS.md
- **Equipment / part separation**: multiview SAM3 lift → EXR face-ids → Blender bmesh split (coherent
  per-instance parts; quality input-dependent).
- **Retopo (opt-in)**: MeshFlow runs on ROCm (dense→clean artist topology). docs/RETOPO-MESHFLOW.md

## Production guidance (poly count / quality)
- **Default ~200k decimation is the sweet spot**: clean painted texture + animates fine. Animation
  already works at 137k–400k, so low-poly is NOT needed for animation — only for engine perf/size.
- Texturing *directly* on aggressive low-poly is patchy (painter conditions on the mesh's own coarse
  normal/position maps). So `--retopo` (MeshFlow ultra-low) is experimental, not the default.
- True engine-low-poly with good texture = bake-from-dense (`bake_hi_to_lo.py`, WIP — see below).

## Honest limitations / WIP
- **bake_hi_to_lo.py WIP**: selected-to-active Blender bake produces black; needs debugging (the
  proper hi→lo texture transfer for engine-budget low-poly).
- **Part-seg quality input-dependent**: clean on distinct parts (pedal knobs); empty/scattered on
  fused-ambiguous parts (swordsman's fused hair+blade — SAM3 can't isolate it).
- **Scenes are 2.5D** via single-image HunyuanWorld-Mirror; Track B compositional is the better path
  (placement/scale/lighting need a calibration pass for polished multi-object renders).
- **Resume bug**: stale "completed" manifest records can wrongly skip stages when code/inputs change
  out-of-band; `--no-resume` / `--only` (now forces) work around it. Proper fix = hash code+mesh into
  the stage fingerprint (TODO).
- **MeshFlow / RigAnything / MoGe / PartField weights are non-commercial** (research/internal); a
  commercial release needs permissively-licensed substitutes.

## Licenses to track for commercial use
RigAnything (Adobe NC), MeshFlow (FAIR NC), MoGe/PartField (likely NVIDIA NC). Texturing
(Hunyuan3D-2.1) + Blender + SAM/rembg are usable; audit before shipping.
