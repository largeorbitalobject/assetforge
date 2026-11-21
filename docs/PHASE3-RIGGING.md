# Phase 3 — Open-ended rigging & animation (the articulation graph)

Researched 2026-06-19 via a deep web survey + on-box ROCm feasibility pass (and a focused
equipment-segmentation study). This supersedes the original PLAN.md Phase 3 ("SkeletonConfig for
creatures") with a strictly more general frame.

## The core idea

Stop thinking "rig = skin a skeleton." The frame that covers **both a minotaur and a turret** is an
**articulation graph**: rigid/deformable **parts** (nodes) connected by **typed joints** (edges) —
`fixed | revolute | prismatic | ball`, each with an axis, pivot/origin, and limits. A skinned
skeleton is just the **dense-revolute-tree special case**.

- **Minotaur** → `kind=organic`, ~52 revolute joints + skin weights.
- **Turret** → `kind=mechanical`, 3 parts, 2 joints: head **revolute about +Z** (yaw), barrel
  **revolute about a lateral axis** (pitch). Empty skin.
- **Sword in hand** → a part with a **fixed/attachment** joint to the hand bone (swap/detach).
- **Armor** → a part **skinned to existing bones** (swap sets, hide/show, per-piece PBR).

## Architecture: one router, two branches, one Blender backend

```
image+mesh ──► VLM classify (MiniCPM-o) ─► kind?
   ├─ organic   ─► RigAnything (joints+parents+skin)  ─► roles auto-detect ─┐
   └─ mechanical ─► part-seg (SAM3+lift / PartField) ─► VLM types joints     │
                     ─► GEOMETRIC axis/pivot estimator (PCA/ring/contact) ───┤
                     ─► (optional) render-and-critique correction loop       │
                                                                             ▼
                                              graph.json  ──►  Blender author (ZERO CUDA)
                                              organic: armature+skin (build_rig.py, reused)
                                              mechanical: bones/empties + DOF-lock constraints (build_artic.py, NEW)
                                                                             ▼
                                                                      rigged GLB ─► animate
```

Both branches converge on **one Blender authoring step** (standalone `blender` binary, no CUDA).
The graph is a strict superset of a URDF (→ trivially exportable for SAPIEN physics validation) and
of a skinned skeleton.

## On-box stack (verified vs dead)

| Role | Use (ROCm-clean / on-box) | Dead on gfx1151 (avoid) |
|---|---|---|
| Organic auto-rig | **RigAnything** ✅ verified (clean-torch, SDPA, 2.2GB ckpt present) | UniRig (spconv+flash-attn), MagicArticulate (flash-attn pin) |
| Mechanical parts | **SAM3 + multiview lift** (weights cached) ✅; **PartField** (needs ROCm `torch-scatter` wheel) | SAMPart3D, Find3D, P3-SAM (spconv/PTv3/Sonata) |
| Joint type/topology | **MiniCPM-o** VLM actor-critic (plugin present) | — |
| Joint axis/pivot | **geometric estimator** (PCA/ring/contact — *don't* trust VLM for metrics) | — |
| Authoring | **Blender 4.5 headless** (build_rig.py reused; build_artic.py new) | — |
| Animation | **procedural DOF keyframing** (mechanical) + generalized walk_pro.py (organic) | Puppeteer (pytorch3d), ArtGS/SPLATART (gaussian CUDA) |
| Multi-state methods | — (we have a single static mesh) | Ditto, PARIS (need observed motion) |

License note: RigAnything is **Adobe non-commercial**; for shipping commercial rigged organics, swap
to **Make-It-Animatable** (likely clean-ROCm). PartField weights are likely NVIDIA non-commercial.

## graph.json schema (the contract between inference and authoring)

Written to `<run>/04_rig/graph.json`, mesh/Blender space (Z-up):

```jsonc
{
  "kind": "organic" | "mechanical",
  "mesh": "…/cleaned.glb", "up_axis": [0,0,1],
  "parts": [ { "id":"base", "face_ids":[…]|"vgroup":"base",
               "centroid":[x,y,z], "bbox":[[…],[…]], "principal_axes":[[…]] } ],
  "joints": [ { "id":"yaw", "parent":"base", "child":"head",
                "type":"revolute", "origin":[x,y,z], "axis":[0,0,1],
                "limits":{"min":-3.14,"max":3.14}, "rest":0.0 } ],
  // organic only:
  "skin": { "joints_xyz":[…], "parents":[…], "weights_npz":"…/inputs.npz" },
  "roles": { "pelvis":0, "spine":[1,2,3], "neck":23, "legs":[{"hip":44,"knee":45,…}], "tail":[…] }
}
```

## Pipeline integration (fills the existing stubs)

- **config.py**: add `mechanical` asset_type; `kind` derives from asset_type (creature→organic,
  mechanical→mechanical), with a MiniCPM-o image classifier promoting ambiguous `object`→mechanical.
  STAGE_MATRIX *is* the router.
- **stages/rig.py** (currently `NotImplementedError`) → dispatcher writing `graph.json` + preview:
  - organic: RigAnything subprocess → graph (skin+roles) → **promote `skel_dump.py` into a real
    SkeletonConfig role classifier** → `build_rig.py` (reused verbatim).
  - mechanical: part-seg → VLM joint topology/type → geometric axis/pivot estimator (~150-line numpy:
    PCA per part, ring detection, contact centroid) → optional render-critique → graph.
- **blender_scripts/build_artic.py** (NEW, highest-value): per part, create a bone/empty at the joint
  origin oriented to its axis; bind the part; set Limit-Rotation/Location constraints so a revolute
  has exactly 1 rotational DOF and prismatic 1 translational DOF. Export GLB (node hierarchy for
  rigid; armature+skin for organic).
- **stages/animate.py** (currently `NotImplementedError`):
  - organic: **config-drive `walk_pro.py` off `graph.json["roles"]`** (kill the hardcoded
    `PELVIS=0/SPINE=[1,2,3]/H=1.21` literals) — this *is* the long-planned SkeletonConfig generalization.
  - mechanical: **driver_artic.py** (NEW, ~100-line bpy) — keyframe each joint's single DOF from a
    small motion-preset DSL (turret scan=sine on yaw; elevate=ramp on pitch; door/drawer open=eased
    revolute/prismatic; lamp/crane=chained revolutes to a target).

## Equipment / accessory separation (sword, armor) — folds in here

Recommended: **segment the fused mesh** (keep Hunyuan gen), text-promptable **by name** via
multiview SAM3/Grounded-SAM-2 lifting (we own `lift_labels.py`) → Blender CPU split/cap/remesh.
- Sword → bone-parent to a **hand socket** (fixed joint; attach/detach/swap). Easy.
- Rigid armor → weight each piece to **one bone** (swap sets, hide/show, per-piece PBR, damage states). Easy.
- "**No body under the armor**" is fundamental (Hunyuan only makes the visible shell) → fit an
  **SMPL-X** body inside (bonus: gives a humanoid skeleton) or mask the covered body at runtime.
- Part-aware *generation* alternative: **PartCrafter** (MIT, TripoSG, clean ROCm) emits separate part
  meshes but unlabeled + object-trained — keep as an experiment, not the spine. Most others
  (OmniPart, Hunyuan3D-Part/X-Part) are spconv/CUDA dead-ends.

## Build order (de-risk the one new unknown first)

0. **MILESTONE 0 — de-risk axis/pivot on the turret fixture.** Hand-author a turret `graph.json` +
   `build_artic.py`; prove in a glTF viewer the head rotates **only** about +Z and the barrel elevates
   **only** about its lateral axis. Isolates the single risky unknown before any plumbing.
1. `build_artic.py` + `driver_artic.py` hardened → turret scans+elevates, door opens, from graph only.
2. Mechanical **inference** front-end via owned code (seg_sam3 + lift_labels + VLM + geometric
   estimator) → turret image+mesh → graph.json, no hand-editing.
3. Organic branch: rig.py = RigAnything → graph → build_rig.py; promote skel_dump → role classifier.
   Acceptance: re-rig minotaur matches `minotaur_rigged.glb`.
4. Generalize animation: config-drive walk_pro.py. **Acceptance gate: rig+walk a second, structurally
   different creature with only a preset — no code edits.**
5. PartField upgrade (build ROCm torch-scatter wheel) + render-critique loop + optional URDF→SAPIEN
   validation. Acceptance: a crane/desk-lamp (multi-joint chain) end-to-end.
6. Polish/licensing: Make-It-Animatable swap for commercial organic output; adopt
   Particulate/URDF-Anything/GEOPARD if they release clean-torch (single-pass mesh→typed-graph).

## Highest risk + honest ceilings

- **Highest risk:** metrically-correct **axis/pivot** for mechanical joints. VLM nails *topology* and
  *type* but is unreliable on exact geometry → derive axis/pivot **geometrically** (ring PCA = yaw
  axis, ring centroid = pivot, barrel long-axis = its axis); VLM only *types* and *picks* the axis;
  close with a render-and-critique ("spinning in place vs orbiting?"). Fallback: hardcoded heuristic
  rules + manual limit touch-up. The turret is the perfect fixture (correct behavior is visually
  unambiguous).
- **Ceilings:** joint **limits** are guesses without observed motion (single static mesh);
  segmentation **granularity** ambiguity moves pivots; Hunyuan **shells** aren't solid bodies (fine
  for keyframed GLB, not for physics); RigAnything **skin** artifacts on thin/overlapping geometry;
  role detection will mislabel **exotic** morphologies (hexapods, wings-as-arms) until VLM-assisted.

## Prerequisites currently missing on-box

- `torch-scatter` ROCm wheel (for PartField) — not installed; the one real porting task.
- `pybullet` / `sapien` (CPU, for optional physics validation) — pip-installable, absent.
- MiniCPM-o server (`:8100`) — plugin present, bring up when needed.
