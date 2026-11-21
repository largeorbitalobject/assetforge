# Better 3D environments — research verdict + plan (2026-06-20)

Answers "how do we build better 3D environments" (the single-image HunyuanWorld-Mirror scene
branch only yields a torn 2.5D relief). Deep web survey + on-box ROCm feasibility.

## Root cause
The torn holey sheet is a SINGLE-INPUT-VIEW artifact, not a model limit: one image -> one depth
map -> one displaced plane can only stretch/tear behind whatever it can't see. The fix is to give
the reconstructor (or assembler) information about the occluded geometry.

## Two assumptions overturned (verified on-box)
- **HunyuanWorld-Mirror already accepts a multi-view IMAGE batch** and exports BOTH a textured mesh
  (`glb_path`) AND gaussian splats (`ply_path`) + depth/normal/camera. So feeding several views fuses
  away the holes — no new model needed.
- **gsplat 1.5.3 works on ROCm** (`from gsplat import rasterization` succeeds on torch 2.8+rocm7.12).
  3DGS is NOT dead here — only graphdeco's `diff-gaussian-rasterization` is. Splats are navigable on-box.

## Verified ROCm-clean & on-box (the toolkit)
Hunyuan3D-2.1 per-object mesh+PBR (the reuse anchor, weights cached) · HunyuanWorld-Mirror (multi-view,
cached 4.8G) · SAM3/SAM3.1 + rembg (segmentation) · Depth-Anything-V2 + Metric3D (via controlnet_aux) ·
FLUX.1-dev/SDXL/SD3.5 (panorama/novel-view/inpaint) · DINOv2 (cached; orientation) · Blender 4.5
(assembly) · gsplat 1.5.3 (splat render). ADD: MoGe-2 (trivial pip; DINOv2 backbone cached) for metric layout.

## Dead on gfx1151 (do not pursue)
LayerPano3D / DreamScene360 / SceneDreamer360 / WonderWorld / WonderJourney / GALA3D (diff-gaussian +
simple-knn / pytorch3d CUDA) · SceneGen + any TRELLIS-based pano (spconv+flash-attn+nvdiffrast+kaolin) ·
HunyuanWorld-Voyager (video-diffusion world model, too heavy). Note: on-box `ComfyUI_HunyuanWorldnode`
is a MISNAMED Hunyuan3D-2 wrapper, NOT HunyuanWorld 1.0.

## The plan — two tracks

### TRACK A — multi-view Mirror (quick win, zero installs)
Generate 3-6 small-baseline novel views in ComfyUI (FLUX/SDXL img2img + inpaint, Depth-Anything-V2 /
ControlNet guidance, IPAdapter to lock the MJ palette) -> feed the batch to the existing Mirror node
(tune confidence_filter/boundary_erosion) -> navigate the splat PLY via gsplat, edit the GLB in Blender.
Ship as `mesh_backend: hunyuanworld_mv`. Result: a navigable, far-less-holey backdrop bubble — but NOT
cleanly per-object editable.

### TRACK B — compositional assembly (the editable-scene target, PRIMARY)
The CAST/ZeroScene recipe, reusing the proven per-object pipeline verbatim:
1. DECOMPOSE: MiniCPM-o names heroes vs amorphous stuff; SAM3 per-instance masks.
2. METRIC GEOMETRY: MoGe-2 (add) / Metric3D (on-box) -> metric XYZ per pixel -> RANSAC ground plane,
   per-hero footprint position + scale + occlusion order. Depth-Anything-V2 = terrain heightfield.
3. AMODAL FIX: FLUX.1-Fill/SDXL inpaint each masked hero crop into a COMPLETE object image (Hunyuan3D
   makes garbage from occluded crops); same step yields the hole-free background plate.
4. PER-OBJECT GEN (UNCHANGED REUSE): Hunyuan3D-2.1 mesh + native PBR per hero -> clean editable GLB.
5. ORIENT: DINOv2/Orient-Anything yaw per hero (fallback VLM).
6. BACKDROP: background plate -> 360 panorama (FLUX/SDXL 360 LoRA, small download) -> equirect depth.
7. ASSEMBLE in Blender headless: emission skydome + depth-displaced terrain shell + heroes scattered at
   metric positions/scales/yaws; optional rigidbody settle + LLM/VLM-critic refine. Export scene GLB.
Result: navigable AND fully editable (each hero a re-texturable GLB) — vastly beats the relief.

## Build order
0. Track A N-view generator + Mirror batch (hours, zero installs) — immediate visible upgrade.
1. **De-risk FIRST: multi-view diffusion CONSISTENCY** on the MJ aesthetic (the one real unknown) —
   probe on the coral garden: do small-baseline generated views fuse cleanly or ghost? If poor, Track B
   becomes primary (it only needs single-object amodal crops, an easier diffusion ask).
2. MoGe-2 metric layout spine (back-projection + ground plane + position/scale) -> layout.json.
3. Track B core: scene_assemble.py (decompose -> amodal -> Hunyuan3D heroes -> Blender assembly -> GLB).
4. Backdrop polish: 360 panorama + equirect depth dome/terrain; settle + critic refine loop.
5. (optional) pilot full HunyuanWorld 1.0 layered-mesh backend (fresh install; EU/UK/SK license excluded).

## Highest risk
Multi-view diffusion CONSISTENCY on the stylized aesthetic (gates Track A + Track B amodal/backdrop).
De-risk with a narrow probe before building. Secondary: monocular depth scale ambiguity -> MoGe-2 metric.

## Honest ceilings
Bounded explorable scene from one viewpoint, not unbounded walking. Far-field backdrop = skydome shell
(limited parallax). Amorphous terrain/foliage/water is the weak link (displaced shell / procedural, coarser
than hero objects). MJ pixel-art style drift in generated views (mitigated by IPAdapter/ControlNet, not
eliminated). Per-hero Hunyuan3D gen dominates wall-clock (minutes-to-tens-of-minutes per scene; offline).
