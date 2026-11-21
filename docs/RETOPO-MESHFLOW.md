# Retopology via MeshFlow (opt-in) — clean topology for the dense Hunyuan meshes

The mesh-quality bottleneck: Hunyuan3D emits a dense marching-cubes/surface-net mesh (no edge
loops) and the texture export unwelds it into a triangle soup. **MeshFlow** (facebookresearch/
meshflow, CVPR'26) retopologizes: point-cloud → clean artist-topology triangle mesh, ~1s.

## ROCm verdict: LIKELY-RUNS (rare clean repo)
Pure PyTorch — SDPA attention (`F.scaled_dot_product_attention`, built-in flash path, NOT the
flash-attn pip pkg), CPU geometry (trimesh.sample + numpy graph triangulation), DINOv3 via
torch.hub (image-conditioning only, not needed for point-cloud mode). **Zero** spconv / Minkowski /
pytorch3d-CUDA / nvdiffrast / diff-gaussian / tiny-cuda-nn / torch-scatter / `.cu` / CUDAExtension.
Only ROCm-relevant line: `--extra-index-url .../cu128` in requirements.txt → drop it (ROCm torch 2.8
already present).

## Two caveats (user decisions)
- **Gated weights**: `huggingface.co/facebook/meshflow` — accept terms + HF token to download.
- **License = FAIR Noncommercial Research** (outputs included) → research/internal only; **cannot
  ship in a commercial product**. This is why the retopo stage is OFF by default (keeps the default
  pipeline commercial-clean).

## Activate (one-time)
```bash
git clone https://github.com/facebookresearch/meshflow ~/meshflow
cd ~/meshflow && sed -i '/extra-index-url/d' requirements.txt
/home/orbital/ComfyUI/venv/bin/pip install -r requirements.txt   # uses existing ROCm torch
huggingface-cli login    # accept the facebook/meshflow gate first in a browser
# weights auto-download to ckpt/meshflow on first run
export ASSETFORGE_MESHFLOW=~/meshflow
```
Then `assetforge/scene/retopo.py` works, and `Config.retopo=True` (or a preset) enables it.

## Integration (assetforge/scene/retopo.py)
`retopo(mesh_glb, out_dir, num_verts=2048)` → clean GLB. Wire into `cleanup` when `cfg.retopo`:
Hunyuan dense mesh → MeshFlow (feeds the mesh directly; trimesh.sample internally) → clean triangle
mesh → texture/rig/animate as usual. NOTE: output has **no UVs/texture** (re-texture after) and is
**capped at the vertex budget** (2048/4096) → it's a retopo/decimation-with-better-topology stage,
not detail-preserving; triangles only (no quads).

## Commercial alternative
For a shippable product, MeshFlow's license blocks it. Options: the `weld_glb.py` utility (connectivity
only, no retopo), Blender QuadriFlow/decimate (permissive, lower quality), or a commercially-licensed
retopo service. Revisit if a permissive retopo model appears.
