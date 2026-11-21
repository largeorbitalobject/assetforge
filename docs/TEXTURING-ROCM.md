# Texturing on gfx1151 / ROCm — the native Hunyuan3D-2.1 painter

**Status (2026-06-19): WORKING end-to-end. No corruption, no rebuild, no hybrid needed.**

For a long time the Hunyuan3D-2.1 native PBR painter (`hy3dpaint`) was treated as an AMD
dead-end — ROCm issue [#5981](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/issues) reports
corrupt `custom_rasterizer` texture output on gfx12, and the prebuilt `.so` wouldn't even import.
A focused investigation/repair pass proved **both problems are absent or trivially fixable on this
box.**

## Root cause #1 — the import error was import ORDER, not an ABI break

The kernel `.so` reported `libc10.so: cannot open shared object file` (then
`librocm-openblas.so.0`). Those errors **only occur when the kernel is imported with no torch in
the process.** torch 2.8.0+rocm7.12's Python-side `rocm_sdk` loader preloads the venv's
`_rocm_sdk_core` + `_rocm_sdk_libraries_gfx1151` libs by absolute path (RTLD_GLOBAL), satisfying
the kernel's `DT_NEEDED`. So the fix is simply:

```python
import torch                 # MUST come first
import custom_rasterizer     # now resolves cleanly
```

This ordering is naturally satisfied in the real pipeline (`custom_rasterizer/__init__.py` and
`MeshRender`/ComfyUI all import torch first).

> **Do NOT "fix" it with LD_LIBRARY_PATH.** Adding the `_rocm_sdk` dirs or system
> `/opt/rocm-7.1.0` lets the older system ROCm (7.1.0) **shadow** the venv's ROCm 7.12 libs,
> producing real `ncclCommDump` / `hsa_ext_image_create_v2` undefined-symbol crashes. Keep
> `LD_LIBRARY_PATH` clean. `assetforge.rocm_env.gpu_env()` deliberately never touches it.

## Root cause #2 — ROCm #5981 is not present on this torch nightly

This venv runs `torch 2.8.0+rocm7.12.0a20260204`, a TheRock gfx12 nightly from the family AMD
confirmed fixes #5981 (the `...a20260218` build). The depth-test/atomicMin-pack + barycentric path
that #5981 corrupts is **pristine** here. Smoke test on the 234k-vert minotaur: barycentric
row-sums exactly 1.0 on covered pixels, crisp silhouettes, no wrong-face bleed, no out-of-silhouette
noise. An independent visual judge confirmed corruption = no, quality = good across 14 outputs.
(Every historical "corrupt on AMD" report was on **stable** ROCm 6.4–7.2.)

## Reproduction recipe

```bash
# --- REQUIRED GPU env (assetforge.rocm_env.GPU_ENV) ---
export HSA_OVERRIDE_GFX_VERSION=11.5.1 HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:8192
export HSA_ENABLE_SDMA=0 HSA_USE_SVM=0 HSA_FORCE_FINE_GRAIN_PCIE=1 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
# keep LD_LIBRARY_PATH CLEAN.

VENV=/home/orbital/ComfyUI/venv/bin/python

# 1. verify import (torch first)
$VENV -c 'import torch, custom_rasterizer as cr; print(cr.rasterize)'

# 2. one-time weights (~11.5 GB) into hy3dpaint/ckpt/ + HF cache
$VENV -c "from huggingface_hub import snapshot_download as d; d('tencent/Hunyuan3D-2.1', allow_patterns=['hunyuan3d-paintpbr-v2-1/*']); d('facebook/dinov2-giant')"
curl -sL -o /home/orbital/ComfyUI/app/custom_nodes/ComfyUI-Hunyuan3d-2-1.broken/hy3dpaint/ckpt/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# 3. paint any mesh (generalized driver)
PYTHONPATH=/home/orbital/projects/assetforge $VENV -m assetforge.texturing.paint \
  --mesh in.glb --image ref.png --out /tmp/out --texture-size 1024
```

## Integration notes (baked into `texturing/paint.py`)

- `textureGenPipeline.py` is the **ComfyUI-node** variant (relative imports + `import folder_paths`,
  bake code commented out). Import it as a package with `ComfyUI/app` and the node dir on
  `sys.path`; drive the multiview gen + `bake_from_multiview` manually rather than running standalone.
- **UV-unwrap** (`mesh_uv_wrap`) is CPU-bound (~270s for 234k verts) — cached to `unwrap.npz`.
- **Transient HIP fault** on long op-chains: pull the baked atlas + mesh arrays to CPU *promptly*
  and rebuild the GLB from numpy (replicating `MeshRender.get_mesh`'s inverse coord transform) instead
  of letting `save_mesh`'s trailing `.cpu()` chain run. Avoid MeshRender's MR-channel bake.
- **Keep texture/render resolution SQUARE** — `back_project:1264` uses `img_y*resolution[0]+img_x`
  (should be `*resolution[1]`); harmless only when H == W.

## To re-enable the ComfyUI node

The install dir is still suffixed `.broken`. To use it from the ComfyUI GUI:
`mv .../ComfyUI-Hunyuan3d-2-1.broken .../ComfyUI-Hunyuan3d-2-1` and restart ComfyUI.
`custom_rasterizer` is already installed in the venv; import order is auto-satisfied by the node.
(AssetForge drives the painter directly via the venv, so renaming is optional.)

## On-shelf hybrid fallback (only if a future driver reintroduces #5981)

The native dependency reduces to **one** primitive: `cr.rasterize` (`cr.interpolate` is pure torch).
Replace only that symbol — `texturing/hybrid/` holds the building blocks (`bake_maps.py` for
UV-space, `gather_bake.py` for view-space + z-buffer). Contract a replacement must match exactly:

```
rasterize(pos[1,V,4] f32 clip-space, tri[F,3] i32, resolution=(H,W))
  -> (findices[H,W] i32  # 1-BASED face id, 0 = background (load-bearing: downstream uses >0 as mask)
      barycentric[H,W,3] f32)  # perspective-correct, sum=1 on covered px, 0 on bg
NDC->pixel:  sx=(x/w*0.5+0.5)*(W-1)+0.5 ; sy=(0.5+0.5*y/w)*(H-1)+0.5  (NO y-flip) ; sz=z/w*0.49999+0.5
             sample at pixel-center+0.5 ; nearest-z wins.
```

Must handle **both** regimes with the same triangle rasterizer: (A) view-space (real MVP clip coords)
and (B) UV-space (`pos=[u*2-1, v*2-1, 0, 1]` at `resolution=texture_size`). Recommended: a ~150–250
line pure-torch software rasterizer (`render_shim.py`), cross-checked vs the native `cr.rasterize`
(mask IoU, bary row-sum, face-id histogram) before trusting it. The full spec is in the workflow
result and `MEMORY.md`.
