"""On-shelf hybrid texturing fallback (only needed if a driver regression reintroduces ROCm #5981).
See docs/TEXTURING-ROCM.md for the cr.rasterize replacement contract. Building blocks:
bake_maps.py (UV-space atlas), gather_bake.py (view-space projection + z-buffer)."""
