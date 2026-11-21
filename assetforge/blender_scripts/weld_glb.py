"""Weld a GLB (merge-by-distance) to restore connectivity from the unwelded texture-export soup,
keeping per-loop UVs + the embedded texture. Shrinks files + gives downstream tools real topology.

    blender -b -P weld_glb.py -- <in.glb> <out.glb>
"""
import bpy, sys
a = sys.argv[sys.argv.index("--") + 1:]
inp, out = a[0], a[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=inp)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
n0 = sum(len(o.data.vertices) for o in meshes)
for o in meshes:
    bpy.ops.object.select_all(action="DESELECT"); o.select_set(True)
    bpy.context.view_layer.objects.active = o
    dim = max(o.dimensions) or 1.0
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=dim * 5e-4)   # merge coincident verts; UVs are per-loop -> preserved
    bpy.ops.object.mode_set(mode="OBJECT")
n1 = sum(len(o.data.vertices) for o in meshes)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=out, use_selection=False, export_format="GLB")
print(f"WELD verts {n0} -> {n1}")
