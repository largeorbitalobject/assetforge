"""Clean QUAD retopology via Blender QuadriFlow — even, deformation-friendly topology at a tunable
face budget (animation-grade; keeps far more detail than MeshFlow's ultra-low output).

    blender -b -P quadriflow_remesh.py -- <in.glb> <out.glb> <target_faces>
"""
import bpy, sys
a = sys.argv[sys.argv.index("--") + 1:]
inp, out, target = a[0], a[1], int(a[2])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=inp)
ms = [o for o in bpy.data.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in ms: o.select_set(True)
bpy.context.view_layer.objects.active = ms[0]
if len(ms) > 1: bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
# weld first so QuadriFlow sees a connected surface (texture-export soup otherwise)
bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.remove_doubles(threshold=(max(obj.dimensions) or 1.0) * 5e-4)
bpy.ops.object.mode_set(mode="OBJECT")
n0 = len(obj.data.polygons)
bpy.ops.object.quadriflow_remesh(target_faces=target, use_preserve_sharp=False,
                                 use_preserve_boundary=False, smooth_normals=True)
print(f"QUADRIFLOW faces {n0} -> {len(obj.data.polygons)} (target {target})")
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=out, use_selection=False, export_format="GLB")
print("SAVED", out)
