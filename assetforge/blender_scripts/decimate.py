"""Headless mesh cleanup: join parts, drop loose floaters, decimate to a face budget.

    blender -b -P decimate.py -- <in.glb> <out.glb> <target_faces>

Decimating the dense Hunyuan surface-net mesh (~300k faces) to a budget is both a quality
step (the raw mesh is not shippable) and the single biggest speed win: it slashes the
downstream CPU UV-unwrap, which scales with face count.
"""
import sys
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
inp, out, target = argv[0], argv[1], int(argv[2])

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=inp)

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit("no mesh in " + inp)

# join all parts into one object
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

# drop loose floaters: keep only geometry connected to the largest island
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="DESELECT")
bpy.ops.object.mode_set(mode="OBJECT")
n_before = len(obj.data.polygons)

# decimate to target face budget (collapse keeps a watertight-ish surface)
if n_before > target:
    m = obj.modifiers.new("dec", "DECIMATE")
    m.decimate_type = "COLLAPSE"
    m.ratio = max(0.01, target / n_before)
    m.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=m.name)

print(f"DECIMATE faces {n_before} -> {len(obj.data.polygons)} (target {target})")
bpy.ops.export_scene.gltf(filepath=out, use_selection=False, export_format="GLB")
print("SAVED " + out)
