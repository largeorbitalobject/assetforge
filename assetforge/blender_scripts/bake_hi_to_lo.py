"""[WIP — selected-to-active bake currently produces black; needs debugging] Bake a high-poly TEXTURED mesh's albedo (+ tangent normal map) onto a low-poly mesh's UVs.
The game-asset way: keep dense-mesh texture quality on an animation-friendly low-poly mesh.

    blender -b -P bake_hi_to_lo.py -- <hi_textured.glb> <lo.glb> <out.glb> [tex_size]
"""
import bpy, sys, os
a = sys.argv[sys.argv.index("--") + 1:]
hi, lo, out = a[0], a[1], a[2]
TS = int(a[3]) if len(a) > 3 else 2048
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 4

def imp(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    objs = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1: bpy.ops.object.join()
    return bpy.context.view_layer.objects.active

src = imp(hi); tgt = imp(lo)

# smart-UV unwrap the low-poly target
bpy.ops.object.select_all(action="DESELECT"); tgt.select_set(True); bpy.context.view_layer.objects.active = tgt
bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.005)
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode="OBJECT")

# target material with a fresh image to bake into
mat = bpy.data.materials.new("baked"); mat.use_nodes = True
img = bpy.data.images.new("albedo", TS, TS)
tex = mat.node_tree.nodes.new("ShaderNodeTexImage"); tex.image = img
mat.node_tree.nodes.active = tex
mat.node_tree.links.new(tex.outputs["Color"], mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
tgt.data.materials.clear(); tgt.data.materials.append(mat)

# selected-to-active diffuse bake (src -> tgt)
bpy.ops.object.select_all(action="DESELECT")
src.select_set(True); tgt.select_set(True); bpy.context.view_layer.objects.active = tgt
sc.render.bake.use_selected_to_active = True
sc.render.bake.cage_extrusion = float(max(tgt.dimensions)) * 0.20
sc.render.bake.max_ray_distance = float(max(tgt.dimensions)) * 0.30
sc.render.bake.use_pass_direct = False; sc.render.bake.use_pass_indirect = False; sc.render.bake.use_pass_color = True
try:
    bpy.ops.object.bake(type="DIFFUSE")
    print("BAKE diffuse OK")
except Exception as e:
    print("BAKE FAIL", e)
# pack + export
img.pack()
bpy.ops.object.select_all(action="DESELECT"); tgt.select_set(True); bpy.context.view_layer.objects.active = tgt
bpy.ops.export_scene.gltf(filepath=out, use_selection=True, export_format="GLB")
print("SAVED", out)
