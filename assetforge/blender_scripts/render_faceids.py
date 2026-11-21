"""Render a mesh from N views as (beauty, face-id) pairs + camera info, for multiview part lifting.

    blender -b -P render_faceids.py -- <mesh.glb> <out_dir> [n_az]

face-id buffer: each face gets a unique RGB (id encoded base-256) via a FACE color attribute +
flat emission, so a rendered pixel decodes to its face index. Beauty render uses the GLB's texture.
Writes out_dir/{view}_beauty.png, {view}_fid.png, and cams.json (per-view ortho params).
"""
import bpy, sys, json, math, os, struct
import mathutils

a = sys.argv[sys.argv.index("--") + 1:]
glb, out_dir = a[0], a[1]
os.makedirs(out_dir, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes: o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1: bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
me = obj.data
nF = len(me.polygons)

# unique color per face (FACE-domain byte color attribute)
ca = me.color_attributes.new(name="fid", type="BYTE_COLOR", domain="CORNER")
import numpy as np
cols = np.zeros((len(me.loops), 4), np.float32)
for p in me.polygons:
    fid = p.index + 1   # 1-based; 0 = background
    r = (fid & 255) / 255.0; g = ((fid >> 8) & 255) / 255.0; b = ((fid >> 16) & 255) / 255.0
    for li in p.loop_indices:
        cols[li] = (r, g, b, 1.0)
ca.data.foreach_set("color", cols.reshape(-1))

# face-id material (flat emission of the color attribute)
fidmat = bpy.data.materials.new("fidmat"); fidmat.use_nodes = True
nt = fidmat.node_tree; nt.nodes.clear()
attr = nt.nodes.new("ShaderNodeVertexColor"); attr.layer_name = "fid"
em = nt.nodes.new("ShaderNodeEmission"); outn = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(attr.outputs["Color"], em.inputs["Color"]); nt.links.new(em.outputs["Emission"], outn.inputs["Surface"])

beauty_mats = list(me.materials)  # keep original textured material(s)

sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.samples = 1
sc.view_settings.view_transform = "Standard"   # no tone-mapping (exact face-id colors)
sc.render.image_settings.color_mode = "RGB"; sc.render.resolution_x = sc.render.resolution_y = 768
sc.render.film_transparent = True
sc.render.filter_size = 0.01   # crisp face-id colors (no edge blending)

# bounds + ortho camera
import numpy as np
V = np.array([(obj.matrix_world @ v.co)[:] for v in me.vertices])
center = mathutils.Vector(V.mean(0)); R = float(np.ptp(V, 0).max()) * 0.5
camd = bpy.data.cameras.new("c"); camd.type = "ORTHO"; camd.ortho_scale = R * 2.3
cam = bpy.data.objects.new("c", camd); sc.collection.objects.link(cam); sc.camera = cam
sun = bpy.data.objects.new("s", bpy.data.lights.new("s", "SUN")); sc.collection.objects.link(sun); sun.data.energy = 3

n_az = int(a[2]) if len(a) > 2 else 6
views = [("front", 0, 0), ("right", 90, 0), ("back", 180, 0), ("left", 270, 0), ("top", 0, 85), ("bottom", 0, -85)][:n_az]
cams = {}
for name, az, el in views:
    ar = math.radians(az); er = math.radians(el); dist = R * 4
    cam.location = center + mathutils.Vector((dist*math.cos(er)*math.sin(ar), -dist*math.cos(er)*math.cos(ar), dist*math.sin(er)))
    cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
    # face-id pass
    for s in me.materials: pass
    me.materials.clear(); me.materials.append(fidmat)
    for p in me.polygons: p.material_index = 0
    sc.render.image_settings.file_format = "OPEN_EXR"; sc.render.image_settings.color_depth = "32"
    sc.render.filepath = os.path.join(out_dir, f"{name}_fid"); bpy.ops.render.render(write_still=True)
    # beauty pass
    me.materials.clear()
    for m in beauty_mats: me.materials.append(m)
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = os.path.join(out_dir, f"{name}_beauty.png"); bpy.ops.render.render(write_still=True)
    cams[name] = {"az": az, "el": el}
json.dump({"n_faces": nF, "views": cams}, open(os.path.join(out_dir, "cams.json"), "w"))
print("FACEIDS", nF, "views", len(views))
