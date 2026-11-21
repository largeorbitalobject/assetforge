"""Assemble a compositional scene in Blender (CPU): skydome + ground/terrain + placed hero GLBs.

    blender -b -P assemble_scene.py -- <scene_spec.json> <out_glb> <render_png>

scene_spec.json:
{
  "skydome": "equirect_or_backdrop.png" | null,
  "ground":  {"size": 20.0, "heightmap": "depth.png"|null, "subdiv": 200, "z": 0.0},
  "heroes":  [{"glb": "hero.glb", "pos": [x,y,z], "scale": 1.0, "yaw": 0.0} ...]
}
Backbone of Track B: every hero is an independent editable GLB placed in the scene. Pure CPU.
"""
import bpy, sys, json, math, os, mathutils

a = sys.argv[sys.argv.index("--") + 1:]
spec = json.load(open(a[0])); out_glb, render_png = a[1], a[2]
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


def img_node_mat(name, path, emission=False):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; bsdf = nt.nodes.get("Principled BSDF")
    tex = nt.nodes.new("ShaderNodeTexImage")
    try: tex.image = bpy.data.images.load(path)
    except Exception as e: print("img load fail", e); return m
    if emission:
        em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = 1.6
        nt.links.new(tex.outputs["Color"], em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], nt.nodes["Material Output"].inputs["Surface"])
    else:
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return m


# --- skydome: large inverted sphere with emission backdrop ---
if spec.get("skydome") and os.path.exists(spec["skydome"]):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=60, segments=64, ring_count=32)
    dome = bpy.context.active_object; dome.name = "skydome"
    # flip normals inward + spherical UVs
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.normals_make_consistent(inside=True)
    bpy.ops.uv.sphere_project() if False else None
    bpy.ops.mesh.uv_texture_add() if False else None
    bpy.ops.object.mode_set(mode="OBJECT")
    dome.data.materials.append(img_node_mat("skymat", spec["skydome"], emission=True))
    dome.visible_shadow = False
    try: dome.visible_diffuse = True
    except Exception: pass

# --- backdrop mesh (MoGe-derived scene geometry) ---
if spec.get("backdrop") and os.path.exists(spec["backdrop"]):
    bpy.ops.import_scene.gltf(filepath=spec["backdrop"])
    print("imported backdrop")

# --- ground / terrain ---
g = spec.get("ground") or {}
size = g.get("size", 20.0)
bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, g.get("z", 0.0)))
ground = bpy.context.active_object; ground.name = "ground"
if g.get("heightmap") and os.path.exists(g["heightmap"]):
    sub = ground.modifiers.new("sub", "SUBSURF"); sub.subdivision_type = "SIMPLE"
    sub.levels = sub.render_levels = 6
    bpy.ops.object.modifier_apply(modifier=sub.name)
    disp = ground.modifiers.new("disp", "DISPLACE")
    tex = bpy.data.textures.new("ht", "IMAGE"); tex.image = bpy.data.images.load(g["heightmap"])
    disp.texture = tex; disp.strength = g.get("relief", size * 0.15)
    bpy.ops.object.modifier_apply(modifier=disp.name)
gm = bpy.data.materials.new("groundm"); gm.use_nodes = True
gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.4, 0.42, 0.45, 1)
ground.data.materials.append(gm)

# --- heroes ---
placed = []
for h in spec.get("heroes", []):
    if not os.path.exists(h["glb"]):
        print("missing hero", h["glb"]); continue
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=h["glb"])
    new = [o for o in bpy.data.objects if o not in before and o.parent is None]
    s = float(h.get("scale", 1.0)); pos = h.get("pos", [0, 0, 0]); yaw = math.radians(h.get("yaw", 0.0))
    for o in new:
        o.scale = (o.scale[0]*s, o.scale[1]*s, o.scale[2]*s)
        o.location = (o.location[0]+pos[0], o.location[1]+pos[1], o.location[2]+pos[2])
        o.rotation_euler = (o.rotation_euler[0], o.rotation_euler[1], o.rotation_euler[2] + yaw)
        placed.append(o)
print(f"placed {len(placed)} hero objects")

# --- export ---
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=out_glb, use_selection=False, export_format="GLB")
print("SCENE_GLB", out_glb)

# --- render preview ---
# auto-frame the camera on all non-skydome content
content = [o for o in bpy.data.objects if o.type=="MESH" and o.name!="skydome"]
mn=mathutils.Vector((1e9,)*3); mx=mathutils.Vector((-1e9,)*3)
for o in content:
    for v in o.bound_box:
        w=o.matrix_world@mathutils.Vector(v)
        for i in range(3): mn[i]=min(mn[i],w[i]); mx[i]=max(mx[i],w[i])
ctr=(mn+mx)/2; rad=max((mx-mn))*0.5 or size
camd = bpy.data.cameras.new("c"); cam = bpy.data.objects.new("c", camd); sc.collection.objects.link(cam); sc.camera = cam
cam.location = ctr + mathutils.Vector((rad*1.6, -rad*2.0, rad*1.1))
cam.rotation_euler = (ctr - cam.location).to_track_quat('-Z', 'Y').to_euler()
for en, rot in [(4.0,(math.radians(50),0,math.radians(35))),
                (2.5,(math.radians(60),0,math.radians(-110))),
                (2.0,(math.radians(70),0,math.radians(170)))]:
    L = bpy.data.objects.new("L", bpy.data.lights.new("L", "SUN")); sc.collection.objects.link(L)
    L.data.energy = en; L.rotation_euler = rot
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
try: sc.render.engine = "BLENDER_EEVEE_NEXT"
except Exception: sc.render.engine = "CYCLES"
sc.render.resolution_x = 640; sc.render.resolution_y = 420; sc.render.filepath = render_png
bpy.ops.render.render(write_still=True); print("SCENE_RENDER", render_png)
