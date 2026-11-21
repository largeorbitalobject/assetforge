"""Render a parts-segmented mesh with each part a distinct flat color, from N views, for VLM
visual reasoning about articulation.

    blender -b -P render_parts_colored.py -- <mesh.glb> <out_dir>

Writes out_dir/{view}.png and out_dir/legend.json {part_name: [r,g,b]}. If the GLB has multiple
named objects (parts), each gets its own color; otherwise the whole mesh is one color.
"""
import bpy, sys, json, math, os, mathutils

a = sys.argv[sys.argv.index("--") + 1:]
glb, out_dir = a[0], a[1]
os.makedirs(out_dir, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]

PALETTE = [(0.90,0.20,0.20),(0.20,0.50,0.95),(0.20,0.80,0.30),(0.95,0.75,0.15),
           (0.70,0.30,0.85),(0.20,0.85,0.85),(0.95,0.45,0.15),(0.55,0.55,0.55)]
legend = {}
for i, o in enumerate(meshes):
    col = PALETTE[i % len(PALETTE)]
    m = bpy.data.materials.new(f"c{i}"); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (*col, 1)
    out = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(em.outputs[0], out.inputs["Surface"])
    o.data.materials.clear(); o.data.materials.append(m)
    legend[o.name] = [round(c, 2) for c in col]
json.dump(legend, open(os.path.join(out_dir, "legend.json"), "w"))

import numpy as np
V = np.vstack([np.array([(o.matrix_world @ v.co)[:] for v in o.data.vertices]) for o in meshes])
center = mathutils.Vector(V.mean(0)); R = float(np.ptp(V, 0).max()) * 0.5 or 1
sc = bpy.context.scene
try: sc.render.engine = "BLENDER_EEVEE_NEXT"
except Exception: sc.render.engine = "CYCLES"
sc.view_settings.view_transform = "Standard"
sc.render.resolution_x = sc.render.resolution_y = 512; sc.render.film_transparent = True
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[1].default_value = 0.5
camd = bpy.data.cameras.new("c"); cam = bpy.data.objects.new("c", camd); sc.collection.objects.link(cam); sc.camera = cam
for name, az, el in [("front", 0, 5), ("side", 90, 5), ("threeq", 40, 20)]:
    ar, er = math.radians(az), math.radians(el); d = R * 3.6
    cam.location = center + mathutils.Vector((d*math.cos(er)*math.sin(ar), -d*math.cos(er)*math.cos(ar), d*math.sin(er)))
    cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = os.path.join(out_dir, f"{name}.png"); bpy.ops.render.render(write_still=True)
print("PARTS_COLORED", json.dumps(legend))
