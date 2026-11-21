"""Milestone-0 fixture: build a synthetic turret with known-correct joint axes.

    blender -b -P make_turret.py -- <out_parts.glb> <out_graph.json>

Three named part objects (base / head / barrel) exported in one GLB, plus a hand-authored
articulation graph.json with the GROUND-TRUTH joints:
  yaw   : base -> head,  revolute about +Z at the base top  (head must spin IN PLACE)
  pitch : head -> barrel, revolute about +X (lateral) at the barrel root (barrel elevates)
Because the truth is known, any authoring/axis error is visually unambiguous.
"""
import bpy, json, sys, math, mathutils

argv = sys.argv[sys.argv.index("--") + 1:]
out_glb, out_graph = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)


def mat(name, col):
    m = bpy.data.materials.new(name); m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = col
    return m


# base: squat cylinder on the ground, top at z=0.6
bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=0.6, location=(0, 0, 0.3))
base = bpy.context.active_object; base.name = "base"
base.data.materials.append(mat("m_base", (0.30, 0.30, 0.35, 1)))

# head: box sitting on the base (so yaw spins it in place about Z)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.95))
head = bpy.context.active_object; head.name = "head"; head.scale = (0.7, 0.95, 0.45)
bpy.ops.object.transform_apply(scale=True)
head.data.materials.append(mat("m_head", (0.55, 0.45, 0.18, 1)))

# barrel: long cylinder pointing +Y, rooted at the head front (pitch about X elevates it)
bpy.ops.mesh.primitive_cylinder_add(radius=0.16, depth=1.6, location=(0, 0.9, 1.0),
                                    rotation=(math.radians(90), 0, 0))
barrel = bpy.context.active_object; barrel.name = "barrel"
bpy.ops.object.transform_apply(rotation=True)
barrel.data.materials.append(mat("m_barrel", (0.18, 0.18, 0.20, 1)))

# export the three named objects in one GLB
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=out_glb, use_selection=True, export_format="GLB")


def cen_bbox(o):
    cs = [o.matrix_world @ mathutils.Vector(v.co) for v in o.data.vertices]
    xs = [c.x for c in cs]; ys = [c.y for c in cs]; zs = [c.z for c in cs]
    return ([sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)],
            [[min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]])


graph = {"kind": "mechanical", "mesh": out_glb, "up_axis": [0, 0, 1], "parts": [], "joints": []}
for o in (base, head, barrel):
    c, b = cen_bbox(o)
    graph["parts"].append({"id": o.name, "object": o.name, "centroid": c, "bbox": b})
graph["joints"] = [
    {"id": "yaw", "parent": "base", "child": "head", "type": "revolute",
     "origin": [0, 0, 0.6], "axis": [0, 0, 1],
     "limits": {"min": -3.14159, "max": 3.14159}, "rest": 0.0,
     "motion": {"preset": "scan", "amp": 1.4, "speed": 1.0}},
    {"id": "pitch", "parent": "head", "child": "barrel", "type": "revolute",
     "origin": [0, 0.2, 1.0], "axis": [1, 0, 0],
     "limits": {"min": -0.2, "max": 1.2}, "rest": 0.0,
     "motion": {"preset": "elevate", "target": 0.8}},
]
json.dump(graph, open(out_graph, "w"), indent=2)
print("WROTE", out_glb, out_graph)
