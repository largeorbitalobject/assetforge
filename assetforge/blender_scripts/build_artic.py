"""Author an articulation graph into a constrained, animatable rig (zero CUDA, pure Blender).

    blender -b -P build_artic.py -- <parts.glb> <graph.json> <out_rest.glb> <out.blend>

Contract: each graph part has an `object` = a named mesh object in parts.glb. For each joint
(parent->child) we create an Empty "J_<id>" at the joint origin, parent it into the joint tree,
parent the child part-object under it, and LOCK all rotation channels except the joint's axis so a
revolute has exactly 1 rotational DOF (prismatic: 1 translational DOF). The skinned-skeleton organic
case is handled by build_rig.py; this is the rigid/mechanical authoring path.

Axis handling (Milestone 0): joint axes are mapped to the dominant world euler channel (X/Y/Z).
Arbitrary (non-axis-aligned) joints are a later enhancement (orient the empty so local Z == axis).
"""
import bpy, sys, json, mathutils

argv = sys.argv[sys.argv.index("--") + 1:]
parts_glb, graph_path, out_glb, out_blend = argv[0], argv[1], argv[2], argv[3]
graph = json.load(open(graph_path))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=parts_glb)

# map part id -> mesh object (gltf may suffix names; match by prefix)
objs = {o.name: o for o in bpy.data.objects if o.type == "MESH"}
def find_obj(name):
    if name in objs:
        return objs[name]
    for k, o in objs.items():
        if k == name or k.startswith(name):
            return o
    raise KeyError(f"part object {name!r} not found in {list(objs)}")

AXIS_CHANNEL = {0: "X", 1: "Y", 2: "Z"}
def dom_channel(axis):
    a = [abs(x) for x in axis]
    return a.index(max(a))  # 0/1/2

# joint lookups
joints = {j["id"]: j for j in graph["joints"]}
child_part_to_joint = {j["child"]: j for j in graph["joints"]}  # which joint controls a part

def empty_name(jid):
    return f"J_{jid}"

# create empties + parent the joint tree, child parts, and set DOF locks
# process in an order where a joint's parent-joint (if any) already exists
done = set()
def ensure_joint(jid):
    if jid in done:
        return
    j = joints[jid]
    parent_part = j["parent"]
    # if the parent part is itself controlled by a joint, build that first
    pj = child_part_to_joint.get(parent_part)
    if pj is not None:
        ensure_joint(pj["id"])

    e = bpy.data.objects.new(empty_name(jid), None)
    e.empty_display_type = "ARROWS"; e.empty_display_size = 0.4
    e.location = mathutils.Vector(j["origin"])
    bpy.context.collection.objects.link(e)

    # parent the empty under the controlling empty of its parent part (else world)
    if pj is not None:
        pe = bpy.data.objects[empty_name(pj["id"])]
        e.parent = pe
        e.matrix_parent_inverse = pe.matrix_world.inverted()

    # parent the child part-object under this empty, keeping its world transform
    child = find_obj(joints[jid]["child"])
    child.parent = e
    child.matrix_parent_inverse = e.matrix_world.inverted()

    # DOF lock: only the joint axis channel is free
    ch = dom_channel(j["axis"])
    e.rotation_mode = "XYZ"
    if j["type"] == "revolute" or j["type"] == "ball":
        if j["type"] == "revolute":
            e.lock_rotation = [i != ch for i in range(3)]
            e.lock_location = [True, True, True]
    elif j["type"] == "prismatic":
        e.lock_location = [i != ch for i in range(3)]
        e.lock_rotation = [True, True, True]
    elif j["type"] == "fixed":
        e.lock_rotation = [True, True, True]; e.lock_location = [True, True, True]
    # store metadata for the driver
    e["joint_id"] = jid; e["joint_type"] = j["type"]; e["free_channel"] = ch
    e["lim_min"] = j.get("limits", {}).get("min", -3.14159) if j.get("limits") else -3.14159
    e["lim_max"] = j.get("limits", {}).get("max", 3.14159) if j.get("limits") else 3.14159
    done.add(jid)
    print(f"JOINT {jid}: {j['type']} axis={j['axis']} (channel {AXIS_CHANNEL[ch]}) origin={j['origin']}")

for jid in list(joints):
    ensure_joint(jid)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=out_glb, use_selection=True, export_format="GLB")
bpy.ops.wm.save_as_mainfile(filepath=out_blend)
print("AUTHORED", out_glb, out_blend)
