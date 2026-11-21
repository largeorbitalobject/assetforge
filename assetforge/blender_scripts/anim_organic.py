"""Roles-driven organic walk/idle. Builds the rig AND animates in one process so bone i == joint i
== role index (avoids the glTF round-trip bone-order scramble). Generalized from walk_pro.py.

    blender -b -P anim_organic.py -- <npz> <mesh.glb> <graph.json> <out_glb> <render_dir> [quality]

Reads graph.json["roles"] (pelvis/spine/neck/legs[]/arms[]) + skin.height. Works for any leg/arm
count: legs alternate phase, arms swing opposite. FK (robust, no IK tuning).
"""
import bpy, sys, json, math, numpy as np
from mathutils import Vector, Quaternion

a = sys.argv[sys.argv.index("--") + 1:]
npz_p, mesh_p, graph_p, out_glb, render_dir = a[0], a[1], a[2], a[3], a[4]
quality = a[5] if len(a) > 5 else "draft"

d = np.load(npz_p, allow_pickle=True)
flip = np.array([1, -1, -1], np.float32)
joints = d["joints"].astype(np.float32) * flip
parents = d["parents"].astype(int)
skin = d["skinning_weights"].astype(np.float32)
pc = d["pointcloud"].astype(np.float32) * flip
graph = json.load(open(graph_p))
roles = graph.get("roles", {})
H = float((graph.get("skin") or {}).get("height") or (joints[:, 2].max() - joints[:, 2].min()))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=mesh_p)
obj = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]

# --- build armature: bone j spans joints[parent]->joints[j]; bone j named b{j}, order == joint order ---
arm_data = bpy.data.armatures.new("rig"); arm = bpy.data.objects.new("rig", arm_data)
bpy.context.collection.objects.link(arm); bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = arm_data.edit_bones; ebs = []
for j in range(len(joints)):
    p = parents[j]; b = eb.new(f"b{j}")
    if p == j:
        b.head = Vector(joints[j]); b.tail = Vector(joints[j]) + Vector((0, 0.04, 0))
    else:
        b.head = Vector(joints[p]); b.tail = Vector(joints[j])
    if (b.tail - b.head).length < 1e-4:
        b.tail = b.head + Vector((0, 0.03, 0))
    ebs.append(b)
for j in range(len(joints)):
    if parents[j] != j:
        ebs[j].parent = ebs[parents[j]]
bpy.ops.object.mode_set(mode="OBJECT")

# --- skin transfer: mesh vert -> nearest pointcloud -> top-4 weights (chunked NN) ---
verts = np.array([(obj.matrix_world @ v.co)[:] for v in obj.data.vertices], np.float32)
from scipy.spatial import cKDTree
W = skin[cKDTree(pc.astype(np.float32)).query(verts.astype(np.float32), k=1)[1]]
order = np.argsort(-W, axis=1)[:, :4]
vgs = [obj.vertex_groups.new(name=f"b{j}") for j in range(len(joints))]
for vi in range(len(verts)):
    sel = order[vi]; s = W[vi][sel].sum() + 1e-9
    for j in sel:
        w = float(W[vi][j] / s)
        if w > 1e-3:
            vgs[j].add([vi], w, "REPLACE")
mod = obj.modifiers.new("arm", "ARMATURE"); mod.object = arm; obj.parent = arm

# --- pose setup ---
bones = list(arm.data.bones); Bn = {b.name: b for b in bones}
bpy.context.view_layer.objects.active = arm; bpy.ops.object.mode_set(mode="POSE")
for pb in arm.pose.bones:
    pb.rotation_mode = "QUATERNION"
WX, WY, WZ = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))

def setrot(idx, axis_angles, f):
    if idx is None or idx < 0 or idx >= len(bones):
        return
    nm = bones[idx].name; pb = arm.pose.bones[nm]
    M = Bn[nm].matrix_local.to_3x3().inverted()
    q = Quaternion()
    for axis, ang in axis_angles:
        if abs(ang) < 1e-6:
            continue
        q = Quaternion((M @ axis).normalized(), ang) @ q
    pb.rotation_quaternion = q; pb.keyframe_insert("rotation_quaternion", frame=f)

P = dict(HIP=0.42, KNEE_SW=0.45, HOCK_SW=0.40, HOCK_PUSH=0.20, TOE=0.35,
         BOB=0.038, PYAW=0.20, PROLL=0.11, PPITCH=0.02, SYAW=0.12, SBEND=0.04,
         NECK_CY=0.10, HEAD_LV=0.06, ARM_SW=0.55, EL_BEND=0.55, EL_DRAG=0.18)
swingwin = lambda phi: max(0.0, math.sin(phi - math.pi))
pushwin = lambda phi: (max(0.0, math.sin(phi)) ** 6 if phi < math.pi else 0.0)

def do_leg(L, phi, f):
    hip = P["HIP"] * math.cos(phi)
    knee = -(P["KNEE_SW"] * swingwin(phi))
    hock = (P["HOCK_SW"] * swingwin(phi) - P["HOCK_PUSH"] * pushwin(phi))
    toe = -(P["TOE"] * pushwin(phi)) + 0.15 * swingwin(phi)
    setrot(L.get("hip"), [(WX, hip)], f)
    setrot(L.get("knee"), [(WX, knee)], f)
    setrot(L.get("ankle"), [(WX, hock)], f)   # ankle == hock
    setrot(L.get("toe"), [(WX, toe)], f)

def do_arm(A, phi, f):
    sw = P["ARM_SW"] * math.cos(phi)
    setrot(A.get("shoulder"), [(WX, sw)], f)
    setrot(A.get("elbow"), [(WX, -(P["EL_BEND"] * 0.5 + P["EL_BEND"] * 0.5 * max(0, math.cos(phi - P["EL_DRAG"]))))], f)

pelvis = roles.get("pelvis", 0)
spine = [s for s in roles.get("spine", []) if s != pelvis]
neck = roles.get("neck")
legs = roles.get("legs", [])
arms = roles.get("arms", [])

F = 40; sc = bpy.context.scene; sc.frame_start = 1; sc.frame_end = F; sc.render.fps = 24
root = arm.pose.bones[bones[pelvis].name] if pelvis < len(bones) else arm.pose.bones[bones[0].name]
for f in range(1, F + 1):
    t = 2 * math.pi * (f - 1) / F
    for i, L in enumerate(legs):
        do_leg(L, t + math.pi * (i % 2), f)
    for i, A in enumerate(arms):
        do_arm(A, t + math.pi * (i % 2) + math.pi, f)   # opposite to same-parity leg
    setrot(pelvis, [(WZ, P["PYAW"] * math.cos(t)), (WY, P["PROLL"] * math.sin(t)),
                    (WX, P["PPITCH"] * math.cos(2 * t))], f)
    root.location = (0, 0, -P["BOB"] * math.cos(2 * t)); root.keyframe_insert("location", frame=f)
    for k, si in enumerate(spine):
        w = (k + 1) / max(1, len(spine))
        setrot(si, [(WZ, -P["SYAW"] * w * math.cos(t)), (WY, P["SBEND"] * w * math.sin(t))], f)
    if neck is not None:
        setrot(neck, [(WZ, P["NECK_CY"] * math.cos(t)), (WX, P["HEAD_LV"] * math.cos(2 * t))], f)
if arm.animation_data and arm.animation_data.action:
    for fc in arm.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"

# --- export animated GLB ---
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=out_glb, use_selection=False, export_format="GLB")
print("ANIM_GLB", out_glb)

# --- render montage of sampled frames ---
import os
os.makedirs(render_dir, exist_ok=True)
for p in obj.data.polygons:
    p.use_smooth = True
allv = np.array([(obj.matrix_world @ v.co)[:] for v in obj.data.vertices])
cen = Vector(allv.mean(0)); R = float(np.ptp(allv, 0).max()) * 0.5
try: sc.render.engine = "BLENDER_EEVEE_NEXT"
except Exception: sc.render.engine = "CYCLES"
sc.world = bpy.data.worlds.new("W"); sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
for en, rot in [(3, (math.radians(55), 0, math.radians(40))), (1.4, (math.radians(60), 0, math.radians(-110)))]:
    s = bpy.data.objects.new("L", bpy.data.lights.new("L", "SUN")); sc.collection.objects.link(s)
    s.data.energy = en; s.rotation_euler = rot
camd = bpy.data.cameras.new("c"); cam = bpy.data.objects.new("c", camd); sc.collection.objects.link(cam); sc.camera = cam
ang = math.radians(74); dist = R * 3.7
cam.location = cen + Vector((dist * math.sin(ang), -dist * math.cos(ang), R * 0.05))
cam.rotation_euler = (cen - cam.location).to_track_quat("-Z", "Y").to_euler()
sc.render.resolution_x = sc.render.resolution_y = 360; sc.render.film_transparent = True
for f in [1 + round(i * (F - 1) / 5) for i in range(6)]:
    sc.frame_set(f); sc.render.filepath = os.path.join(render_dir, f"a{f:03d}.png")
    bpy.ops.render.render(write_still=True)
print("ANIM_DONE")
