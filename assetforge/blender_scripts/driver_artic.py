"""Procedurally animate an authored articulation rig (zero CUDA, pure Blender).

    blender <rig.blend> -b -P driver_artic.py -- <graph.json> <out_anim.glb> <render_dir> [nframes]

For each joint, keyframe its single free DOF per a motion preset (read from graph or defaulted):
  scan    : angle = amp * sin(2*pi*speed*t), clamped to limits           (revolute)
  elevate : ramp rest->target over first third, then hold                (revolute/prismatic)
  open    : eased rest->max                                              (revolute/prismatic)
Exports a GLB with baked keyframes and renders a montage of sampled frames. Also prints a
quantitative self-check (head centroid stationarity, barrel tip rise) so correctness is provable
without an interactive viewer.
"""
import bpy, sys, json, math, mathutils

argv = sys.argv[sys.argv.index("--") + 1:]
graph_path, out_glb, render_dir = argv[0], argv[1], argv[2]
NF = int(argv[3]) if len(argv) > 3 else 48
graph = json.load(open(graph_path))
joints = {j["id"]: j for j in graph["joints"]}

empties = {o["joint_id"]: o for o in bpy.data.objects if o.type == "EMPTY" and "joint_id" in o.keys()}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def value(j, t):
    """Joint DOF value at normalized time t in [0,1]."""
    m = j.get("motion", {}) or {}
    preset = m.get("preset", "scan")
    lim = j.get("limits") or {"min": -3.14159, "max": 3.14159}
    if preset == "scan":
        amp = m.get("amp", 1.2); speed = m.get("speed", 1.0)
        return clamp(amp * math.sin(2 * math.pi * speed * t), lim["min"], lim["max"])
    if preset == "elevate":
        target = m.get("target", 0.6 * lim["max"])
        return clamp(target * min(1.0, t / 0.33), lim["min"], lim["max"])
    if preset == "open":
        e = (1 - math.cos(math.pi * min(1.0, t))) / 2  # ease in-out
        return clamp(lim["min"] + e * (lim["max"] - lim["min"]), lim["min"], lim["max"])
    return 0.0


sc = bpy.context.scene
sc.frame_start = 1; sc.frame_end = NF

# keyframe each joint's free channel
for jid, e in empties.items():
    ch = int(e["free_channel"]); jtype = e["joint_type"]
    j = joints[jid]
    for f in range(1, NF + 1):
        t = (f - 1) / (NF - 1)
        v = value(j, t)
        sc.frame_set(f)
        if jtype == "prismatic":
            loc = list(e.location);
            # prismatic moves along axis from rest origin — approximate along channel
            base = mathutils.Vector(j["origin"])
            e.location = base + mathutils.Vector([(v if i == ch else 0) for i in range(3)])
            e.keyframe_insert("location", index=ch)
        else:
            e.rotation_euler[ch] = v
            e.keyframe_insert("rotation_euler", index=ch)

# ---- quantitative self-check ----
def world_centroid(obj):
    co = [obj.matrix_world @ v.co for v in obj.data.vertices]
    n = len(co)
    return mathutils.Vector((sum(c.x for c in co) / n, sum(c.y for c in co) / n, sum(c.z for c in co) / n))

mesh_objs = {o.name: o for o in bpy.data.objects if o.type == "MESH"}
def get(nm):
    for k, o in mesh_objs.items():
        if k == nm or k.startswith(nm):
            return o
    return None

head = get("head"); barrel = get("barrel")
checks = []
if head:
    sc.frame_set(1); bpy.context.view_layer.update(); c0 = world_centroid(head)
    sc.frame_set(NF // 2); bpy.context.view_layer.update(); c1 = world_centroid(head)
    drift = math.hypot(c1.x - c0.x, c1.y - c0.y)
    checks.append(f"HEAD_XY_DRIFT={drift:.4f} (want ~0 = spins in place)")
if barrel:
    sc.frame_set(1); bpy.context.view_layer.update(); b0 = world_centroid(barrel)
    sc.frame_set(NF); bpy.context.view_layer.update(); b1 = world_centroid(barrel)
    checks.append(f"BARREL_Z_RISE={b1.z - b0.z:+.4f} (want >0 = elevates)")
print("SELFCHECK " + " | ".join(checks))

# ---- render a montage of sampled frames ----
import os
os.makedirs(render_dir, exist_ok=True)
# camera + lights
cam_d = bpy.data.cameras.new("c"); cam = bpy.data.objects.new("c", cam_d)
sc.collection.objects.link(cam); sc.camera = cam
cam.location = (4.2, -4.6, 3.2)
cam.rotation_euler = (mathutils.Vector((0, 0, 1.0)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
for ang in (0, 130, 250):
    l = bpy.data.lights.new(f"l{ang}", 'SUN'); l.energy = 3.0
    lo = bpy.data.objects.new(f"l{ang}", l); sc.collection.objects.link(lo)
    lo.rotation_euler = (math.radians(55), 0, math.radians(ang))
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[1].default_value = 0.7
try: sc.render.engine = 'BLENDER_EEVEE_NEXT'
except Exception: sc.render.engine = 'CYCLES'
sc.render.resolution_x = 480; sc.render.resolution_y = 480; sc.render.film_transparent = True

sample_frames = [1 + round(i * (NF - 1) / 5) for i in range(6)]
paths = []
for f in sample_frames:
    sc.frame_set(f); p = os.path.join(render_dir, f"f{f:03d}.png")
    sc.render.filepath = p; bpy.ops.render.render(write_still=True); paths.append(p)
print("FRAMES " + " ".join(paths))

bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=out_glb, use_selection=True, export_format="GLB")
print("ANIMATED", out_glb)
