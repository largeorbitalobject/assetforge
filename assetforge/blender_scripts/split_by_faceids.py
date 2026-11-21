"""Split a mesh into part GLBs by face-index sets, IN BLENDER (consistent face indexing with
render_faceids.py, which is the only reliable index space — trimesh reorders + the textured mesh
is an unwelded triangle soup).

    blender -b -P split_by_faceids.py -- <glb> <labels.json> <out_dir> [min_comp_frac] [split_instances]

labels.json: {label: [face_index,...]} (0-based, matching render_faceids polygon.index).
For each label: select those polygons on the (unwelded, identically-joined) mesh, separate them,
then WELD the extracted part (merge-by-distance) and split into connected components -> drop noise
(< min_comp_frac) and optionally emit per-instance parts. Also exports the remainder as 'body'.
"""
import bpy, sys, json, os

a = sys.argv[sys.argv.index("--") + 1:]
glb, labels_json, out_dir = a[0], a[1], a[2]
min_comp_frac = float(a[3]) if len(a) > 3 else 0.08
split_instances = (a[4].lower() in ("1", "true", "yes")) if len(a) > 4 else False
labels = json.load(open(labels_json))
os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes: o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1: bpy.ops.object.join()
base = bpy.context.view_layer.objects.active
nF = len(base.data.polygons)


def export_obj(obj, path):
    bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(filepath=path, use_selection=True, export_format="GLB")


def weld_and_components(obj, min_frac):
    """Merge by distance, split into loose parts, drop small; return list of component objects."""
    bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    import numpy as _np
    dim = max(obj.dimensions) or 1.0
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=dim * 1e-3)
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    comps = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    print(f"   [weld] dim={dim:.3f} thr={dim*1e-3:.5f} comps={len(comps)} sizes={sorted([len(c.data.polygons) for c in comps], reverse=True)[:8]}")
    if not comps:
        comps = [obj]
    total = sum(len(c.data.polygons) for c in comps) or 1
    comps = [c for c in comps if len(c.data.polygons) >= max(1, min_frac * total)]
    comps.sort(key=lambda c: len(c.data.polygons), reverse=True)
    return comps


out = {}
for label, faces in labels.items():
    faces = [int(f) for f in faces if 0 <= int(f) < nF]
    if not faces:
        continue
    # duplicate the base, keep only this label's faces
    bpy.ops.object.select_all(action="DESELECT"); base.select_set(True)
    bpy.context.view_layer.objects.active = base
    bpy.ops.object.duplicate()
    dup = bpy.context.view_layer.objects.active
    fset = set(faces)
    import bmesh
    me = dup.data
    bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    todel = [fc for fc in bm.faces if fc.index not in fset]
    bmesh.ops.delete(bm, geom=todel, context="FACES")
    bm.to_mesh(me); bm.free(); me.update()
    if len(dup.data.polygons) == 0:
        bpy.data.objects.remove(dup, do_unlink=True); continue
    comps = weld_and_components(dup, min_comp_frac)
    if not comps:
        continue
    if split_instances:
        for i, c in enumerate(comps):
            p = os.path.join(out_dir, f"part_{label}_{i}.glb"); export_obj(c, p); out[f"{label}_{i}"] = p
    else:
        # rejoin kept comps into one part
        bpy.ops.object.select_all(action="DESELECT")
        for c in comps: c.select_set(True)
        bpy.context.view_layer.objects.active = comps[0]
        if len(comps) > 1: bpy.ops.object.join()
        p = os.path.join(out_dir, f"part_{label}.glb"); export_obj(comps[0], p); out[label] = p

print("SPLIT", json.dumps(out))
