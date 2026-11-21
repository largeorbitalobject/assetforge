import bpy, sys, numpy as np, mathutils, math
npz, meshglb, out, glb_out = sys.argv[-4], sys.argv[-3], sys.argv[-2], sys.argv[-1]
d=np.load(npz, allow_pickle=True)
joints=d['joints'].astype(np.float32); parents=d['parents'].astype(int)
skin=d['skinning_weights'].astype(np.float32); pc=d['pointcloud'].astype(np.float32)
# inverse coord swap (RigAnything used P@[[1,0,0],[0,0,-1],[0,1,0]]); invert with transpose
# swapped(RigAnything) -> Blender Z-up = (x, -y, -z)
_flip=np.array([1,-1,-1],np.float32)
joints=joints*_flip; pc=pc*_flip
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=meshglb)
obj=[o for o in bpy.context.scene.objects if o.type=='MESH'][0]
# armature
arm_data=bpy.data.armatures.new("rig"); arm=bpy.data.objects.new("rig",arm_data)
bpy.context.collection.objects.link(arm); bpy.context.view_layer.objects.active=arm
bpy.ops.object.mode_set(mode='EDIT')
eb=arm_data.edit_bones; bones=[]
for j in range(len(joints)):
    p=parents[j]; b=eb.new(f"b{j}")
    if p==j: b.head=joints[j]; b.tail=joints[j]+mathutils.Vector((0,0.04,0))
    else:
        b.head=mathutils.Vector(joints[p]); b.tail=mathutils.Vector(joints[j])
    if (b.tail-b.head).length<1e-4: b.tail=b.head+mathutils.Vector((0,0.03,0))
    bones.append(b)
for j in range(len(joints)):
    p=parents[j]
    if p!=j: bones[j].parent=bones[p]
bpy.ops.object.mode_set(mode='OBJECT')
# skinning transfer: mesh vert -> nearest pointcloud -> weights
verts=np.array([ (obj.matrix_world@v.co)[:] for v in obj.data.vertices],np.float32)
from scipy.spatial import cKDTree
nn=cKDTree(pc.astype(np.float32)).query(verts.astype(np.float32), k=1)[1]
W=skin[nn]  # (Nv,52)
# top-4 weights per vertex
order=np.argsort(-W,axis=1)[:,:4]
vgs=[obj.vertex_groups.new(name=f"b{j}") for j in range(len(joints))]
for vi in range(len(verts)):
    ws=W[vi]; tot=0.0
    sel=order[vi]; s=ws[sel].sum()+1e-9
    for j in sel:
        w=float(ws[j]/s)
        if w>1e-3: vgs[j].add([vi], w, 'REPLACE')
mod=obj.modifiers.new("arm","ARMATURE"); mod.object=arm; mod.use_vertex_groups=True
obj.parent=arm
bpy.ops.object.select_all(action='SELECT')
bpy.ops.export_scene.gltf(filepath=glb_out, use_selection=False, export_format='GLB')
print("rig exported", glb_out, "joints", len(joints))
# ---- visualize: translucent mesh + red bone cylinders ----
mt=bpy.data.materials.new("m"); mt.use_nodes=True; b=mt.node_tree.nodes['Principled BSDF']
b.inputs['Base Color'].default_value=(0.7,0.7,0.72,1); b.inputs['Alpha'].default_value=0.30
mt.blend_method='BLEND'
obj.data.materials.clear(); obj.data.materials.append(mt)
for p in obj.data.polygons: p.use_smooth=True
redm=bpy.data.materials.new("red"); redm.use_nodes=True
em=redm.node_tree.nodes.new("ShaderNodeEmission"); em.inputs[0].default_value=(1,0.1,0.1,1); em.inputs[1].default_value=2
redm.node_tree.links.new(em.outputs[0], redm.node_tree.nodes['Material Output'].inputs['Surface'])
rad=float(np.ptp(joints,axis=0).max())*0.5 or 1
for j in range(len(joints)):
    p=parents[j]
    if p==j: continue
    h=mathutils.Vector(joints[p]); t=mathutils.Vector(joints[j]); v=t-h; L=v.length
    if L<1e-4: continue
    bpy.ops.mesh.primitive_cylinder_add(radius=rad*0.012, depth=L, location=(h+t)/2)
    cyl=bpy.context.active_object; cyl.data.materials.append(redm)
    z=mathutils.Vector((0,0,1)); cyl.rotation_mode='QUATERNION'; cyl.rotation_quaternion=z.rotation_difference(v.normalized())
# joints as spheres
for j in range(len(joints)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=rad*0.022, location=mathutils.Vector(joints[j]))
    bpy.context.active_object.data.materials.append(redm)
# camera/lights/render
mn=np.min(verts,0); mx=np.max(verts,0); center=mathutils.Vector(((mn+mx)/2)); R=float(np.max(mx-mn))*0.5 or 1
sc=bpy.context.scene; sc.render.engine='CYCLES'; sc.cycles.device='CPU'; sc.cycles.samples=24
sc.render.film_transparent=False; sc.world=bpy.data.worlds.new("W"); sc.world.use_nodes=True
sc.world.node_tree.nodes['Background'].inputs[0].default_value=(0.1,0.11,0.13,1)
s=bpy.data.objects.new("k",bpy.data.lights.new("k",'SUN')); sc.collection.objects.link(s); s.data.energy=3; s.rotation_euler=(math.radians(55),0,math.radians(35))
camd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",camd); sc.collection.objects.link(cam); sc.camera=cam
sc.render.resolution_x=520; sc.render.resolution_y=720; sc.render.image_settings.file_format='PNG'
ps=[]
for nm,az in [('front',0),('threeq',40)]:
    a=math.radians(az); el=math.radians(6); dist=R*2.8
    cam.location=center+mathutils.Vector((dist*math.sin(a),-dist*math.cos(a),R*0.1+dist*math.sin(el)))
    dvec=center-cam.location; cam.rotation_euler=dvec.to_track_quat('-Z','Y').to_euler()
    p=out.replace('.png',f'_{nm}.png'); sc.render.filepath=p; bpy.ops.render.render(write_still=True); ps.append(p)
print("RIGVIZ", " ".join(ps))
