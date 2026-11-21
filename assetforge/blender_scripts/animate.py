import bpy, sys, numpy as np, mathutils, math
rigglb, outdir = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=rigglb)
arm=[o for o in bpy.context.scene.objects if o.type=='ARMATURE'][0]
obj=[o for o in bpy.context.scene.objects if o.type=='MESH'][0]
# rest positions (armature space)
bpy.context.view_layer.objects.active=arm
heads={b.name:np.array(b.head_local) for b in arm.data.bones}
allh=np.array(list(heads.values())); zmin,zmax=allh[:,2].min(),allh[:,2].max(); H=zmax-zmin
cx=np.median(allh[:,0])
def cls(name):
    h=heads[name]; rz=(h[2]-zmin)/(H+1e-9); dx=h[0]-cx
    if rz>0.6 and abs(dx)>0.10*H: return ('armR' if dx>0 else 'armL')
    if rz<0.45 and abs(dx)>0.04*H: return ('legR' if dx>0 else 'legL')
    if rz<0.55 and h[1]>allh[:,1].mean()+0.05*H: return 'tail'  # behind
    return 'spine'
groups={}
for b in arm.data.bones:
    groups.setdefault(cls(b.name),[]).append(b.name)
print("groups:", {k:len(v) for k,v in groups.items()})
# animate in pose mode
bpy.ops.object.mode_set(mode='POSE')
for pb in arm.pose.bones: pb.rotation_mode='XYZ'
scn=bpy.context.scene; FPS=24; N=48; scn.frame_start=1; scn.frame_end=N; scn.render.fps=FPS
import math
def key(pb, f, rx=0,ry=0,rz=0):
    pb.rotation_euler=(rx,ry,rz); pb.keyframe_insert('rotation_euler', frame=f)
root=arm.pose.bones[0]
for f in range(1,N+1):
    t=2*math.pi*(f-1)/N
    # breathing/bob via root
    root.location=(0,0,0.015*math.sin(t)); root.keyframe_insert('location',frame=f)
    # arms: gentle swing; right arm waves bigger
    for nm in groups.get('armR',[]): key(arm.pose.bones[nm], f, rx=0.12*math.sin(t)+0.10, rz=0.10*math.sin(t*1.0))
    for nm in groups.get('armL',[]): key(arm.pose.bones[nm], f, rx=-0.10*math.sin(t))
    # legs subtle alternate
    for nm in groups.get('legR',[]): key(arm.pose.bones[nm], f, rx=0.06*math.sin(t))
    for nm in groups.get('legL',[]): key(arm.pose.bones[nm], f, rx=-0.06*math.sin(t))
    # tail sway
    for nm in groups.get('tail',[]): key(arm.pose.bones[nm], f, rz=0.18*math.sin(t*1.5))
    # spine subtle
    for nm in groups.get('spine',[]): key(arm.pose.bones[nm], f, ry=0.02*math.sin(t))
for fc in arm.animation_data.action.fcurves:
    for kp in fc.keyframe_points: kp.interpolation='BEZIER'
# material clay
m=bpy.data.materials.new("c"); m.use_nodes=True; b=m.node_tree.nodes['Principled BSDF']
b.inputs['Base Color'].default_value=(0.62,0.6,0.58,1); b.inputs['Roughness'].default_value=0.6
obj.data.materials.clear(); obj.data.materials.append(m)
for p in obj.data.polygons: p.use_smooth=True
# scene/cam/light
allv=np.array([(obj.matrix_world@v.co)[:] for v in obj.data.vertices]); cen=mathutils.Vector(allv.mean(0)); R=float(np.ptp(allv,0).max())*0.5
scn.render.engine='CYCLES'; scn.cycles.device='CPU'; scn.cycles.samples=16
scn.world=bpy.data.worlds.new("W"); scn.world.use_nodes=True; scn.world.node_tree.nodes['Background'].inputs[0].default_value=(0.1,0.11,0.13,1)
for en,rot in [(3,(math.radians(55),0,math.radians(35))),(1.3,(math.radians(60),0,math.radians(-110)))]:
    s=bpy.data.objects.new("L",bpy.data.lights.new("L",'SUN')); scn.collection.objects.link(s); s.data.energy=en; s.rotation_euler=rot
camd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",camd); scn.collection.objects.link(cam); scn.camera=cam
a=math.radians(28); dist=R*2.9
cam.location=cen+mathutils.Vector((dist*math.sin(a),-dist*math.cos(a),R*0.15)); cam.rotation_euler=(cen-cam.location).to_track_quat('-Z','Y').to_euler()
scn.render.resolution_x=360; scn.render.resolution_y=480; scn.render.image_settings.file_format='PNG'
import os; os.makedirs(outdir,exist_ok=True)
scn.render.filepath=outdir+"/f_"
bpy.ops.render.render(animation=True)
print("ANIM_DONE frames", N)
