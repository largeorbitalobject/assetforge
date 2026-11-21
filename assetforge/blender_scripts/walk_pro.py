import bpy, sys, numpy as np, math, json
from mathutils import Vector, Quaternion
rigglb, outdir, quality = sys.argv[-3], sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=rigglb)
arm=[o for o in bpy.context.scene.objects if o.type=='ARMATURE'][0]
obj=[o for o in bpy.context.scene.objects if o.type=='MESH'][0]
bones=list(arm.data.bones); H=1.21
bpy.context.view_layer.objects.active=arm; bpy.ops.object.mode_set(mode='POSE')
Bn={b.name:b for b in arm.data.bones}
for pb in arm.pose.bones: pb.rotation_mode='QUATERNION'
WX=Vector((1,0,0)); WY=Vector((0,1,0)); WZ=Vector((0,0,1))
# accumulate multi-axis rotations per bone per frame
def setrot(idx, axis_angles, f):
    nm=bones[idx].name; pb=arm.pose.bones[nm]; M=Bn[nm].matrix_local.to_3x3().inverted()
    q=Quaternion()
    for axis,ang in axis_angles:
        if abs(ang)<1e-6: continue
        q=Quaternion((M@axis).normalized(),ang)@q
    pb.rotation_quaternion=q; pb.keyframe_insert('rotation_quaternion',frame=f)
# ROLES
PELVIS=0; SPINE=[1,2,3]; NECK=23; HEAD=24
LEGA={'hip':44,'knee':45,'hock':46,'toe':47}   # x+
LEGB={'hip':48,'knee':49,'hock':50,'toe':51}   # x-
ARMA={'sh':26,'el':27}  # x+  (right arm)
ARMB={'sh':5,'el':6}    # x-  (left arm)
# ---- PARAMS ----
P=dict(
 HIP=0.42, KNEE_BASE=0.0, KNEE_SW=0.45, HOCK_BASE=0.0, HOCK_SW=0.40, HOCK_PUSH=0.20, TOE=0.35,
 BOB=0.038, PYAW=0.20, PROLL=0.11, PPITCH=0.02, SYAW=0.12, SBEND=0.04,
 NECK_CY=0.10, HEAD_LV=0.06, ARM_SW=0.55, EL_BEND=0.55, EL_DRAG=0.18, FWD=0.0)
def swingwin(phi):  # 0 in stance, bump in swing (phi in [0,2pi), swing=[pi,2pi])
    return max(0.0, math.sin(phi-math.pi))
def pushwin(phi):   # narrow bump near toe-off ~phi=pi
    return max(0.0, math.sin(phi))**6 if phi<math.pi else 0.0
def leg(roles, phi, f, side):
    hip = P['HIP']*math.cos(phi)                       # fwd at contact, back at toe-off
    knee= -(P['KNEE_BASE'] + P['KNEE_SW']*swingwin(phi))
    hock=  (P['HOCK_BASE'] + P['HOCK_SW']*swingwin(phi) - P['HOCK_PUSH']*pushwin(phi))
    toe = -(P['TOE']*pushwin(phi)) + 0.15*swingwin(phi)
    setrot(roles['hip'], [(WX,hip)], f)
    setrot(roles['knee'],[(WX,knee)], f)
    setrot(roles['hock'],[(WX,hock)], f)
    setrot(roles['toe'], [(WX,toe)], f)
def arm_swing(roles, phi, f):
    sw = P['ARM_SW']*math.cos(phi)           # opposite handled by phase
    setrot(roles['sh'], [(WX,sw)], f)
    setrot(roles['el'], [(WX, -(P['EL_BEND']*0.5 + P['EL_BEND']*0.5*max(0,math.cos(phi-P['EL_DRAG']))))], f)
F=40; sc=bpy.context.scene; sc.frame_start=1; sc.frame_end=F; sc.render.fps=24
root=arm.pose.bones[bones[0].name]
for f in range(1,F+1):
    t=2*math.pi*(f-1)/F
    phiA=t; phiB=t+math.pi
    leg(LEGA, phiA, f, +1); leg(LEGB, phiB, f, -1)
    # arms swing opposite to same-side leg
    arm_swing(ARMA, phiA+math.pi, f); arm_swing(ARMB, phiB+math.pi, f)
    # pelvis: yaw twist, roll(weight), pitch, bob + slight fwd
    setrot(PELVIS, [(WZ, P['PYAW']*math.cos(t)), (WY, P['PROLL']*math.sin(t)), (WX, P['PPITCH']*math.cos(2*t))], f)
    root.location=(0, P['FWD']*math.sin(t), -P['BOB']*math.cos(2*t)); root.keyframe_insert('location',frame=f)
    # spine counter-yaw distributed + side bend
    for k,si in enumerate(SPINE):
        w=(k+1)/len(SPINE)
        setrot(si, [(WZ, -P['SYAW']*w*math.cos(t)), (WY, P['SBEND']*w*math.sin(t))], f)
    # neck/head: counter chest yaw, keep head level
    setrot(NECK, [(WZ, P['NECK_CY']*math.cos(t)), (WX, P['HEAD_LV']*math.cos(2*t))], f)
for fc in arm.animation_data.action.fcurves:
    for kp in fc.keyframe_points: kp.interpolation='BEZIER'
# material + render side
m=bpy.data.materials.new("c"); m.use_nodes=True; b=m.node_tree.nodes['Principled BSDF']
b.inputs['Base Color'].default_value=(0.62,0.6,0.58,1); b.inputs['Roughness'].default_value=0.6
obj.data.materials.clear(); obj.data.materials.append(m)
for p in obj.data.polygons: p.use_smooth=True
allv=np.array([(obj.matrix_world@v.co)[:] for v in obj.data.vertices]); cen=Vector(allv.mean(0)); R=float(np.ptp(allv,0).max())*0.5
sc.render.engine='CYCLES'; sc.cycles.device='CPU'
sc.cycles.samples= 48 if quality=='final' else 10
sc.world=bpy.data.worlds.new("W"); sc.world.use_nodes=True; sc.world.node_tree.nodes['Background'].inputs[0].default_value=(0.1,0.11,0.13,1)
for en,rot in [(3,(math.radians(55),0,math.radians(40))),(1.4,(math.radians(60),0,math.radians(-110)))]:
    s=bpy.data.objects.new("L",bpy.data.lights.new("L",'SUN')); sc.collection.objects.link(s); s.data.energy=en; s.rotation_euler=rot
camd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",camd); sc.collection.objects.link(cam); sc.camera=cam
a=math.radians(74); dist=R*3.7
cam.location=cen+Vector((dist*math.sin(a),-dist*math.cos(a),R*0.05)); cam.rotation_euler=(cen-cam.location).to_track_quat('-Z','Y').to_euler()
res=(440,580) if quality=='final' else (300,400)
sc.render.resolution_x,sc.render.resolution_y=res; sc.render.image_settings.file_format='PNG'
import os; os.makedirs(outdir,exist_ok=True); sc.render.filepath=outdir+"/w_"
bpy.ops.render.render(animation=True); print("WALKPRO_DONE")
