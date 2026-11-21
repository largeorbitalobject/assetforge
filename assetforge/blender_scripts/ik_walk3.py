import bpy, sys, numpy as np, math
from mathutils import Vector, Quaternion
rigglb, outdir, quality = sys.argv[-3], sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=rigglb)
arm=[o for o in bpy.context.scene.objects if o.type=='ARMATURE'][0]
obj=[o for o in bpy.context.scene.objects if o.type=='MESH'][0]
bones=list(arm.data.bones); H=1.21; M=arm.matrix_world
def wtail(i): return M@Vector(bones[i].tail_local)
def whead(i): return M@Vector(bones[i].head_local)
PELVIS=0; SPINE=[1,2,3]; NECK=23
ARMA={'sh':26,'el':27}; ARMB={'sh':5,'el':6}
LEGA={'hip':44,'knee':45,'hock':46,'toe':47}; LEGB={'hip':48,'knee':49,'hock':50,'toe':51}
pelv=whead(PELVIS)
# ---- identify tail verts (posterior, central, lower) ----
V=np.array([(M@v.co)[:] for v in obj.data.vertices]); cx=float(np.median(V[:,0]))
post=V[:,1]-pelv.y   # +Y = behind
# bone point cloud (heads+tails of all existing bones)
BP=np.array([list(M@Vector(b.head_local)) for b in bones]+[list(M@Vector(b.tail_local)) for b in bones])
# min distance from each vert to nearest bone point
def mindist(pts):
    out=np.empty(len(pts))
    for i in range(0,len(pts),2000):
        d=((pts[i:i+2000,None,:]-BP[None,:,:])**2).sum(-1).min(1); out[i:i+2000]=np.sqrt(d)
    return out
zr=V[:,2]-pelv.z; xr=np.abs(V[:,0]-cx)
# tail = central column (data shows tail is xr<0.07 at all heights; calves are lateral xr>0.10)
tverts=np.where((post>0.13*H)&(zr<0.02*H)&(xr<0.08*H))[0]
print("TAILV",len(tverts))
# base near pelvis-posterior
base=np.array([cx, pelv.y+0.10*H, pelv.z])
dist=np.linalg.norm(V[tverts]-base,axis=1)
order=np.argsort(dist); tverts=tverts[order]; dist=dist[order]
# 4 segments by distance quantile
NB=4; nodes=[Vector(base)]
edges=np.quantile(dist,[0.25,0.5,0.75,1.0])
binidx=np.digitize(dist, edges[:-1])  # 0..3
for q in range(NB):
    pts=V[tverts[binidx==q]]
    nodes.append(Vector(pts.mean(0)) if len(pts)>0 else nodes[-1]+Vector((0,0.05*H,-0.03*H)))
# ---- add tail bones (edit) ----
bpy.context.view_layer.objects.active=arm; bpy.ops.object.mode_set(mode='EDIT')
eb=arm.data.edit_bones; prev=eb[bones[PELVIS].name]; tailnames=[]
Minv=arm.matrix_world.inverted()
nodesL=[Minv@n for n in nodes]   # world -> armature-local for edit bones
for k in range(NB):
    b=eb.new(f"tail{k}"); b.head=nodesL[k]; b.tail=nodesL[k+1]
    if (b.tail-b.head).length<1e-4: b.tail=b.head+Vector((0,0.04*H,0))
    b.parent=prev; b.use_connect=(k>0); prev=b; tailnames.append(b.name)
bpy.ops.object.mode_set(mode='OBJECT')
bones=list(arm.data.bones)  # re-fetch: edit-mode invalidated old refs
# ---- skin tail verts to tail bones (rigid per segment, clear old weights) ----
tvgs=[obj.vertex_groups.get(n) or obj.vertex_groups.new(name=n) for n in tailnames]
allvg=list(obj.vertex_groups)
pmin,pmax=float(dist.min()),float(dist.max())
# save each tail vert's ORIGINAL weights so we can restore them in proportion (calf verts keep leg weight)
orig={int(vi):[(g.group,g.weight) for g in obj.data.vertices[int(vi)].groups] for vi in tverts}
for ii,vi in enumerate(tverts):
    vi=int(vi); q=int(binidx[ii])
    pp=(dist[ii]-pmin)/(pmax-pmin+1e-9); x=min(pp/0.35,1.0); base=x*x*(3-2*x)   # 0 at tail base
    zg=max(0.0,min(1.0,(zr[vi]-(-0.22*H))/(0.10*H)))   # height gate: 0 at/below calf level, 1 above
    tw=base*zg                                          # tail-bone influence
    for vg in allvg: vg.remove([vi])
    if tw>0.02:
        tvgs[q].add([vi], tw, 'REPLACE')
        if q>0: tvgs[q-1].add([vi], 0.25*tw, 'ADD')
    keep=1.0-min(tw,1.0)                                # restore ORIGINAL weight where tail influence is low
    if keep>0.01:
        for gidx,w in orig[vi]:
            obj.vertex_groups[gidx].add([vi], w*keep, 'ADD')   # calf/base verts -> their own leg/body bone, no stretch
# smooth weights across the tail<->calf boundary to dissolve stray pinned (stretched) verts
nbr=set(int(v) for v in tverts)
for e in obj.data.edges:
    a,b=e.vertices
    if a in nbr or b in nbr: nbr.add(int(a)); nbr.add(int(b))
for v in obj.data.vertices: v.select=False
for vi in nbr: obj.data.vertices[vi].select=True
bpy.context.view_layer.objects.active=obj
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_mode(type='VERT')
bpy.ops.object.vertex_group_smooth(group_select_mode='ALL', factor=0.5, repeat=10)
bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.view_layer.objects.active=arm   # restore armature as active for POSE mode below
# ---- IK legs (ankle target, rigid toe, twist-locked) ----
ankA0=wtail(LEGA['hock']); ankB0=wtail(LEGB['hock']); ground_z=min(wtail(LEGA['toe']).z,wtail(LEGB['toe']).z)
def mk(n,l):
    e=bpy.data.objects.new(n,None); e.empty_display_size=0.05; bpy.context.collection.objects.link(e); e.location=l; return e
tgtA=mk("tgtA",ankA0); tgtB=mk("tgtB",ankB0)
bpy.ops.object.mode_set(mode='POSE')
for roles,tg in [(LEGA,tgtA),(LEGB,tgtB)]:
    pb=arm.pose.bones[bones[roles['hock']].name]; ik=pb.constraints.new('IK'); ik.target=tg; ik.chain_count=3
    for key in ['hip','knee','hock']: arm.pose.bones[bones[roles[key]].name].lock_ik_y=True
F=48; sc=bpy.context.scene; sc.frame_start=1; sc.frame_end=F; sc.render.fps=24
stride=0.26*H; lift=0.10*H; duty=0.62; BOB=0.035*H
def ank(phi, b):
    u=(phi/(2*math.pi))%1.0
    if u<duty: s=u/duty; return Vector((b.x, b.y-stride/2+stride*s, b.z))
    s=(u-duty)/(1-duty); return Vector((b.x, b.y+stride/2-stride*s, b.z+lift*math.sin(math.pi*s)))
Bn={b.name:b for b in arm.data.bones}
WX=Vector((1,0,0)); WY=Vector((0,1,0)); WZ=Vector((0,0,1))
for pb in arm.pose.bones:
    if not pb.constraints: pb.rotation_mode='QUATERNION'
def setrot(name, aa, f):
    pb=arm.pose.bones[name]; Mi=Bn[name].matrix_local.to_3x3().inverted(); q=Quaternion()
    for ax,an in aa:
        if abs(an)>1e-6: q=Quaternion((Mi@ax).normalized(),an)@q
    pb.rotation_quaternion=q; pb.keyframe_insert('rotation_quaternion',frame=f)
root=arm.pose.bones[bones[0].name]
# softer arms, casual hang (ARM small, elbow nearly straight)
PA=dict(PYAW=0.11,PROLL=0.08,SYAW=0.10,SBEND=0.03,NECK=0.08,ARM=0.16,EL=0.06,
        TAIL=0.10,TAIL_F=0.8,TAIL_DELAY=0.4,TAIL_DROP=0.03)
for f in range(1,F+1):
    t=2*math.pi*(f-1)/F
    tgtA.location=ank(t,ankA0); tgtA.keyframe_insert('location',frame=f)
    tgtB.location=ank(t+math.pi,ankB0); tgtB.keyframe_insert('location',frame=f)
    root.location=(0,0,-BOB*(0.5-0.5*math.cos(2*t))); root.keyframe_insert('location',frame=f)
    setrot(bones[PELVIS].name,[(WZ,PA['PYAW']*math.cos(t)),(WY,PA['PROLL']*math.sin(t))],f)
    for k,si in enumerate(SPINE):
        w=(k+1)/len(SPINE); setrot(bones[si].name,[(WZ,-PA['SYAW']*w*math.cos(t)),(WY,PA['SBEND']*w*math.sin(t))],f)
    setrot(bones[NECK].name,[(WZ,PA['NECK']*math.cos(t))],f)
    for roles,phi in [(ARMA,t+math.pi),(ARMB,t+2*math.pi)]:
        setrot(bones[roles['sh']].name,[(WX,PA['ARM']*math.cos(phi))],f)
        setrot(bones[roles['el']].name,[(WX,-PA['EL'])],f)   # tiny constant bend, no lift
    # tail: light independent sway + follow-through, NOT tied to legs
    tphase=t*PA['TAIL_F']
    for k,tn in enumerate(tailnames):
        setrot(tn,[(WZ, PA['TAIL']*math.sin(tphase - k*PA['TAIL_DELAY'])),
                   (WX, PA['TAIL_DROP']*math.sin(tphase*0.5 - k*0.4))],f)
for fc in arm.animation_data.action.fcurves:
    for kp in fc.keyframe_points: kp.interpolation='BEZIER'
m=bpy.data.materials.new("c"); m.use_nodes=True; b=m.node_tree.nodes['Principled BSDF']
b.inputs['Base Color'].default_value=(0.62,0.6,0.58,1); b.inputs['Roughness'].default_value=0.6
obj.data.materials.clear(); obj.data.materials.append(m)
for p in obj.data.polygons: p.use_smooth=True
deps=bpy.context.evaluated_depsgraph_get(); _minz=1e9
for _f in range(1,F+1,3):
    sc.frame_set(_f); deps.update(); _ev=obj.evaluated_get(deps); _me=_ev.to_mesh()
    _z=min((obj.matrix_world@v.co).z for v in _me.vertices); _minz=min(_minz,_z); _ev.to_mesh_clear()
ground_z=_minz-0.003; sc.frame_set(1)
bpy.ops.mesh.primitive_plane_add(size=H*6, location=(0,0,ground_z))
gp=bpy.context.active_object; gm=bpy.data.materials.new("g"); gm.use_nodes=True
gm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(0.18,0.19,0.22,1); gp.data.materials.append(gm)
sc.render.engine='CYCLES'; sc.cycles.device='CPU'; sc.cycles.samples=48 if quality=='final' else 12
sc.world=bpy.data.worlds.new("W"); sc.world.use_nodes=True; sc.world.node_tree.nodes['Background'].inputs[0].default_value=(0.09,0.10,0.12,1)
for en,rot in [(3.2,(math.radians(52),0,math.radians(40))),(1.4,(math.radians(64),0,math.radians(-110)))]:
    s=bpy.data.objects.new("L",bpy.data.lights.new("L",'SUN')); sc.collection.objects.link(s); s.data.energy=en; s.rotation_euler=rot
allv=np.array([(obj.matrix_world@v.co)[:] for v in obj.data.vertices]); cen=Vector((allv[:,0].mean(),allv[:,1].mean(),(allv[:,2].max()+ground_z)/2)); R=float(np.ptp(allv,0).max())*0.5
camd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",camd); sc.collection.objects.link(cam); sc.camera=cam
a=math.radians(72); dist=R*3.9
cam.location=cen+Vector((dist*math.sin(a),-dist*math.cos(a),R*0.12)); cam.rotation_euler=(cen-cam.location).to_track_quat('-Z','Y').to_euler()
res=(440,560) if quality=='final' else (320,400)
sc.render.resolution_x,sc.render.resolution_y=res; sc.render.image_settings.file_format='PNG'
import os; os.makedirs(outdir,exist_ok=True); sc.render.filepath=outdir+"/w_"
bpy.ops.render.render(animation=True); print("IK3_DONE")
