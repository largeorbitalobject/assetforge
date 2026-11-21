import bpy, sys, numpy as np, json
rigglb = sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=rigglb)
arm=[o for o in bpy.context.scene.objects if o.type=='ARMATURE'][0]
bones=list(arm.data.bones); name2i={b.name:i for i,b in enumerate(bones)}; N=len(bones)
par=[ name2i[b.parent.name] if b.parent else i for i,b in enumerate(bones)]
P=np.array([list(b.head_local) for b in bones],np.float32)
T=np.array([list(b.tail_local) for b in bones],np.float32)
ch={i:[] for i in range(N)}
for i in range(N):
    if par[i]!=i: ch[par[i]].append(i)
def depth(i):
    d=0
    while par[i]!=i: i=par[i]; d+=1
    return d
zmin,zmax=P[:,2].min(),P[:,2].max(); H=zmax-zmin; cx=float(np.median(P[:,0])); ymean=float(P[:,1].mean())
print(f"N={N} z[{zmin:.2f},{zmax:.2f}] H={H:.2f} cx={cx:.3f} ymean={ymean:.3f}")
print("idx par depth nch leaf   x      y      z    relZ   sideX")
for i in range(N):
    rz=(P[i,2]-zmin)/H; sx=(P[i,0]-cx)/H
    print(f"{i:3d} {par[i]:3d} {depth(i):3d} {len(ch[i]):3d} {'L' if not ch[i] else ' '}   {P[i,0]:+.2f} {P[i,1]:+.2f} {P[i,2]:+.2f}  {rz:.2f}  {sx:+.2f}")
json.dump({'par':par,'P':P.tolist(),'T':T.tolist(),'zmin':float(zmin),'zmax':float(zmax),'H':float(H),'cx':cx,'ymean':ymean},
          open(sys.argv[-1].replace('.glb','_skel.json') if False else '/home/orbital/.claude/jobs/fde14dc0/tmp/tex/skel.json','w'))
print("SKEL_JSON_SAVED")
