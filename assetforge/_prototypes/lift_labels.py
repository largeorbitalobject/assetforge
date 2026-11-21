import numpy as np, cv2, json, os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
TEX="/home/orbital/.claude/jobs/fde14dc0/tmp/tex"
cams=json.load(open(f"{TEX}/cams.json")); meta=cams['meta']
cam_dist=meta['cam_dist']; rad=meta['rad']; ortho=meta['ortho']; res=meta['res']; center=np.array(meta['center'])
CONCEPTS=['fur','armor','horns','tail']
pos=cv2.imread(f"{TEX}/pos_atlas.exr",cv2.IMREAD_UNCHANGED); nrm=cv2.imread(f"{TEX}/nrm_atlas.exr",cv2.IMREAD_UNCHANGED)
H,W=pos.shape[:2]
X=(pos[:,:,2]-0.5)*2*rad+center[0]; Y=(pos[:,:,1]-0.5)*2*rad+center[1]; Z=(pos[:,:,0]-0.5)*2*rad+center[2]
NX=(nrm[:,:,2]-0.5)*2; NY=(nrm[:,:,1]-0.5)*2; NZ=(nrm[:,:,0]-0.5)*2
cov=cv2.imread(f"{TEX}/cov_atlas.png",cv2.IMREAD_GRAYSCALE); mask=cov>40
idx=np.where(mask); Nn=len(idx[0])
P=np.stack([X[idx],Y[idx],Z[idx]],1); Nw=np.stack([NX[idx],NY[idx],NZ[idx]],1)
Nw/=(np.linalg.norm(Nw,axis=1,keepdims=True)+1e-9); Ph=np.concatenate([P,np.ones((Nn,1))],1)
votes={c:np.zeros(Nn) for c in CONCEPTS}; wsum=np.zeros(Nn); EPS=rad*0.03
for v in cams['views']:
    if not os.path.exists(f"{TEX}/gen_{v}.png"): continue
    M=np.array(cams['views'][v]['M']); Minv=np.linalg.inv(M)
    Pc=Ph@Minv.T; xc,yc,zc=Pc[:,0],Pc[:,1],Pc[:,2]; depth=-zc
    px=((xc/(ortho/2))*0.5+0.5)*res; py=(0.5-0.5*(yc/(ortho/2)))*res
    pxi=np.round(px).astype(int); pyi=np.round(py).astype(int)
    inb=(pxi>=0)&(pxi<res)&(pyi>=0)&(pyi<res)
    cam_z=M[:3,2]/np.linalg.norm(M[:3,2]); facing=Nw@cam_z
    flat=np.where(inb,pyi*res+pxi,0); zbuf=np.full(res*res,1e9); np.minimum.at(zbuf,flat[inb],depth[inb])
    vis=inb&(depth<=zbuf[flat]+EPS)&(facing>0.05)
    w=np.where(vis,np.clip(facing,0,1)**2,0.0); pxc=np.clip(pxi,0,res-1); pyc=np.clip(pyi,0,res-1)
    wsum+=w
    for c in CONCEPTS:
        mp=f"{TEX}/mask_{v}_{c}.png"
        if not os.path.exists(mp): continue
        mk=cv2.imread(mp,cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
        votes[c]+= (mk[pyc,pxc])*w
# normalize votes, argmax -> label; skin = none
# CONCEPTS order = ['fur','armor','horns','tail']
nv={c:votes[c]/(wsum+1e-9) for c in CONCEPTS}
covered=wsum>1e-6
# armor is the DEFAULT body material; carve out the rest by priority
label=np.full(Nn,-1)            # -1 = uncovered (true gaps only)
label[covered]=1               # default armor
label[(covered)&(nv['fur']>0.18)]=0      # fur ruff/limbs
label[(covered)&(nv['tail']>0.12)]=3     # tail
label[(covered)&(nv['horns']>0.12)]=2    # horns (highest priority)
# write per-concept mask atlases (smoothed)
for i,c in enumerate(CONCEPTS):
    atl=np.zeros((H,W),np.float32); m=(label==i)
    vals=np.zeros(Nn); vals[m]=1.0; atl[idx]=vals
    atl=cv2.GaussianBlur(atl,(0,0),2.0)
    cv2.imwrite(f"{TEX}/seg_{c}.png",(np.clip(atl,0,1)*255).astype(np.uint8))
# skin mask
atl=np.zeros((H,W),np.float32); vals=np.zeros(Nn); vals[label==-1]=1.0; atl[idx]=vals
atl=cv2.GaussianBlur(atl,(0,0),2.0); cv2.imwrite(f"{TEX}/seg_skin.png",(np.clip(atl,0,1)*255).astype(np.uint8))
print("LIFT_DONE", {c:int((label==i).sum()) for i,c in enumerate(CONCEPTS)}, "skin",int((label==-1).sum()))
