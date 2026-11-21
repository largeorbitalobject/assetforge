import numpy as np, cv2, json, os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
TEX="/home/orbital/.claude/jobs/fde14dc0/tmp/tex"
cams=json.load(open(f"{TEX}/cams.json")); meta=cams['meta']
cam_dist=meta['cam_dist']; rad=meta['rad']; ortho=meta['ortho']; res=meta['res']; center=np.array(meta['center'])
VW={'front':1.5,'back':1.2,'fl':1.0,'fr':1.0,'bl':1.0,'br':1.0}
pos=cv2.imread(f"{TEX}/pos_atlas.exr",cv2.IMREAD_UNCHANGED); nrm=cv2.imread(f"{TEX}/nrm_atlas.exr",cv2.IMREAD_UNCHANGED)
H,W=pos.shape[:2]
X=(pos[:,:,2]-0.5)*2*rad+center[0]; Y=(pos[:,:,1]-0.5)*2*rad+center[1]; Z=(pos[:,:,0]-0.5)*2*rad+center[2]
NX=(nrm[:,:,2]-0.5)*2; NY=(nrm[:,:,1]-0.5)*2; NZ=(nrm[:,:,0]-0.5)*2
cov=cv2.imread(f"{TEX}/cov_atlas.png",cv2.IMREAD_GRAYSCALE); mask=cov>40
idx=np.where(mask); Nn=len(idx[0]); print("texels",Nn)
P=np.stack([X[idx],Y[idx],Z[idx]],1); Nw=np.stack([NX[idx],NY[idx],NZ[idx]],1)
Nw/=(np.linalg.norm(Nw,axis=1,keepdims=True)+1e-9); Ph=np.concatenate([P,np.ones((Nn,1))],1)
accT=np.zeros((Nn,3)); wT=np.zeros((Nn,1))
EPS=rad*0.03
for v in cams['views']:
    gp=f"{TEX}/gen_{v}.png"
    if not os.path.exists(gp): continue
    M=np.array(cams['views'][v]['M']); Minv=np.linalg.inv(M)
    gen=cv2.imread(gp,cv2.IMREAD_COLOR).astype(np.float32)/255.0
    Pc=Ph@Minv.T; xc,yc,zc=Pc[:,0],Pc[:,1],Pc[:,2]; depth=-zc
    ndcx=xc/(ortho/2); ndcy=yc/(ortho/2)
    px=(ndcx*0.5+0.5)*res; py=(0.5-0.5*ndcy)*res
    pxi=np.round(px).astype(int); pyi=np.round(py).astype(int)
    inb=(pxi>=0)&(pxi<res)&(pyi>=0)&(pyi<res)
    cam_z=M[:3,2]/np.linalg.norm(M[:3,2]); facing=Nw@cam_z
    # self z-buffer occlusion
    flat=np.where(inb, pyi*res+pxi, 0)
    zbuf=np.full(res*res, 1e9)
    np.minimum.at(zbuf, flat[inb], depth[inb])
    znear=zbuf[flat]
    visible=inb & (depth<=znear+EPS) & (facing>0.05)
    pxc=np.clip(pxi,0,res-1); pyc=np.clip(pyi,0,res-1)
    col=gen[pyc,pxc]
    w=np.where(visible,(np.clip(facing,0,1)**2)*VW[v],0.0)[:,None]
    accT+=col*w; wT+=w
    print("view",v,"visible",int((w>0).sum()))
good=(wT[:,0]>1e-6)
atlas=np.zeros((H,W,3),np.float32); vals=np.zeros((Nn,3)); vals[good]=accT[good]/wT[good]; atlas[idx]=vals
atlas8=np.clip(atlas*255,0,255).astype(np.uint8)
covered=np.zeros((H,W),np.uint8); covered[idx]=(good).astype(np.uint8)*255
holes=(covered==0).astype(np.uint8)
atlas8=cv2.inpaint(atlas8, holes, 4, cv2.INPAINT_TELEA)
cv2.imwrite(f"{TEX}/albedo_gather.png", atlas8)
print("GATHER_DONE coverage %.1f%%"%(100*good.mean()))
