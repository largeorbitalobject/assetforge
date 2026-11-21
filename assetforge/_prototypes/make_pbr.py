import cv2, numpy as np
TEX="/home/orbital/.claude/jobs/fde14dc0/tmp/tex"
alb=cv2.imread(f"{TEX}/albedo_gather.png").astype(np.float32)/255.0  # BGR
H,W=alb.shape[:2]
def L(n):
    p=f"{TEX}/seg_{n}.png"; import os
    return (cv2.imread(p,cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0) if os.path.exists(p) else np.zeros((H,W),np.float32)
furf,armorf,hornf,tailf,skinf=L('fur'),L('armor'),L('horns'),L('tail'),L('skin')
luma=(alb*np.array([0.114,0.587,0.299])).sum(2)  # BGR weights
# --- base color per region (BGR) ---
gold=np.array([0.18,0.62,0.92])          # rich gold (BGR)
horn_c=np.array([0.10,0.17,0.27])        # dark keratin
fur_c=np.array([0.80,0.87,0.93])         # warm white
tail_c=np.array([0.72,0.80,0.86])        # off-white
armor_col=np.clip(gold[None,None,:]*(0.45+0.9*luma[...,None]),0,1)
fur_col  =np.clip(alb*0.35+fur_c[None,None,:]*0.75,0,1)
tail_col =np.clip(alb*0.4 +tail_c[None,None,:]*0.65,0,1)
horn_col =np.clip(horn_c[None,None,:]*(0.6+0.9*luma[...,None]),0,1)
base=alb.copy()
for col,f in [(armor_col,armorf+skinf),(fur_col,furf),(tail_col,tailf),(horn_col,hornf)]:
    f=np.clip(f,0,1)[...,None]; base=base*(1-f)+col*f
# --- metallic / roughness ---
metallic=np.clip(armorf+skinf-furf-hornf-tailf,0,1)
rough=np.clip(0.35*(armorf+skinf)+0.85*furf+0.82*tailf+0.5*hornf,0,1)
rough[rough<1e-3]=0.5
cv2.imwrite(f"{TEX}/pbr_base.png",(np.clip(base,0,1)*255).astype(np.uint8))
cv2.imwrite(f"{TEX}/pbr_metal.png",(metallic*255).astype(np.uint8))
cv2.imwrite(f"{TEX}/pbr_rough.png",(rough*255).astype(np.uint8))
print("PBR_MAPS_DONE armor%%=%.0f fur%%=%.0f horn%%=%.0f"%(100*armorf.mean(),100*furf.mean(),100*hornf.mean()))
