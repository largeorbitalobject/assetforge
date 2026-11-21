import torch, numpy as np, cv2, os, sys
from PIL import Image
from transformers import Sam3Processor, Sam3Model
TEX="/home/orbital/.claude/jobs/fde14dc0/tmp/tex"
REPO=os.environ.get("SAM3_REPO","facebook/sam3.1")
dev="cuda" if torch.cuda.is_available() else "cpu"
print("loading",REPO,"on",dev)
model=Sam3Model.from_pretrained(REPO).to(dev).eval()
proc=Sam3Processor.from_pretrained(REPO)
VIEWS=['front','fl','fr','back','bl','br']
# concept -> text prompt (order = priority; later wins ties in lift)
CONCEPTS={'fur':'white fur','armor':'golden metal armor plate','horns':'large horns','tail':'animal tail'}
for v in VIEWS:
    img=Image.open(f"{TEX}/gen_{v}.png").convert("RGB"); W,H=img.size
    for c,prompt in CONCEPTS.items():
        inp=proc(images=img, text=prompt, return_tensors="pt").to(dev)
        with torch.no_grad(): out=model(**inp)
        res=proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5, target_sizes=[(H,W)])[0]
        masks=res.get('masks'); scores=res.get('scores')
        acc=np.zeros((H,W),np.float32)
        if masks is not None and len(masks)>0:
            m=masks.cpu().numpy()
            for i in range(m.shape[0]):
                acc=np.maximum(acc, m[i].astype(np.float32))
        cv2.imwrite(f"{TEX}/mask_{v}_{c}.png",(acc*255).astype(np.uint8))
        print(v,c,"instances",0 if masks is None else len(masks))
print("SEG_DONE")
