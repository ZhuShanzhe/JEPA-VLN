from pathlib import Path
import json,os,sys
import numpy as np
import torch
from v9_live import Features
from v9_core import crop_box,field_labels

D=Path(__file__).resolve().parent
C=D.parent/"collection"


def save(path,**data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,**data);os.replace(tmp,path)


def encode(views,split,mem):
    rows=json.loads((C/(split+".index.json")).read_bytes())
    for k,row in enumerate(rows):
        output=D/("memory" if mem else "features")/(row["sid"]+".npz")
        if output.exists():continue
        with np.load(row["observed"]) as obs:frames=obs["rgb"]
        if not mem:
            code,sim=views.encode(frames)
            save(output,code=code.numpy(),sim=sim.numpy())
        else:
            with np.load(row["labels"]) as lab:
                grid=lab["grid"];ids=lab["instance_id"]
                use=field_labels(grid,lab["eligible"]); size=grid.sum((-1,-2))*use
                crops=[]; chosen=[]
                for i in np.argsort(-size.max(0).astype(float))[:6]:
                    h=int(size[:,i].argmax())
                    if size[h,i]<192:continue
                    x1,y1,x2,y2=crop_box(grid[h,i]);crops.append(frames[h,y1:y2,x1:x2]);chosen.append(int(ids[i]))
            if crops:
                c,_=views.encode(crops);c=c.float().reshape(-1,7,7,64)[:,2:5,2:5].mean((1,2)).numpy()
            else:c=np.zeros((0,64),np.float32)
            save(output,ids=chosen,vectors=c)
        if k%100==0:print(split,mem,k,len(rows),flush=True)


if __name__=="__main__":
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    v=Features(D)
    (D/"words.json").write_text(json.dumps(v.names))
    for split in ["train","calibration","holdout"]:encode(v,split,False)
    for split in ["train","calibration"]:encode(v,split,True)
    print("complete",flush=True)
