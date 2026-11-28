"""Compute embeddings of training-only terminal images from the frozen visual encoder."""
import os,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pose_horizon_20261128'))
from long_data import FrozenImageEncoder
import torch
torch.set_num_threads(2)
encoder=FrozenImageEncoder(HERE.parent)
os.umask(0o077)
while True:
    for f in sorted((HERE/'private_render').glob('*/labels.json')):
        dest=f.parent/'future.npy'
        if dest.exists():continue
        rows=json.loads(f.read_text())
        idx=[i for i,r in enumerate(rows) if r['reached']]
        feat=np.zeros((len(rows),1024),np.float16)
        for offset in range(0,len(idx),48):
            ids=idx[offset:offset+48]
            img=[]
            for i in ids:
                with Image.open(f.parent/rows[i]['image']) as im:
                    img.append(np.array(im.convert('RGB')))
            feat[ids]=encoder.encode(img)
        tmp=dest.with_name('future.partial.npy')
        np.save(tmp,feat)
        tmp.replace(dest)
        print(f.parent.name,flush=True)
    if len(list((HERE/'private_render').glob('*/future.npy')))==18:break
    time.sleep(2)
