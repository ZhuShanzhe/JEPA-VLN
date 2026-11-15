"""Cache frozen, pooled V-JEPA temporal features for the wide branch-release."""
import argparse,json,os,sys,time,types
from pathlib import Path
import numpy as np, torch
from PIL import Image
from offline_branch_loader import inputs,supervised_future
root=Path(__file__).resolve().parents[2];base=root/'a_exploration/paired_branches_v2'
parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--watch',action='store_true');a=parser.parse_args()
sys.path.insert(0,str(root/'src/vjepa2'));app=types.ModuleType('app');app.__path__=[str(root/'src/vjepa2/app')];sys.modules['app']=app
from src.hub.backbones import _clean_backbone_key,vjepa2_1_vit_base_384
enc,head=vjepa2_1_vit_base_384(pretrained=False);del head
path=root/'models/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt'
ckpt=torch.load(path,mmap=True,map_location='cpu',weights_only=True)
enc.load_state_dict(_clean_backbone_key(ckpt['ema_encoder']),strict=True);del ckpt
enc.eval().cuda()
mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1,1);std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1,1)
def clip(frames):
    selected=np.linspace(0,len(frames)-1,4,dtype=np.int64)
    batch=np.stack([np.asarray(Image.fromarray(frames[i]).resize((384,384),Image.Resampling.BILINEAR)) for i in selected])
    return torch.from_numpy(batch.copy()).permute(3,0,1,2)
@torch.inference_mode()
def embed(clips):
    x=(torch.stack(clips).cuda().float().div(255.)-mean)/std
    tokens=enc(x);vec=tokens.float().mean(1).cpu().numpy()
    assert vec.shape==(len(clips),768) and np.isfinite(vec).all()
    return vec
cohort=json.loads((base/'cohort.json').read_text());requested=sum(j['samples'] for j in cohort['jobs'])
limit=requested if a.limit is None else a.limit
while True:
    available=sorted(base.glob('*/*/seed*/start*/manifest.json'))[:limit]
    for m in available:
        data=json.loads(m.read_text());n=data['frontier_count']
        p=m.parent/'features_vjepa21_b384.npz'
        if p.exists():continue
        first=clip(inputs(m.parent)['history_rgb']);ctx=embed([first])[0]
        targets=[]
        for i in range(0,n,4):
            clips=[clip(supervised_future(m.parent,b)['future_rgb']) for b in range(i,min(n,i+4))]
            targets.extend(embed(clips))
        tmp=p.with_suffix('.part')
        with tmp.open('wb') as f:np.savez(f,context_embedding=ctx,future_targets=np.stack(targets))
        tmp.replace(p)
    ready=len(list(base.glob('*/*/seed*/start*/features_vjepa21_b384.npz')))
    print('CACHED',ready,'/',limit,flush=True)
    if not a.watch or ready>=limit:break
    time.sleep(20)
(base/'feature_protocol.json').write_text(json.dumps(dict(checkpoint_sha256='848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d',model='V-JEPA2.1 ViT-B/16 384 EMA',input='4 evenly spaced RGB frames within each segment, bilinear resize to 384 x 384',normalization='ImageNet RGB',pool='mean over spatiotemporal tokens, float32',use='context for predictor input; future features only for supervised targets',cached=ready),indent=2))
