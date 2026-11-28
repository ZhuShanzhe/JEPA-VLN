"""Fit the layout-conditioned JEPA on curved HSSD trajectories; select on calibration houses only."""
import copy,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from layout_model import PROJECT,HERE,LayoutMap,encode_current,describe,learned_net,sketch
from observation_actor import box_anchor
BASE=PROJECT/'new_view_20261129'
SEED=20261129


def load(split,houses):
    xx=[];yy=[];zz=[];turns=[]
    for house in houses:
        folder=HERE/'private_render'/f'{split}_{house}'
        src=BASE/'private_render'/f'{split}_{house}'
        rows=json.loads((folder/'labels.json').read_text())
        images=np.load(folder/'future.npy')
        with np.load(BASE/'private_data'/f'{split}_{house}.npz') as d:
            ids=d['source_index']
            ctx=d['context'][np.array([np.flatnonzero(ids==i)[0] for i in range(16)])].astype(np.float32)
        inputs={}
        for idx in range(len(ctx)):
            with np.load(src/f'{idx}_sensor.npz') as d:
                anchor,box=box_anchor(d['box'],d['depth'])
                history=[dict(x=0,z=0,yaw=0,depth=d['depth']),dict(x=0,z=0,yaw=np.pi/6,depth=d['left_depth'])]
                patch=LayoutMap.from_depth_history(history,episode=house)
                inputs[idx]=patch,anchor,box,encode_current(ctx[idx],anchor,box)
        for r in rows:
            i=r['idx']
            patch,anchor,box,current=inputs[i]
            feature=describe(tuple(r['plan']),patch,np.zeros(2),0,anchor,box)
            xx.append(np.r_[current,feature])
            yy.append([r['reached'],r['px']>=369])
            turns.append(bool(r['bend']))
        zz.append(sketch(images.astype(np.float32)-ctx[[r['idx'] for r in rows],:1024]))
    return tuple(torch.as_tensor(x,dtype=torch.float32) for x in [np.stack(xx),np.array(yy),np.vstack(zz),turns])


def objective(logits,y):
    reached=y[:,0].bool()
    return F.binary_cross_entropy_with_logits(logits[:,0],y[:,0])+F.binary_cross_entropy_with_logits(logits[reached,1],y[reached,1])


def main():
    os.umask(0o077);torch.manual_seed(SEED);torch.set_num_threads(2)
    sets=json.loads((BASE/'view_design.json').read_text())['cohorts']
    assert not (set(sets['blind']) & set(sets['train']+sets['val']))
    x,y,z,curves=load('train',sets['train']);v,w,u,vcurves=load('val',sets['val'])
    mu=x.mean(0);sd=x.std(0).clamp_min(.05)
    x=((x-mu)/sd).clamp(-6,6);v=((v-mu)/sd).clamp(-6,6)
    observed=y[:,0].bool()
    zm=z[observed].mean(0);zs=z[observed].std(0).clamp_min(.1)
    z=((z-zm)/zs).clamp(-6,6)
    net=learned_net(x.shape[1])
    opt=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.013)
    best=float('inf')
    for epoch in range(1,111):
        for ids in torch.randperm(len(x)).split(384):
            obs=y[ids,0].bool();future,logits=net(x[ids])
            loss=objective(logits,y[ids])+.45*F.smooth_l1_loss(future[obs],z[ids][obs])
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            future,logits=net(v)
            val=float(objective(logits,w)+.45*F.smooth_l1_loss(future[w[:,0].bool()],((u-zm)/zs).clamp(-6,6)[w[:,0].bool()]))
            if val<best:
                best=val;mark=epoch;weights=copy.deepcopy(net.state_dict())
    out=HERE/'private_layout.pt'
    torch.save(dict(mu=mu,sd=sd,zm=zm,zs=zs,weights=weights),out)
    net.load_state_dict(weights)
    with torch.no_grad():
        f,lg=net(v);sel=w[:,0].bool()
        reach=((lg[:,0]>0)==w[:,0].bool()).float().mean().item()
        vis=((lg[sel,1]>0)==w[sel,1].bool()).float().mean().item()
    report={'train_houses':len(sets['train']),'calibration_houses':len(sets['val']),'train_requests':16*len(sets['train']),
            'calibration_requests':16*len(sets['val']),'train_routes':len(x),'train_curved':int(curves.sum()),'train_reached':int(y[:,0].sum()),
            'train_reached_and_clear':int(y[:,1].sum()),'validation_routes':len(v),'validation_reached':int(sel.sum()),
            'validation_reach_accuracy':reach,'conditional_clear_accuracy':vis,'epoch':mark,'seed':SEED,'checkpoint_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
            'target':'Frozen 24-coordinate projected future CLIP full-frame embedding residual when reached; actual reachability on all routes and instance area >=369 on reached endpoints.'}
    (HERE/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)
if __name__=='__main__':main()
