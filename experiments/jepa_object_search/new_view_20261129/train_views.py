"""Matched visibility heads on training houses. Weigh hard requests; no endpoint labels at inference."""
import copy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from view_model import current_inputs, sketch, NewViewHead, SKETCH, SEED

HERE = Path(__file__).resolve().parent
OUT = HERE / "private_model"


def load_split(split):
    records = []
    houses = []
    seen = hard_count = 0
    for file in sorted((HERE / "private_data").glob(split + "_*.npz")):
        if ".partial" in file.name: continue
        houses.append(file.stem.split("_", 1)[1].split("_")[0])
        with np.load(file, allow_pickle=False) as d:
            arrays = {k: d[k] for k in ("group","view_id","kind","action","reach","visible","context","pose","map","future","future_roi","start_pixels")}
        group, views = arrays["group"], arrays["view_id"]
        for name in np.unique(group):
            rows = np.flatnonzero(group == name)
            admitted = rows[arrays["reach"][rows].astype(bool)]
            old = rows[arrays["kind"][rows] == "old"]
            assert len(old) == 7
            hard = not arrays["reach"][old].any()
            first_small = int(arrays["start_pixels"][rows[0]]) < 369
            seen += 1
            hard_count += int(hard)
            if not len(admitted): continue
            # For a given viewpoint the camera pose is identical even when several routes arrive.
            selected = []
            for vid in np.unique(views[admitted]):
                alt = admitted[views[admitted] == vid]
                selected.append(alt[np.argmin((arrays["action"][alt] > 0).sum(1))])
            i = np.asarray(selected, dtype=int)
            ctx = arrays["context"][i].astype(np.float32)
            obs_map = arrays["map"][i]
            x, prior = current_inputs(ctx, arrays["pose"][i], obs_map)
            full = sketch(arrays["future"][i] - ctx[:, :1024])
            roi = arrays["future_roi"][i].astype(np.float32)
            valid_roi = (np.linalg.norm(roi, axis=1) > .1)
            delta = np.column_stack((full, sketch(roi - ctx[:, 1024:])))
            mask = np.column_stack((np.ones_like(full, dtype=bool), np.broadcast_to(valid_roi[:, None], full.shape)))
            labels = arrays["visible"][i].astype(np.float32)
            assert np.isin(labels, [0, 1]).all()
            records.append((x,prior,labels,delta,mask,float((2 if hard else 1)*(1.5 if first_small else 1)),
                            bool(hard),obs_map.shape[1]))
    expected=json.loads((HERE/"view_design.json").read_text())["cohorts"][split]
    if sorted(houses)!=sorted(h.split("_")[0] for h in expected):
        raise ValueError("All frozen house shards must be present before fitting: "+split)
    if not records:raise ValueError("No reachable labelled cameras in "+split)
    x,pr,y,f,m = [np.concatenate([r[k] for r in records], axis=0) for k in range(5)]
    w, ix, pos, neg = [], 0, [], []
    for _,_,v,_,_,weight,*_ in records:
        w.extend([weight/len(v)]*len(v))
        good, bad = np.flatnonzero(v == 1), np.flatnonzero(v == 0)
        if len(good) and len(bad):
            a,b = np.meshgrid(good+ix,bad+ix,indexing="ij")
            pos.extend(a.ravel());neg.extend(b.ravel())
        ix += len(v)
    return dict(x=torch.from_numpy(x), pr=torch.from_numpy(pr), y=torch.from_numpy(y),
                targets=torch.from_numpy(f), mask=torch.from_numpy(m), w=torch.tensor(w),
                pos=torch.tensor(pos,dtype=torch.long),neg=torch.tensor(neg,dtype=torch.long),
                map_dims=records[0][-1], houses=houses,
                counts=dict(houses=len(houses),requests=seen,old_seven_blocked=hard_count,
                            requests_with_reached_endpoint=len(records),old_seven_blocked_with_endpoint=sum(r[-2] for r in records),
                            unique_reached_views=len(y),clear_views=int(y.sum()),future_roi_views=int(m[:,-1].sum()),
                            same_request_positive_negative_pairs=len(pos)))


def fit(variant,tr,va):
    torch.manual_seed(SEED)
    net=NewViewHead(tr["x"].shape[1])
    opt=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.012)
    mu=tr["x"].mean(0); scale=tr["x"].std(0,unbiased=False).clamp_min(.05)
    count=tr["mask"].sum(0).clamp_min(1)
    tmu=(tr["targets"]*tr["mask"]).sum(0)/count
    tscale=(((tr["targets"]-tmu).square()*tr["mask"]).sum(0)/count).sqrt().clamp_min(.10)
    def inputs(d):
        x=((d["x"]-mu)/scale).clamp(-6,6)
        if variant=="no_map": x[:,-tr["map_dims"]:]=0
        return x
    xt,xv=inputs(tr),inputs(va)
    yt=(tr["targets"]-tmu)/tscale
    yt=yt.clamp(-6,6)
    def classification(d,logits):
        bce=F.binary_cross_entropy_with_logits(logits,d["y"],reduction="none")
        result=(bce*d["w"]).sum()/d["w"].sum()
        if len(d["pos"]):
            result=result+.30*F.softplus(logits[d["neg"]]-logits[d["pos"]]).mean()
        return result
    best=float("inf")
    for epoch in range(1,101):
        logits,embedding=net(xt,tr["pr"])
        loss=classification(tr,logits)
        if variant!="direct":
            # Predict the observed future with the current frame as a skip connection.
            mse=(embedding-yt).square()
            loss=loss+.45*(mse*tr["mask"]).sum()/tr["mask"].sum()
        opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            output=net(xv,va["pr"])[0]
            score=float(classification(va,output))
            if score<best:
                best=score
                selected=epoch
                weights=copy.deepcopy(net.state_dict())
    file=OUT/(variant+".pt")
    torch.save(dict(weights=weights,mean=mu,scale=scale,target_mean=tmu,target_scale=tscale,
                    variant=variant,map_channels=tr["map_dims"]),file)
    return dict(epoch=selected,calibration_objective=best,sha256=hashlib.sha256(file.read_bytes()).hexdigest())


def main():
    os.umask(0o077);OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    tr,va=load_split("train"),load_split("val")
    assert not set(tr["houses"]).intersection(va["houses"])
    design=json.loads((HERE.parent/"path_feedback_20261128/natural_design.json").read_text())["cohorts"]
    for split, data in (("train",tr),("val",va)):
        expected={h.split("_")[0] for h in design[split]}
        assert set(data["houses"])==expected, "incomplete " + split
    report=dict(train=tr["counts"],val=va["counts"],seed=SEED,
                target="Predict projected normalized frozen CLIP future scene and same-instance crop residuals (24 coordinates each); supervise visible on physically reached endpoints only, 369-pixel criterion. Current-input geometric projected area prior shared by matched controls.",
                weights="Hard original requests with all seven physically blocked weighted x2, initially small cues weighted x1.5; reached endpoints deduplicated per request and pairs ranked within request; all weights use training/calibration data only.",
                models={k:fit(k,tr,va) for k in ("jepa","direct","no_map")})
    (HERE/"view_training.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)

if __name__=="__main__":main()
