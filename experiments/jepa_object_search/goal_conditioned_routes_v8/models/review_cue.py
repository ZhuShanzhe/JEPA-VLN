"""Train and calibrate a causal prompt to recheck the current target at the current location."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_v8 import restore_heads

MODES=("current","observed_12")
MIN_PRECISION=.65
RIDGE=.04
PROTOCOL={
    "question":"A cold-start query and a physical origin: does at least one target instance already have a valid report viewpoint within 0.25 m measured on the navigation mesh?",
    "input":"Raw RGB already observed here, frozen RN50 target anchors and V8 grounder; all 16 terms scored; no depth, map, future features, class-presence flags, object IDs, physical poses, host outputs or labels in prediction.",
    "split":"144 native starts in 12 training houses; 48 native starts in four calibration houses; last 12 starts in three previously reserved switch-panel houses, reused for a new current-state target after the route gate was published. Fit or choose nothing on that panel.",
    "model":"Two separate fixed L2 logistic heads with the same input specification and ridge of 0.04. Features summarize the current grounder and target similarities. Each calibrated independently. No category IDs or learned category intercepts.",
    "observations":"current uses only original yaw bin 0. observed_12 uses twelve truly rendered orientations available in the offline observation panel, with no future motion; this mode requires all twelve directions to have already been observed before use in a host.",
    "weight":"Fit uses each physical origin with equal mass and equal weight among verified queries there. Metric denominators include every query with verified class viewpoints in that house. Unavailable classes are scored blind but excluded by the independent evaluator.",
    "calibration":"Per view mode maximize micro recall subject to empirical micro precision >=0.65 and >=1 predicted prompt. Resolve ties by higher precision, then higher threshold. If no such threshold is found, abstain and report it. Record count of distinct positive origins; no additional tuning.",
    "held_evaluation":"Freeze weights, thresholds, observations-only predictions and their hashes before any held labels are opened. Report TP/FP/FN/TN, fired fraction, recall, precision, AP, positive origin and prompt counts across houses; report official six and ten additional derived HSSD classes. No invocation of the original VLM and no changes to any paid run.",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, record):
    with path.open("x",encoding="utf-8") as out:
        json.dump(record,out,indent=2,allow_nan=False)


def arrays(visible,near,similarity):
    """Inputs [actual views,49,queries]; each output row belongs to a single target."""
    v=visible.numpy(); n=near.numpy(); s=similarity.numpy()
    def top(x,k):
        return np.sort(x,axis=1)[:,-k:,:].mean(axis=1)
    ratio=n/(v+1e-6)
    evidence=np.stack([v.max(1),top(v,4),v.mean(1),(v>=.25).mean(1),
                       n.max(1),top(n,4),n.mean(1),(n>=.12).mean(1),
                       top(s,3),s.mean(1),top(ratio*v**.5,3)],axis=-1)
    # Retain both the best-supported actual view for that target and cross-view maxima.
    qi=np.arange(s.shape[-1]);best=evidence[:,:,5].argmax(0)
    return np.concatenate([evidence[best,qi],evidence.max(0),evidence.mean(0)],axis=-1).astype(float)


@torch.inference_mode()
def observation_features(ground, code, anchor_sim, anchors):
    if code.shape!=(12,49,64) or anchor_sim.shape!=(12,49,len(anchors)):
        raise ValueError("expected twelve co-located genuinely observed 7x7 grids")
    visible,near=ground(code.float(),anchor_sim.float(),anchor_sim.float(),anchors.float())
    return [arrays(visible[:1],near[:1],anchor_sim[:1]), arrays(visible,near,anchor_sim)]


def sigmoid(values):
    return 1/(1+np.exp(-np.clip(values,-40,40)))


def fit(x,y,w):
    w=w/w.sum()
    center=w@x
    scale=np.maximum(np.sqrt(w@((x-center)**2)),1e-4)
    design=np.column_stack([np.ones(len(x)),(x-center)/scale])
    b=np.zeros(design.shape[1]);pen=np.diag([0.]+[RIDGE]*(len(b)-1))
    for _ in range(40):
        q=sigmoid(design@b)
        grad=design.T@(w*(q-y))+pen@b
        h=design.T@((w*q*(1-q))[:,None]*design)+pen
        step=np.linalg.solve(h,grad)
        b-=step
        if abs(step).max()<1e-9:break
    return dict(center=center.tolist(),scale=scale.tolist(),coefficient=b.tolist())


def scores(x,model):
    b=np.asarray(model["coefficient"])
    return sigmoid(b[0]+((x-model["center"])/model["scale"])@b[1:])


def metrics(y,score,fired):
    y=np.asarray(y,dtype=bool);fired=np.asarray(fired,dtype=bool)
    tp=int((y&fired).sum()); fp=int((~y&fired).sum())
    fn=int((y&~fired).sum()); tn=int((~y&~fired).sum())
    order=np.argsort(-score,kind="stable")
    ys=y[order]
    # Average precision with joint handling of identical scores.
    end=np.r_[np.flatnonzero(np.diff(score[order])!=0),len(y)-1]
    cum=ys.cumsum()[end]
    ap=(np.diff(np.r_[0,cum])*(cum/(1+end))).sum()/max(1,int(y.sum()))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,pairs=len(y),positives=int(y.sum()),
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                prompted_fraction=(tp+fp)/len(y),average_precision=float(ap) if y.any() else None)


def threshold(y,score):
    best=None
    for cut in np.unique(score):
        pred=score>=cut
        tp=(pred&y).sum();p=tp/pred.sum();r=tp/max(y.sum(),1)
        if p<MIN_PRECISION:continue
        key=(r,p,cut)
        if best is None or key>best:best=key
    return float(best[-1]) if best else None


def valid_targets(state):
    label=np.asarray(state["eligible"],dtype=int)
    return label>=0,label>0


def fit_heads(train,cal):
    if {x["house"] for x in train}&{x["house"] for x in cal}: raise ValueError("overlapping houses")
    out={}
    for mode in range(2):
        xs,ys,ws=[],[],[]
        for s in train:
            mask,y=valid_targets(s)
            xs.append(s["fields"][mode][mask]); ys.append(y[mask]);ws.append(np.ones(mask.sum())/mask.sum())
        model=fit(np.vstack(xs),np.concatenate(ys),np.concatenate(ws))
        xy=[(scores(s["fields"][mode],model)[valid_targets(s)[0]],valid_targets(s)[1][valid_targets(s)[0]]) for s in cal]
        pred=np.concatenate([k for k,_ in xy]);y=np.concatenate([l for _,l in xy])
        model["cut"]=threshold(y,pred);out[MODES[mode]]=model
    return out


def blind_predictions(rows,heads):
    result=[]
    for state in rows:
        if set(state)!={"sid","house","fields"}:raise ValueError("blind input contains non-observational fields")
        result.append(dict(sid=state["sid"],house=state["house"],
                           scores=[scores(state["fields"][i],heads[k]).tolist() for i,k in enumerate(MODES)]))
    return result


def report(rows,pred,heads,domains):
    pp={a["sid"]:a for a in pred};out={}
    assert len(pp)==len(rows)==len(pred)
    for i,mode in enumerate(MODES):
        cut=heads[mode]["cut"];parts={}
        for dom in ("all","official","derived"):
            subsets={}
            for group in ["all"]+sorted({a["house"] for a in rows}):
                yy=[];ss=[];any_hit=0;positive=0;fires=0
                for a in rows:
                    if group!="all" and group!=a["house"]:continue
                    k=pp[a["sid"]];assert a["house"]==k["house"]
                    valid,y=valid_targets(a)
                    mask=valid if dom=="all" else valid&(np.asarray(domains)==dom)
                    if not mask.any():continue
                    y=y[mask];score=np.asarray(k["scores"][i])[mask]
                    hit=np.zeros(len(y),bool) if cut is None else score>=cut
                    yy.extend(y);ss.extend(score);positive+=y.any();fires+=hit.any();any_hit+=(hit&y).any()
                if yy:
                    y=np.asarray(yy);s=np.asarray(ss);f=s>=cut if cut is not None else np.zeros(len(s),bool)
                    z=metrics(y,s,f);z.update(positive_starts=int(positive),prompted_starts=int(fires),true_prompt_starts=int(any_hit))
                    subsets[group]=z
            parts[dom]=subsets
        out[mode]=dict(threshold=cut,met_calibration_precision=cut is not None,metrics=parts)
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("fit","predict","evaluate"))
    parser.add_argument("--ground-checkpoint",type=Path)
    parser.add_argument("--train",type=Path)
    parser.add_argument("--calibration",type=Path)
    parser.add_argument("--observed",type=Path)
    parser.add_argument("--frozen",type=Path)
    parser.add_argument("--predictions",type=Path)
    parser.add_argument("--labels",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.mode in ("fit","predict"):
        if args.ground_checkpoint is None:parser.error("--ground-checkpoint required")
        ground,_,words=restore_heads(args.ground_checkpoint)
        def load(path,has_labels):
            data=torch.load(path,weights_only=True,map_location="cpu")
            if set(data)!={"anchors","queries","states"} or data["queries"]!=words:
                raise ValueError("unaligned vocabulary or unexpected cache fields")
            anchors=data["anchors"]
            if not torch.allclose(anchors,ground.anchors,atol=1e-6):raise ValueError("unaligned feature prototypes")
            rows=[]
            for x in data["states"]:
                required={"sid","house","code","anchor_sim"}|({"eligible"} if has_labels else set())
                if set(x)!=required:raise ValueError("unexpected state fields")
                row=dict(sid=x["sid"],house=x["house"],fields=observation_features(ground,x["code"],x["anchor_sim"],anchors))
                if has_labels:row["eligible"]=np.asarray(x["eligible"])
                rows.append(row)
            if len({s["sid"] for s in rows})!=len(rows):raise ValueError("duplicate origin")
            return rows
        if args.mode=="fit":
            tr=load(args.train,True);cal=load(args.calibration,True)
            write_new(args.output,dict(models=fit_heads(tr,cal),words=words,grounder=digest(args.ground_checkpoint)))
        else:
            obj=json.loads(args.frozen.read_text())
            if obj["grounder"]!=digest(args.ground_checkpoint) or obj["words"]!=words: raise ValueError("checkpoint mismatch")
            write_new(args.output,dict(checkpoint=digest(args.frozen),words=words,predictions=blind_predictions(load(args.observed,False),obj["models"])))
    else:
        obj=json.loads(args.frozen.read_text())
        pred=json.loads(args.predictions.read_text())
        labels=json.loads(args.labels.read_text())
        if pred["checkpoint"]!=digest(args.frozen) or labels["words"]!=obj["words"] or pred["words"]!=obj["words"]:
            raise ValueError("frozen model/labels/predictions differ")
        write_new(args.output,report(labels["rows"],pred["predictions"],obj["models"],labels["domains"]))


if __name__=="__main__":main()
