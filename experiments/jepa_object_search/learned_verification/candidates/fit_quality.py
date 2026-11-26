"""Train on HSSD train houses; tune the delivery threshold only on validation houses."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from source_quality import features,quality,overlap


def load(p):
    return list(map(json.loads,p.read_text().splitlines()))


def fit(rows):
    X=np.array([features(row,w,h) for _,_,w,h,row,_ in rows])
    y=np.array([gt for *_,gt in rows],float)
    mu=X.mean(0); std=X.std(0).clip(.001)
    x=np.c_[np.ones(len(y)),(X-mu)/std]
    beta=np.zeros(x.shape[1]); penalty=np.diag([0.]+[2.]*(x.shape[1]-1))
    for _ in range(30):
        p=1/(1+np.exp(-np.clip(x@beta,-35,35)))
        H=(x.T*(p*(1-p)))@x+penalty
        delta=np.linalg.solve(H,x.T@(p-y)+penalty@beta)
        beta-=delta
        if np.max(abs(delta))<1e-7:break
    return dict(mode="trained",mean=mu.tolist(),scale=std.tolist(),weights=beta[1:].tolist(),bias=float(beta[0]))


def frame_scores(frames,rows,spec):
    scores=defaultdict(list)
    for fid,_,w,h,row,g in rows:
        scores[fid].append((quality(spec,row,w,h),g))
    return [(r,max(scores[r["frame_id"]],key=lambda k:k[0],default=(-1.,False))) for r in frames]


def metrics(scores,t):
    offered=[(r,s) for r,s in scores if s[0]>=t]
    tp=sum(int(s[1]) for r,s in offered)
    pos=sum(bool(r["boxes"]) for r,_ in scores)
    nneg=len(scores)-pos
    return dict(frames=len(scores),positive_frames=pos,empty_frames=nneg,offered=len(offered),
        localized=tp,false_offer=len(offered)-tp,empty_frames_offered=sum(not r["boxes"] for r,_ in offered),
        precision=tp/max(1,len(offered)),recall_per_positive=tp/max(1,pos))


def paired(scores):
    groups=defaultdict(list)
    for r,s in scores:
        groups[(r["house"],r["category"])].append((r,s))
    out=dict(paired_windows=0,latest_hits=0,score_hits=0,positive_windows=0,detector_available_windows=0)
    for g in groups.values():
        # Each 4-frame window supplies exactly one slot to both policies.
        for j in range(0,len(g)-3,4):
            window=g[j:j+4]; out["paired_windows"]+=1
            out["positive_windows"]+=int(any(r["boxes"] for r,s in window))
            out["latest_hits"]+=int(window[-1][1][1])
            out["score_hits"]+=int(max(window,key=lambda k:k[1][0])[1][1])
            out["detector_available_windows"]+=int(any(s[0]>=0 for _,s in window))
    return out


def main(root,dst):
    data=load(root/"data/frames.jsonl")
    p={r["frame_id"]:r["boxes"] for r in load(root/"detector/predictions.jsonl")}
    assert len(data)==len(p) and all(r["frame_id"] in p for r in data)
    partitions={k:[r for r in data if r["split"]==k] for k in ("train","val","test")}
    houses={k:set(r["house"] for r in v) for k,v in partitions.items()}
    assert len(set.union(*houses.values()))==sum(map(len,houses.values()))
    rows={}
    for k,v in partitions.items():
        rows[k]=[(r["frame_id"],r["category"],384,384,b,max((overlap(b["xyxy"],t["xyxy"]) for t in r["boxes"]),default=0)>=.5)
                 for r in v for b in p[r["frame_id"]]]
    specs=[dict(mode="detector"),fit(rows["train"])]
    report={"split_houses":{k:len(v) for k,v in houses.items()},"train_boxes":len(rows["train"]),"methods":{}}
    candidates=[]
    for spec in specs:
        val=frame_scores(partitions["val"],rows["val"],spec)
        options=[(metrics(val,t),float(t)) for t in sorted(set(s for r,(s,_) in val if s>=0))]
        # At least 80% true localization precision on validation, then maximize true coverage.
        eligible=[v for v in options if v[0]["precision"]>=.80]
        chosen=max(eligible or options,key=lambda mt:(mt[0]["localized"] if eligible else mt[0]["precision"],mt[0]["precision"],mt[1]))
        spec["threshold"]=chosen[1]
        candidates.append((chosen[0]["localized"] if eligible else 0,chosen[0]["precision"],spec))
        test=frame_scores(partitions["test"],rows["test"],spec)
        report["methods"][spec["mode"]]=dict(validation=chosen[0],test=metrics(test,spec["threshold"]),
            equal_one_slot_per_window=paired(test),target_categories={k:metrics([(r,s) for r,s in test if r["category"]==k],
                spec["threshold"]) for k in ("wall_art","clock")})
    selected=max(candidates,key=lambda v:(v[0],v[1],v[2]["mode"]=="detector"))[2]
    report["selected"]=selected["mode"]
    report["control"]="same frozen proposals; source-slot evaluation is an offline grouping by house/category, not robot trajectories"
    for name,val in [("selector.json",selected),("trained_selector.json",specs[1]),("hssd_quality.json",report)]:
        (dst/name).write_text(json.dumps(val,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root",type=Path)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    main(args.data_root,args.output_dir)
