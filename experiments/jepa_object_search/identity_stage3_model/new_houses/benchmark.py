"""Score frozen live proposals with four verifiers on a previously unexamined house panel."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


def load(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def altered_pose(name,T):
    rng=np.random.default_rng(int(hashlib.sha256(("pose-error-v1:"+name).encode()).hexdigest()[:16],16))
    phi=rng.uniform(0,2*np.pi)
    yaw=np.deg2rad(rng.choice([-5.,5.]))
    rotation=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    new=T.copy()
    new[:3,:3]=rotation@T[:3,:3]
    new[:3,3]+=.15*np.array([np.cos(phi),0,np.sin(phi)])
    return new


def evaluate(project,out,stage,checkpoint_dir=None):
    private=out/"private"
    if checkpoint_dir is None:
        output_private,output_reports=private,out
    else:
        output_reports=out/"seed_checks"/checkpoint_dir.name
        output_private=output_reports/"private"
        output_private.mkdir(parents=True,exist_ok=True)
    frames={r["name"]:r for r in load(private/"frames.jsonl")}
    tasks=load(private/"cases.jsonl")
    sys.path[:0]=[str(project/n) for n in ("identity_stage3_model/train","jepa_reid_v21/association",
                "jepa_report_v19","jepa_reid_v23/report","jepa_reid_v21/detector","jepa_reid_v21/model")]
    from evaluate import label_boxes
    from appearance import WORK_WIDTH,build_reference,match_reference
    from report_evidence import eligible_boxes,current_support
    @lru_cache(maxsize=20)
    def image(fid):
        with Image.open(private/"observations"/(fid+".png")) as im:
            return np.asarray(im.convert("RGB"))
    dpath=private/"dino.jsonl"
    if stage=="detect":
        from proposal_worker import ProposalWorker
        worker=ProposalWorker(project/"b_verification/env/bin/python",project,device="cuda")
        try:
            with dpath.with_suffix(".tmp").open("w") as f:
                for fid,r in frames.items():
                    boxes=[b["bbox"] for b in worker.detect(image(fid),r["category"])]
                    f.write(json.dumps(dict(name=fid,boxes=boxes))+"\n")
        finally:
            worker.close()
        dpath.with_suffix(".tmp").replace(dpath)
        return
    boxes={r["name"]:r["boxes"] for r in load(dpath)}
    assert set(boxes)==set(frames)
    labs={fid:[b["instance"] if b else None for b in label_boxes(boxes[fid],r["boxes"])] for fid,r in frames.items()}
    def outcome(fid,idx,state,uid):
        if idx is None:
            return state
        who=labs[fid][idx]
        return "correct" if who==uid else "unverified" if who is None else "wrong"
    methods=("legacy","position","direct","jepa")
    def save(method,items):
        (output_private/(method+".jsonl")).write_text("".join(json.dumps(r)+"\n" for r in items))
    if stage=="legacy":
        cv2.setNumThreads(1)
        items=[]
        for t in tasks:
            s,q,uid=t["source"],t["query"],t["uid"]
            result="source_missing"
            if uid in labs[s]:
                ref=build_reference(image(s),boxes[s][labs[s].index(uid)])
                result="uncertain"
                if ref is not None:
                    live=image(q)
                    rois=eligible_boxes([{"bbox":b} for b in boxes[q]],live.shape)
                    cv2.setRNGSeed(7)
                    candidate,_=current_support(ref,match_reference(ref,live),live,rois)
                    if candidate is not None:
                        box=np.asarray(candidate)*live.shape[1]/WORK_WIDTH
                        who,=label_boxes([box],frames[q]["boxes"])
                        result="unverified" if who is None else "correct" if who["instance"]==uid else "wrong"
            items.append(dict(clean=result))
        save("legacy",items)
        return
    if stage=="learned":
        from identity import IdentityVerifier
        opts={"checkpoint_dir":checkpoint_dir} if checkpoint_dir is not None else {}
        direct=IdentityVerifier(project,"direct",**opts)
        jepa=IdentityVerifier(project,"jepa",**opts)
        @lru_cache(maxsize=None)
        def encode(fid,shift):
            T=np.asarray(frames[fid]["T"])
            if shift:
                clean=encode(fid,False)
                new=altered_pose(fid,T)
                dR=new[:3,:3]@T[:3,:3].T
                points=[None if p is None else new[:3,3]+dR@(np.asarray(p)-T[:3,3]) for p in clean["points"]]
                return dict(clean,points=points,c2w=new)
            with np.load(private/"observations"/(fid+".npz"),allow_pickle=False) as data:
                return direct.encode(image(fid),boxes[fid],data["depth"],np.asarray(frames[fid]["K"]),T)
        outputs=defaultdict(list)
        for t in tasks:
            s,q,uid=t["source"],t["query"],t["uid"]
            variants={method:{} for method in methods[1:]}
            for shift,tag in ((False,"clean"),(True,"pose_stress")):
                if uid not in labs[s]:
                    for m in variants:
                        variants[m][tag]="source_missing"
                    continue
                i=labs[s].index(uid)
                a,b=encode(s,shift),encode(q,shift)
                for model,method in ((direct,"direct"),(jepa,"jepa")):
                    scores=model.score(a,b)[i:i+1]
                    decision,=model.decide(scores)
                    variants[method][tag]=outcome(q,decision["index"],decision["state"],uid)
                visible=[(np.linalg.norm(np.asarray(p)-a["points"][i]),j) for j,p in enumerate(b["points"])
                         if p is not None and a["points"][i] is not None]
                best=min(visible,default=(float("inf"),None))
                idx=best[1] if best[0]<=.347302806383428 else None
                variants["position"][tag]=outcome(q,idx,"uncertain",uid)
            for method in variants:
                outputs[method].append(variants[method])
        for method,result in outputs.items():
            save(method,result)
        return
    results={m:load((private if m=="legacy" else output_private)/(m+".jsonl")) for m in methods}
    assert all(len(v)==len(tasks) for v in results.values())
    counts={}
    delta={}
    for kind in ("same","absent_with_same_class"):
        indices=[i for i,t in enumerate(tasks) if t["kind"]==kind]
        exposed=[i for i in indices if tasks[i]["uid"] in labs[tasks[i]["source"]] and
                 (tasks[i]["uid"] in labs[tasks[i]["query"]] if kind=="same" else any(labs[tasks[i]["query"]]))]
        for subgroup,ix in (("all",indices),("source_and_live_boxes",exposed)):
            key=kind+"/"+subgroup
            counts[key]={m:{tag:dict(Counter(results[m][i][tag] for i in ix)) for tag in results[m][0]} for m in methods}
        def hit(method,i):
            r=results[method][i]["clean"]
            return int(r=="correct") if kind=="same" else int(r in ("wrong","unverified"))
        for baseline in methods[:-1]:
            byhouse=defaultdict(list)
            for i in indices:
                byhouse[tasks[i]["house"]].append(hit("jepa",i)-hit(baseline,i))
            values=[np.mean(v) for v in byhouse.values()]
            boot=np.random.default_rng(7129).choice(values,size=(10000,len(values))).mean(1)
            delta[kind+"/"+baseline]=dict(houses=len(values),mean=float(np.mean(values)),ci95=np.quantile(boot,[.025,.975]).tolist())
    data=dict(paid_calls=0, metrics=counts,paired_house_macro_delta=delta,detection_sha256=fingerprint(dpath),
              panel_sha256=fingerprint(private/"cases.jsonl"),
              checkpoints={m:fingerprint((checkpoint_dir or project/"identity_stage3_model/train/private")/(m+".pt")) for m in ("direct","jepa")})
    (output_reports/"results.json").write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("completed",len(tasks),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("project",type=Path)
    ap.add_argument("directory",type=Path)
    ap.add_argument("stage",choices=["detect","legacy","learned","summary"])
    ap.add_argument("--checkpoint-dir",type=Path)
    a=ap.parse_args()
    evaluate(a.project.resolve(),a.directory.resolve(),a.stage,a.checkpoint_dir)
