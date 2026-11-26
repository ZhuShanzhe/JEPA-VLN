"""Frozen legacy, directly supervised and JEPA comparison on identical archived proposals."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def build(project):
    rt = project/"identity_stage3_audit/revisit/private"
    twin = project/"jepa_reid_v21/data/controls"
    frames = {}
    revisit_groups, layouts = defaultdict(dict), defaultdict(list)
    fp = [rt/"frames.jsonl",rt/"dino.jsonl",twin/"frames.jsonl",project/"jepa_reid_v21/detector/controls/predictions.jsonl"]
    rpred={r["name"]:r["boxes"] for r in rows(fp[1])}
    for fr in rows(fp[0]):
        key="r/"+fr["name"]
        frames[key]=dict(rgb=rt/"observations"/(fr["name"]+".png"), depth=rt/"observations"/(fr["name"]+".npz"),
                         K=fr["K"],c2w=fr["T_world_from_camera"],truth=fr["boxes"],boxes=rpred[fr["name"]])
        revisit_groups[fr["group"]][fr["stage"]]=(key,fr)
    tpred={r["frame_id"]:[b["xyxy"] for b in r["boxes"]] for r in rows(fp[3])}
    for fr in rows(fp[2]):
        key="t/"+fr["frame_id"]
        frames[key]=dict(rgb=twin/fr["rgb"],depth=twin/fr["depth"],K=fr["K"],c2w=fr["T_world_from_camera"],
                         truth=fr["boxes"],boxes=tpred[fr["frame_id"]])
        layouts[fr["layout"]].append((key,fr))
    tasks=[]
    for group, views in sorted(revisit_groups.items()):
        src,first=views["anchor"]
        for view in ("closer","same_side","opposite_side"):
            if view not in views:
                continue
            live,cur=views[view]
            tasks.append(dict(panel="revisit",unit=group,house=cur["house"],category=cur["category"],source=src,
                              live=live,view=view,uids=[first["uid"]],visible=any(b["instance"]==first["uid"] for b in cur["boxes"])))
    for group, views in sorted(layouts.items()):
        for src,first in sorted(views):
            if first["stratum"]!="double" or len(first["boxes"])!=2:
                continue
            for live,cur in sorted(views):
                if cur["stratum"]!="single" or len(cur["boxes"])!=1:
                    continue
                original=cur["boxes"][0]["instance"]
                if "synthetic" in original or original not in {b["instance"] for b in first["boxes"]}:
                    continue
                copy,=[b for b in first["boxes"] if b["instance"]!=original]
                assert "synthetic" in copy["instance"] and copy["asset"]==cur["boxes"][0]["asset"]
                tasks.append(dict(panel="twins",unit=group,house=cur["house"],category=cur["category"],source=src,live=live,
                                  view="same_camera" if first["view"]==cur["view"] else "changed_camera",
                                  uids=[original,copy["instance"]],visible=True))
    assert Counter(t["panel"] for t in tasks)=={"revisit":97,"twins":82}
    protocol=dict(counts=dict(Counter(t["panel"] for t in tasks)),files_sha256=[sha(p) for p in fp],
                  cases_sha256=digest(tasks),detection_input="Identical unfiltered frozen DINO proposals in all three arms",
                  registration="First source proposal uniquely matched to designated instance at IoU>=0.50, or abstain",
                  inputs="Learned arms: identical RGB/boxes, depth and calibrated simulation camera poses. Legacy: RGB/boxes.",
                  primary_metrics=["revisit correct original / target-visible queries","twins original accepted / copy withheld (including uncertain)",
                                   "twins original accepted / copy explicitly different","twins copy falsely accepted as present original"],
                  units="Repeated observations of physical objects; paired bootstrap with equal house weights",
                  no_scores_used_for_selection=True,paid_calls=0)
    return frames,tasks,protocol


def evaluate(project,frames,tasks,mode,device,verifier=None,encoded_frames=None):
    sys.path[:0]=[str(project/n) for n in ("identity_stage3_model/train","jepa_report_v19","jepa_reid_v23/report","jepa_reid_v21/association")]
    from evaluate import label_boxes
    from appearance import build_reference,match_reference,WORK_WIDTH
    from report_evidence import eligible_boxes,current_support
    model=verifier
    if mode!="legacy" and model is None:
        from identity import IdentityVerifier
        model=IdentityVerifier(project,mode=mode,device=device)
    cv2.setNumThreads(1)
    @lru_cache(maxsize=24)
    def image(k):
        return cv2.cvtColor(cv2.imread(str(frames[k]["rgb"])),cv2.COLOR_BGR2RGB)
    @lru_cache(maxsize=None)
    def labels(key):
        fr=frames[key]
        return [v["instance"] if v is not None else None for v in label_boxes(fr["boxes"],fr["truth"])]
    @lru_cache(maxsize=None)
    def encode(k):
        if encoded_frames is not None and k in encoded_frames:
            return encoded_frames[k]
        fr=frames[k]
        if fr["depth"].suffix==".npz":
            with np.load(fr["depth"],allow_pickle=False) as z:
                depth=z["depth"]
        else:
            depth=np.load(fr["depth"],allow_pickle=False)
        vec=model.encode(image(k),fr["boxes"],depth,np.asarray(fr["K"]),np.asarray(fr["c2w"]))
        if encoded_frames is not None:
            encoded_frames[k]=vec
        return vec
    @lru_cache(maxsize=None)
    def scored(src,live):
        return model.score(encode(src),encode(live))
    def one(src,live,uid):
        sl,cl=labels(src),labels(live)
        exposed=uid in sl
        result=dict(state="uncertain",id=None,source_box=exposed,usable_current=uid in cl)
        if not exposed or not frames[live]["boxes"]:
            return result
        idx=sl.index(uid)
        if mode=="legacy":
            old=build_reference(image(src),frames[src]["boxes"][idx])
            now=image(live)
            if old is None:
                return result
            candidates=eligible_boxes([{"bbox":b} for b in frames[live]["boxes"]],now.shape)
            cv2.setRNGSeed(7)
            found,_=current_support(old,match_reference(old,now),now,candidates)
            if found is not None:
                truth,=label_boxes([np.asarray(found)*now.shape[1]/WORK_WIDTH],frames[live]["truth"])
                result.update(state="same",id=None if truth is None else truth["instance"])
        else:
            mat=scored(src,live)
            assert mat.shape==(len(frames[src]["boxes"]),len(frames[live]["boxes"]))
            decision,=model.decide(mat[idx:idx+1])
            state=decision["state"]
            assert state in ("same","different","uncertain")
            result["state"]=state
            if state=="same":
                assert decision["index"] is not None
                result["id"]=cl[int(decision["index"])]
        return result
    result=[]
    for i,t in enumerate(tasks):
        pred=[one(t["source"],t["live"],uid) for uid in t["uids"]]
        for entry in pred:
            found=entry.pop("id")
            entry["outcome"]=entry["state"] if entry["state"]!="same" else "unverified" if found is None else "original" if found==t["uids"][0] else "other_instance"
        result.append(dict(case=i, panel=t["panel"],category=t["category"],house=t["house"],unit=t["unit"],view=t["view"],visible=t["visible"],pred=pred))
    return result


def summarize(rr):
    metrics={}
    def subset(name,ss):
        output=dict(n=len(ss),houses=len({r["house"] for r in ss}),units=len({r["unit"] for r in ss}))
        output["original"]=dict(Counter(r["pred"][0]["outcome"] for r in ss))
        if ss and ss[0]["panel"]=="twins":
            output["copy"]=dict(Counter(r["pred"][1]["outcome"] for r in ss))
            output["joint_withheld"]=sum(r["pred"][0]["outcome"]=="original" and r["pred"][1]["state"]!="same" for r in ss)
            output["joint_explicit_different"]=sum(r["pred"][0]["outcome"]=="original" and r["pred"][1]["state"]=="different" for r in ss)
        metrics[name]=output
    r=[x for x in rr if x["panel"]=="revisit"]
    visible=[x for x in r if x["visible"]]
    subset("visible",visible)
    subset("visible_with_both_boxes",[x for x in visible if x["pred"][0]["source_box"] and x["pred"][0]["usable_current"]])
    for stage in ("closer","same_side","opposite_side"):
        subset("visible_"+stage,[x for x in visible if x["view"]==stage])
    for cat in ("clock","wall_art"):
        subset("visible_"+cat,[x for x in visible if x["category"]==cat])
    subset("not_annotated",[x for x in r if not x["visible"]])
    twins=[x for x in rr if x["panel"]=="twins"]
    subset("twins",twins)
    subset("twins_changed",[x for x in twins if x["view"]=="changed_camera"])
    subset("twins_fully_boxed",[x for x in twins if all(p["source_box"] for p in x["pred"]) and x["pred"][0]["usable_current"]])
    return metrics


def paired(left,right):
    rng=np.random.default_rng(260127)
    panels={}
    for group in ("visible_correct","twins_joint","twins_copy_false_accept"):
        changes=defaultdict(list)
        for a,b in zip(left,right):
            assert (a["case"],a["house"])==(b["case"],b["house"])
            if group=="visible_correct":
                if a["panel"]!="revisit" or not a["visible"]: continue
                key=lambda r:r["pred"][0]["outcome"]=="original"
            else:
                if a["panel"]!="twins": continue
                if group=="twins_joint":
                    key=lambda r:r["pred"][0]["outcome"]=="original" and r["pred"][1]["state"]!="same"
                else:
                    key=lambda r:r["pred"][1]["outcome"]=="original"
            changes[a["house"]].append(int(key(b))-int(key(a)))
        v=np.array([np.mean(vals) for vals in changes.values()])
        draws=rng.choice(v,size=(10000,len(v))).mean(axis=1)
        panels[group]=dict(houses=len(v),house_macro_delta=float(v.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())
    return panels


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("project","output"):
        p.add_argument(arg,type=Path)
    p.add_argument("--mode",choices=["freeze","legacy","jepa","direct","combine"],required=True)
    p.add_argument("--device",default="cuda")
    a=p.parse_args()
    frames,tasks,protocol=build(a.project)
    a.output.mkdir(exist_ok=True,parents=True)
    fixed=a.output/"protocol.json"
    if a.mode=="freeze":
        if fixed.exists():assert json.loads(fixed.read_text())==protocol
        fixed.write_text(json.dumps(protocol,indent=2)+"\n")
    else:
        assert json.loads(fixed.read_text())==protocol
        private=a.output/"private"
        private.mkdir(exist_ok=True)
        if a.mode=="combine":
            rs={m:rows(private/(m+".jsonl")) for m in ("legacy","direct","jepa")}
            assert all(len(q)==len(tasks) for q in rs.values())
            summary={k:summarize(rr) for k,rr in rs.items()}
            assert summary["legacy"]["visible"]["original"]["original"]==27 and summary["legacy"]["twins"]["joint_withheld"]==21
            result=dict(paid_calls=0,methods=summary,paired_deltas={k+"_to_jepa":paired(rs[k],rs["jepa"]) for k in ("legacy","direct")},
                        checkpoint_sha256={k:sha(a.project/("identity_stage3_model/train/private/"+k+".pt")) for k in ("direct","jepa")},protocol_sha256=sha(fixed),
                        script_sha256=sha(Path(__file__)))
            (a.output/"results.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
        else:
            data=evaluate(a.project,frames,tasks,a.mode,a.device)
            metrics=summarize(data)
            if a.mode=="legacy":
                assert metrics["visible"]["original"]["original"]==27 and metrics["twins"]["joint_withheld"]==21 and metrics["twins"]["copy"]["original"]==19
            temp=private/(a.mode+".tmp")
            temp.write_text("".join(json.dumps(r)+"\n" for r in data))
            temp.replace(private/(a.mode+".jsonl"))
            print(a.mode,metrics["visible"],metrics["twins"])
