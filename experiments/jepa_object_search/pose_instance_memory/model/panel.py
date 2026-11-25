"""Replay frozen detections and all three camera pairs in each held-out scene clip."""
import argparse
from collections import Counter,defaultdict
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from pose_inference import load_head,score_matrix,unique_links
from separation import soft_separation


@torch.no_grad()
def evaluate(args):
    project=args.project
    root=project/"jepa_reid_v21"
    sys.path.insert(0,str(root/"association"))
    from geometry_links import match_views,surface_point
    frames=args.frames or root/"data/frames.jsonl"
    predictions=args.predictions or root/"detector/predictions.jsonl"
    cache=args.cache or root/"model/proposals.pt"
    outcome=args.out or root/"model/panel_results.json"
    all_frames=[json.loads(s) for s in frames.read_text().splitlines()]
    pred={r["frame_id"]:r["boxes"] for s in predictions.read_text().splitlines() if (r:=json.loads(s))}
    cached=torch.load(cache,map_location="cpu",weights_only=True)
    import hashlib
    for source in [frames,predictions]:
        assert cached["hashes"][source.name]==hashlib.sha256(source.read_bytes()).hexdigest()
    fs=cached["features"].numpy()
    proposals=defaultdict(list)
    for k,r in enumerate(cached["records"]):
        if not r["jitter"]:
            proposals[r["frame"]].append((k,r))
    models={}
    thresholds=json.loads((root/"model/pose_results.json").read_text())["methods"]
    for tag in thresholds:
        if tag=="location":
            models[tag]=(None,tag,thresholds[tag]["thresholds"]["detected"])
        else:
            models[tag]=load_head(root/"model"/(tag+".pt"),"cpu")
    methods=["geometric_rule","location","heading_jepa","pose_forecast","pose_direct","pose_jepa","geometry_then_direct","geometry_then_jepa","geometry_jepa_separation_probe"]
    near_limits={}
    if args.near_review:
        frozen=json.loads((root/"hard_cases/summary.json").read_text())["methods"]
        near_limits={m:frozen[m]["global_and_near_5pct"]["threshold"] for m in ("pose_direct","pose_jepa")}
        methods+=["geometry_then_direct_near_review","geometry_then_jepa_near_review"]
    torch.set_num_threads(2)
    sequences=defaultdict(list)
    for fr in all_frames:
        if fr["split"]=="test":
            sequences[fr["sequence"]].append(fr)
    subtotals=defaultdict(lambda:defaultdict(Counter))
    by_house=defaultdict(lambda:defaultdict(Counter))
    exclusive=Counter()
    for group in sequences.values():
        group.sort(key=lambda r:r["view"])
        entries=[]
        for fr in group:
            fid=fr["frame_id"]
            selection=proposals[fid]
            indices=[k for k,r in selection]
            entries.append(dict(frame_id=fid,features=fs[indices],K=fr["K"],c2w=fr["T_world_from_camera"],
                boxes=[pred[fid][r["slot"]]["xyxy"] for k,r in selection],
                points=[r["point"] for k,r in selection],
                labels=[r["instance"] for k,r in selection],truth=set(cached["labels"][fid])))
            view=entries[-1]
            depth=np.load(frames.parent/fr["depth"])
            view["image_size"]=(depth.shape[1],depth.shape[0])
            view["rule_points"]=[surface_point(depth,box,view["K"],view["c2w"]) for box in view["boxes"]]
        together={frozenset((a,b)) for e in entries for a,b in itertools.combinations(e["truth"],2)}
        for l,r in itertools.combinations(entries,2):
            rule=match_views(dict(l,points=l["rule_points"]),dict(r,points=r["rule_points"]))
            separate=soft_separation(l,r)
            for i,a in enumerate(l["labels"]):
                for j,b in enumerate(r["labels"]):
                    item="unverified" if a is None or b is None else ("same" if a==b else
                          ("different_seen_together" if frozenset((a,b)) in together else "different_never_seen_together"))
                    exclusive[item+"_candidates"]+=1
                    exclusive[item+"_flagged"]+=int((i,j) in separate)
            candidate={tag:score_matrix(model,mode,l,r) for tag,(model,mode,t) in models.items()}
            for tag in methods:
                if tag=="geometric_rule":
                    links=rule["matches"]
                else:
                    short=tag.removesuffix("_near_review")
                    m={"geometry_then_direct":"pose_direct","geometry_then_jepa":"pose_jepa",
                       "geometry_jepa_separation_probe":"pose_jepa"}.get(short,short)
                    evidence=rule if tag.startswith("geometry_") else None
                    if tag=="geometry_jepa_separation_probe":
                        # Existing mutually corroborated links take precedence over an experimental hint.
                        evidence=dict(rule,distinct=list(set(rule["distinct"]) | (separate-set(rule["matches"]))))
                    score=candidate[m]
                    if tag.endswith("_near_review") and score.size:
                        dist=np.linalg.norm(np.asarray(l["points"])[:,None,:]-np.asarray(r["points"])[None,:,:],axis=-1)
                        score=np.where((dist<=.60)&(score<near_limits[m]),-np.inf,score)
                    links=unique_links(score,models[m][2],evidence)
                c=Counter(frame_pairs=1,visible_same=len(l["truth"]&r["truth"]),
                          detectable_same=len((set(l["labels"])&set(r["labels"]))-{None}),links=len(links))
                labels_seen=set()
                for i,j in links:
                    a,b=l["labels"][i],r["labels"][j]
                    if a is None or b is None:
                        c["unverified"]+=1
                    elif a==b:
                        c["correct"]+=1
                        labels_seen.add(a)
                    else:
                        c["wrong"]+=1
                assert c["correct"]==len(labels_seen)
                category=group[0]["category"]
                for domain in ("all","category_"+category,"stratum_"+group[0]["stratum"]):
                    subtotals[tag][domain].update(c)
                by_house[tag][group[0]["house"]].update(c)
                if category!="bookcase":
                    subtotals[tag]["excluding_bookcase"].update(c)
    ans={"protocol":"all view combinations; one-to-one links; independently localized IoU >= 0.5; unresolved background proposals have unknown identity",
         "houses":len({fr["house"] for group in sequences.values() for fr in group}),"sequences":len(sequences),
         "soft_separation_diagnostic":dict(exclusive),
         "methods":{tag:{name:{k:n[k] for k in ("frame_pairs","visible_same","detectable_same","links","correct","wrong","unverified")}
                        for name,n in d.items()} for tag,d in subtotals.items()}}
    houses=sorted(by_house["geometry_then_jepa"])
    resample=np.random.default_rng(12511).integers(len(houses),size=(5000,len(houses)))
    contrasts={}
    for other in ("geometry_then_direct","geometric_rule","heading_jepa"):
        diff=np.array([by_house["geometry_then_jepa"][h]["correct"]-by_house[other][h]["correct"] for h in houses])
        den=np.array([by_house[other][h]["visible_same"] for h in houses])
        keep=den[resample].sum(1)>0
        if not den.sum():
            continue
        ci=np.quantile(diff[resample][keep].sum(1)/den[resample][keep].sum(1),[.025,.975])
        contrasts[other]=[float(diff.sum()/den.sum()),*ci.tolist()]
    ans["joint_minus_reference_visible_recall_delta_ci95"]=contrasts
    outcome.write_text(json.dumps(ans,indent=2)+"\n")
    print(json.dumps({k:v["all"] for k,v in ans["methods"].items()}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    for key in ("frames","predictions","cache","out"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--near-review",action="store_true")
    evaluate(p.parse_args())
