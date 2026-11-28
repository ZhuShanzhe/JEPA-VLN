"""Fit from HSSD training assets; held-out houses are opened only after freezing."""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from placement_prior import CONTEXTS, categories


def room_records(assets, house, lookup, atlas_type):
    atlas = atlas_type(assets, house)
    scene = json.loads((assets/"scenes"/(house+".scene_instance.json")).read_text())
    instances = scene["object_instances"]
    points = np.array([item["translation"] for item in instances])
    rooms = atlas.locate(points, instance=True)
    words = []
    for item in instances:
        ids = re.findall(r"[a-fA-F0-9]{40}", item["template_name"])
        words.append(next((lookup[i] for i in ids if i in lookup), set()))
    records = []
    for rid, room_id in enumerate(atlas.names):
        ids = np.flatnonzero(rooms == rid)
        present = set().union(*(words[i] for i in ids))
        anchors = []
        poly = np.asarray(atlas.regions[rid]["poly_loop"])[:, [0,2]]
        for p,q in zip(poly, np.roll(poly,-1,axis=0)):
            length = np.linalg.norm(q-p)
            if length < 0.15:
                continue
            n = int(np.ceil(length/2.0))
            for j in range(n):
                anchors.append((0,p+(j+0.5)/n*(q-p),-1))
        for i in ids:
            for ci,context in enumerate(CONTEXTS[1:],start=1):
                if context in words[i]:
                    anchors.append((ci,points[i,[0,2]],i))
        local = []
        for ci,xz,source_id in anchors:
            neighbors = ids[(np.linalg.norm(points[ids][:,[0,2]]-xz,axis=1) <= 1.25)&(ids!=source_id)]
            if source_id >= 0:
                neighbors = neighbors[np.abs(points[neighbors,1]-points[source_id,1])<=2.0]
            local.append((ci,set().union(*(words[i] for i in neighbors))))
        records.append((int(room_id),present,local))
    return records


def counts(records, idx, nr):
    den = np.zeros(nr)
    pos = np.zeros((nr,len(idx)))
    local_den = np.zeros((nr,len(CONTEXTS)))
    local_pos = np.zeros((nr,len(CONTEXTS),len(idx)))
    for r,present,anchors in records:
        den[r] += 1
        pos[r,[idx[w] for w in present if w in idx]] += 1
        for c,near in anchors:
            local_den[r,c] += 1
            local_pos[r,c,[idx[w] for w in near if w in idx]] += 1
    return den,pos,local_den,local_pos


def brier(count,pos,p):
    return (pos*(1-p)**2 + (count-pos)*p**2).sum() / (count.sum()*p.shape[-1])


def main(project, assets, out):
    sys.path.insert(0,str(project/"jepa_env_v13/data"))
    from rooms import ROOMS, Atlas
    from train_prior import source
    splits = source(project)
    train = sorted(splits["train"][1]+splits["extra_train"][1])
    fresh = sorted(json.loads((project/"jepa_env_v13/simulator/cohort.json").read_text())["fresh"])
    assert len(train)==40 and len(fresh)==12 and not set(train)&set(fresh)
    with (assets/"metadata/fpmodels-with-decomposed.csv").open() as f:
        lookup = {r["id"]:categories(r) for r in csv.DictReader(f)}
    records = [item for house in train for item in room_records(assets,house,lookup,Atlas)]
    vocab = sorted(set.union(*[p for _,p,_ in records]))
    index = {w:i for i,w in enumerate(vocab)}
    n,x,m,y = counts(records,index,len(ROOMS))
    prevalence = (x.sum(0)+0.5)/(n.sum()+1)
    room_p = (x+4*prevalence)/(n[:,None]+4)
    global_local = (y.sum((0,1))+0.5)/(m.sum()+1)
    by_room = (y.sum(1)+12*global_local)/(m.sum(1)[:,None]+12)
    by_cue = (y.sum(0)+12*global_local)/(m.sum(0)[:,None]+12)
    p = (y+6*(by_room[:,None,:]+by_cue[None,:,:]))/(m[:,:,None]+12)
    targets = {w:dict(rooms_present=int(x[:,i].sum()),room=room_p[:,i].round(7).tolist(),
                      local=p[:,:,i].round(7).tolist()) for w,i in index.items()}
    record = dict(schema=1,rooms=ROOMS,contexts=CONTEXTS,targets=targets,train_houses=train,
                  train_rooms=int(n.sum()),train_anchors=int(m.sum()),
                  local_event="other instance within 1.25m horizontal of anchor in same annotated room; 2m vertical for furniture anchors; room-edge anchor samples are polygon edges, not asserted physical walls")
    out.mkdir(exist_ok=True,parents=True)
    path = out/"learned_prior.json"
    encoded=json.dumps(record,separators=(",",":"))
    if path.exists() and path.read_text()!=encoded:
        raise ValueError("existing frozen prior differs")
    path.write_text(encoded)
    fixed_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    per_house = {}
    # Macro category evaluation has a prespecified minimum training support.
    subset=x.sum(0)>=6
    for house in fresh:
        a,b,c,d=counts(room_records(assets,house,lookup,Atlas),index,len(ROOMS))
        a=a[:,None]; c=c[:,:,None]
        per_house[house]=dict(
            rooms=int(a.sum()),anchors=int(c.sum()),
            room_brier=brier(a,b[:,subset],room_p[:,subset]),
            prevalence_brier=brier(a,b[:,subset],prevalence[None,subset]),
            room_context_local_brier=brier(c,d[:,:,subset],p[:,:,subset]),
            room_only_local_brier=brier(c,d[:,:,subset],by_room[:,None,subset]))
    keys=["room_brier","prevalence_brier","room_context_local_brier","room_only_local_brier"]
    summary=dict(train_houses=len(train),train_rooms=int(n.sum()),train_anchors=int(m.sum()),
                 retained_classes=len(vocab),supported_eval_classes=int(subset.sum()),
                 eval_houses=len(fresh),eval_rooms=sum(row["rooms"] for row in per_house.values()),
                 eval_anchors=sum(row["anchors"] for row in per_house.values()),
                 house_macro={k:float(np.mean([row[k] for row in per_house.values()])) for k in keys},
                 split_overlap=0,prior_sha256=fixed_hash,per_house=per_house)
    (out/"placement_validation.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!="per_house"}))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for field in ("project","assets","out"):
        ap.add_argument(field,type=Path)
    args=ap.parse_args()
    main(args.project,args.assets,args.out)
