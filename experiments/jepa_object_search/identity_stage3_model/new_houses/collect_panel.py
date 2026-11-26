"""Preselect held-out house families and acquire multi-instance, multi-view tests in Habitat."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import yaml


def stable(*fields):
    return hashlib.sha256(json.dumps(["independent-id-panel-v1", *fields],sort_keys=True).encode()).hexdigest()


def run(project, assets, out):
    sys.path[:0] = [str(project / n) for n in ("jepa_reid_v20/collect","jepa_reid_v20/model","jepa_reid_v21/data")]
    import collect_views
    from build_frames import CLOCKS, K, annotations, open_sim
    from replay_depth import observe
    from habitat_sim.utils.common import quat_rotate_vector
    collect_views.WORDS.update(CLOCKS)
    old = json.loads((project/"jepa_reid_v20/data/sampling_protocol.json").read_text())["houses"]
    used = {h.split("_")[0] for group in old.values() for h in group}
    splits = yaml.safe_load((assets/"scene_splits.yaml").read_text())
    with (assets/"metadata/fpmodels-with-decomposed.csv").open() as f:
        table = {r["id"]:(r["main_wnsynsetkey"] or r["wnsynsetkey"]).split(".")[0] for r in csv.DictReader(f)}
    def category(sub):
        return "clock" if sub in CLOCKS else sub
    available = {}
    for house in map(str,splits["val"]):
        if house.split("_")[0] in used:
            continue
        data = json.loads((assets/"scenes"/(house+".scene_instance.json")).read_text())
        count = Counter()
        for obj in data["object_instances"]:
            ids = re.findall(r"[a-fA-F0-9]{40}",obj["template_name"])
            label = next((category(table[k]) for k in ids if k in table),None)
            if label in ("clock","wall_art"):
                count[label] += 1
        if count:
            available[house] = count
    ordered = sorted(available,key=lambda h:stable("house",h))
    houses = []
    for candidates, limit in (([h for h in ordered if available[h]["clock"]],6),(ordered,12)):
        for h in candidates:
            if len(houses)>=limit:
                break
            if h.split("_")[0] not in {x.split("_")[0] for x in houses}:
                houses.append(h)
    assert houses and not ({h.split("_")[0] for h in houses} & used)
    out.mkdir(parents=True,exist_ok=True)
    manifest = dict(houses=houses,excluded_families=sorted(used),max_objects_per_class=3,max_views=3,
                    requested_houses=12,eligible=len(available),class_order=["clock","wall_art"])
    config = out/"frozen.json"
    if config.exists() and json.loads(config.read_text())!=manifest:
        raise RuntimeError("frozen independent panel changed")
    config.write_text(json.dumps(manifest,sort_keys=True))
    frames=[]
    audit=[]
    for house in houses:
        shard=out/(house+".json")
        if not shard.exists():
            sim=open_sim(assets,house)
            part=[]
            counters=[]
            try:
                groups=defaultdict(dict)
                for sub,things in collect_views.inventory(sim,assets).items():
                    for sid,asset,center in things:
                        cat=category(sub)
                        if cat in ("clock","wall_art"):
                            groups[cat][sid]=dict(instance=f"{house}:{sid}",asset=asset,world_center=center.tolist())
                for cat,inventory in sorted(groups.items()):
                    for sid in sorted(inventory,key=lambda i:stable(house,cat,i))[:3]:
                        item=inventory[sid]
                        center=np.array(item["world_center"])
                        rng=np.random.default_rng(int(stable(house,cat,sid,"render")[:12],16))
                        offset=rng.uniform(0,2*math.pi)
                        previous=[]
                        for i in range(48):
                            angle=offset + i*math.pi*(3-math.sqrt(5))
                            r=rng.uniform(1.1,3.3)
                            trial=center+[r*math.cos(angle),-.75,r*math.sin(angle)]
                            floor=np.asarray(sim.pathfinder.snap_point(trial))
                            if not np.isfinite(floor).all() or abs(floor[1]-center[1])>2.9:
                                continue
                            eye=floor+[0,1.25,0]
                            d=center-eye
                            radius=math.hypot(d[0],d[2])
                            if not .85<=radius<=3.8:
                                continue
                            bearing=math.degrees(math.atan2(d[2],d[0]))
                            if any(abs((bearing-a+180)%360-180)<15 or np.linalg.norm(floor-p)<.35 for a,p in previous):
                                continue
                            pose=dict(camera_xyz=eye.tolist(),camera_heading_deg=bearing+rng.uniform(-5,5),
                                      pitch_deg=math.degrees(math.atan2(d[1],radius))+rng.uniform(-2,2))
                            obs,state=observe(sim,pose)
                            boxes=annotations(obs["ids"],inventory)
                            if not any(b["instance"]==item["instance"] for b in boxes):
                                continue
                            frame_id=stable(house,sid,i,"frame")[:24]
                            T=np.eye(4)
                            T[:3,:3]=np.column_stack([quat_rotate_vector(state.rotation,v) for v in np.eye(3)])@np.diag([1.,-1.,-1.])
                            T[:3,3]=state.position
                            folder=out/"observations"
                            folder.mkdir(exist_ok=True)
                            Image.fromarray(obs["rgb"][:,:,:3]).save(folder/(frame_id+".png"))
                            np.savez_compressed(folder/(frame_id+".npz"),depth=obs["depth"].astype("float32"),
                                                semantic=obs["ids"].astype("int32"))
                            all_ids=set(map(int,np.unique(obs["ids"])))
                            part.append(dict(name=frame_id,house=house,uid=item["instance"],category=cat,view=len(previous),
                                             boxes=boxes,K=K,T=T.tolist(),present=[o["instance"] for k,o in inventory.items() if k in all_ids]))
                            previous.append((bearing,floor.copy()))
                            if len(previous)==3:
                                break
                        counters.append(dict(category=cat,views=len(previous)))
            finally:
                sim.close()
            temp=shard.with_suffix(".tmp")
            temp.write_text(json.dumps(dict(frames=part,attempts=counters),allow_nan=False))
            temp.replace(shard)
        data=json.loads(shard.read_text())
        frames.extend(data["frames"])
        audit.extend(data["attempts"])
        print("completed house",len(data["frames"]),flush=True)
    assert len({r["name"] for r in frames})==len(frames)
    (out/"frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in frames))
    cases=[]
    by_target=defaultdict(list)
    for r in frames:
        by_target[r["house"],r["category"],r["uid"]].append(r)
    for (house,cat,uid), seq in sorted(by_target.items()):
        anchor=min(seq,key=lambda r:r["view"])
        for other in seq:
            if other!=anchor:
                cases.append(dict(source=anchor["name"],query=other["name"],kind="same",uid=uid,house=house,category=cat))
        negatives=[r for r in frames if r["house"]==house and r["category"]==cat and uid not in r["present"]]
        for r in sorted(negatives,key=lambda r:stable(uid,r["name"],"absent"))[:2]:
            cases.append(dict(source=anchor["name"],query=r["name"],kind="absent_with_same_class",uid=uid,house=house,category=cat))
    assert len({(r["source"],r["query"]) for r in cases})==len(cases)
    index=out/"cases.jsonl"
    index.write_text("".join(json.dumps(r)+"\n" for r in cases))
    counts=dict(assigned_houses=len(houses),houses_with_frames=len({r["house"] for r in frames}),
                physical_instances=len(by_target),instances_attempted=len(audit),views_per_attempt=dict(Counter(r["views"] for r in audit)),
                frames=len(frames),views_by_goal=dict(Counter(r["category"] for r in frames)),
                cases=dict(Counter(r["kind"] for r in cases)),sample_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
                source_families_excluded=len(used),overlapping_training_development_house_families=0)
    print(counts)
    (out.parent/"collection.json").write_text(json.dumps(counts,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for arg in ("project","assets","out"):
        ap.add_argument(arg,type=Path)
    a=ap.parse_args()
    run(a.project,a.assets,a.out)
