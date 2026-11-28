"""Offline review on the three observed RGBs at each executed inspection grant. No API."""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PRIVATE=HERE/"private_dino"
CAMERAS=("right","center","left")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def render():
    from visibility_review import configured_sim, hs, quat_from_coeffs, lines, load_scene_instances, _load_goal_sets, DATA, instance_lut
    sys.path.insert(0,str(ROOT/"local_target_20261128/eval"))
    from run_local import sensor_sim
    panel=read(ROOT/"local_target_20261128/eval_family/panel.json")
    report=read(HERE/"accepted.json")
    catalogue=[]
    for i,spec in enumerate(panel):
        row=read(HERE/"runs"/f"{i}_committed.json")
        path=HERE/"private_events"/f"{i}.poses.jsonl"
        eventpath=HERE/"private_events"/f"{i}.jsonl"
        assert digest(path)==row["private_trace_sha256"] and digest(eventpath)==row["private_events_sha256"]
        grants=[e for e in lines(eventpath) if e.get("event")=="acquired"]
        assert len(grants)==report["by_house"][i]["mechanisms"]["acquired"]
        if not grants:
            continue
        shard=DATA/"eval/hssdov/content"/(spec["scene"]+".json.gz")
        assert digest(shard)==spec["shard_sha256"]
        with gzip.open(shard) as f:
            ep=next(x for x in json.load(f)["episodes"] if str(x["episode_id"])==str(spec["id"]))
        states=lines(path)
        rgb= sensor_sim(ep)
        sem= configured_sim(ep)
        try:
            classes=_load_goal_sets(DATA)["scoring_class_ids"]["hssdov"][ep["object_category"]]
            targets=[r for r in load_scene_instances(DATA/"protocol/instance_viewpoints/hssd","val",ep["scene_id"])
                     if r["class_id"] in classes and r["view_points"]]
            lut=instance_lut(sem,"hssd",DATA/"vocab")
            assert targets and all(lut[t["semantic_id"]]==t["class_id"]+1 for t in targets)
            for event in grants:
                end=event["step"]
                offsets=(end-3,end-4,end-1)
                assert [states[j]["action"] for j in range(end-3,end+1)]==["turn_right","turn_left","turn_left","turn_right"]
                for view,j in zip(CAMERAS,offsets):
                    st=states[j]
                    state=hs.AgentState()
                    state.position=np.float32(st["xyz"])
                    state.rotation=quat_from_coeffs(np.float64(st["quat_xyzw"]))
                    for sim in (rgb,sem):
                        sim.get_agent(0).set_state(state,reset_sensors=True)
                    image=np.asarray(rgb.get_sensor_observations()["rgb"])[:,:,:3]
                    mask=np.asarray(sem.get_sensor_observations()["instance_mask"])
                    assert image.shape[:2]==mask.shape==(192,192)
                    name=f"{i}_{event['goal_id']}_{view}"
                    Image.fromarray(image).save(PRIVATE/(name+".png"))
                    target_mask=np.where(np.isin(mask,[r["semantic_id"] for r in targets]),mask,0).astype(np.int32)
                    np.save(PRIVATE/(name+".npy"),target_mask)
                    catalogue.append(dict(id=name,goal=row["goal"],group=f"{i}_{event['goal_id']}",view=view,
                                          rgb_sha256=digest(PRIVATE/(name+".png"))))
        finally:
            sem.close()
            rgb.close()
    assert len(catalogue)==12 and len({x["group"] for x in catalogue})==4
    save(PRIVATE/"inputs.json",catalogue)


def infer():
    detector=ROOT/"jepa_reid_v21/detector"
    sys.path.insert(0,str(detector))
    from proposals import LocalCandidateDetector, select
    cfg=read(detector/"protocol.json")
    assert (cfg["box_threshold"],cfg["text_threshold"],cfg["nms_iou"],cfg["selected_mode"])==(.35,.25,.85,"augment")
    model=LocalCandidateDetector(ROOT,device=cfg["device"],mode=cfg["selected_mode"])
    rows=[]
    for x in read(PRIVATE/"inputs.json"):
        p=PRIVATE/(x["id"]+".png")
        assert digest(p)==x["rgb_sha256"]
        raw=model.proposals(np.asarray(Image.open(p).convert("RGB")),x["goal"])
        rows.append(dict(id=x["id"],base=select(raw,"base"),augment=select(raw,"augment")))
    save(PRIVATE/"proposals.json",rows)


def box_iou(box,gt):
    ix=max(0,min(box[2],gt[2])-max(box[0],gt[0]))
    iy=max(0,min(box[3],gt[3])-max(box[1],gt[1]))
    a=(box[2]-box[0])*(box[3]-box[1])
    b=(gt[2]-gt[0])*(gt[3]-gt[1])
    return ix*iy/(a+b-ix*iy)


def evaluate_box(b,mask,envelopes):
    box=b["xyxy"]
    x,y=np.clip(np.floor((np.array(box[:2])+box[2:])/2).astype(int),0,191)
    sid=int(mask[y,x])
    best=max((box_iou(box,r) for r in envelopes.values()),default=0.)
    return dict(center=bool(sid),iou25=best>=.25,both=bool(sid) and box_iou(box,envelopes[sid])>=.25,
                confidence=float(b["score"]))


def count_views(rows):
    counts=Counter()
    for r in rows:
        positive=r["positive"]
        preds=r["pred"]
        counts["images"]+=1
        counts["positive_frames"]+=int(positive)
        counts["empty_frames"]+=int(not positive)
        counts["empty_frames_with_any_box"]+=int(not positive and bool(preds))
        counts["positive_frames_without_box"]+=int(positive and not preds)
        counts["frames_with_box"]+=bool(preds)
        counts["boxes"]+=len(preds)
        for metric in ("center","iou25","both"):
            counts["top1_"+metric]+=bool(preds and preds[0][metric])
            counts["any_"+metric]+=any(b[metric] for b in preds)
            counts["top1_wrong_"+metric]+=bool(preds and not preds[0][metric])
            counts["false_boxes_"+metric]+=sum(not b[metric] for b in preds)
    return dict(counts)


def score():
    inputs=read(PRIVATE/"inputs.json")
    rows=read(PRIVATE/"proposals.json")
    assert len(inputs)==len(rows)==12 and [r["id"] for r in rows]==[x["id"] for x in inputs]
    captures={}
    for mode in ("base","augment"):
        observations=[]
        for x,r in zip(inputs,rows):
            mask=np.load(PRIVATE/(x["id"]+".npy"),allow_pickle=False)
            envelopes={}
            for sid in np.unique(mask):
                if sid==0: continue
                yy,xx=np.where(mask==sid)
                envelopes[int(sid)]=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
            preds=[evaluate_box(b,mask,envelopes) for b in r[mode]]
            assert all(a["confidence"]>=b["confidence"] for a,b in zip(preds,preds[1:]))
            observations.append(dict(group=x["group"],view=x["view"],goal=x["goal"],
                                     positive=bool(envelopes),pred=preds))
        groups=[]
        for group in sorted({x["group"] for x in observations}):
            obs=[x for x in observations if x["group"]==group]
            center=next(v for v in obs if v["view"]=="center")
            ranked=sorted([(b["confidence"],v["view"],b) for v in obs for b in v["pred"]],
                          key=lambda a:-a[0])
            groups.append(dict(goal=center["goal"],center_has_target_pixels=center["positive"],
                               views_with_target=[x["view"] for x in obs if x["positive"]],
                               views_with_any_correct_box=[x["view"] for x in obs if any(b["both"] for b in x["pred"])],
                               highest_confidence_view=ranked[0][1] if ranked else None,
                               highest_confidence_box_correct=bool(ranked and ranked[0][2]["both"])))
        captures[mode]=dict(by_image=count_views(observations),by_group=groups,
                            by_view={v:count_views([r for r in observations if r["view"]==v]) for v in CAMERAS})
    weights=ROOT/"b_verification/detector_run/weights/groundingdino_swint_ogc.pth"
    config=ROOT/"jepa_reid_v21/detector/protocol.json"
    report=dict(frames=12,grants=4,paid_requests=0,model="GroundingDINO Swin-T OGC",weights_sha256=digest(weights),
                prior_detector_protocol_sha256=digest(config),thresholds=dict(box=.35,text=.25,nms=.85),
                positive="at least one pixel of a task-accepted scorable instance",
                both="box center on target mask AND IoU >= 0.25 to visible-pixel envelope of that same instance",
                main="augment",results=captures)
    save(HERE/"dino_cue_review.json",report)
    print(json.dumps(report["results"]["augment"]),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("render","infer","score"))
    args=p.parse_args()
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    PRIVATE.mkdir(mode=0o700,exist_ok=True)
    {"render":render,"infer":infer,"score":score}[args.action]()
