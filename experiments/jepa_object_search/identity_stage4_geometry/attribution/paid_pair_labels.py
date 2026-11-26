"""Independent scoring of every stored identity request in the two completed clock runs."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from audit_archive import BASE,PRIVATE,P,lines,read,save
from verify_views import semantic_scene,state
from evaluation.main.evaluator import _env_action,_near_target_ids,_reset_environment


def episode(arm):
    pilot=P/"mapping_runtime/pilots"/("identity_stage4_clock_v2_"+arm)
    r,=(pilot/"comparison/accepted/second").glob("*.json")
    ticket=read(r); path=Path(ticket["result"])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==ticket["result_sha256"]
    return read(path),path.parent


def region(sem,xyxy,unit,targets):
    n=sem.shape[0]
    x,y,u,v=np.clip(np.round(np.asarray(xyxy)*n/unit),0,n).astype(int)
    assert u>x and v>y
    mask=sem[y:v,x:u]; size=mask.size
    ids={i:int((mask==i).sum()) for i in targets}
    return dict(area=int(size),center=int(sem[min((y+v)//2,n-1),min((x+u)//2,n-1)]),
                task_pixels={i:k for i,k in ids.items() if k},fraction=max(ids.values())/size)


def truth(arm):
    doc,folder=episode(arm)
    ep,=doc["episodes"]; steps=ep["paper"]["steps"]
    act=lines(folder/"agent/action_trace.jsonl")
    captures=sorted((folder/"private_stage4_pairs").glob("*/complete.json"))
    cap=[read(f) for f in captures]
    hist=ep["paper"]["agent_trace"]
    obs=hist["observations"]
    source_times={int(r["mapping_frame_id"]):r["step"] for r in act if (r.get("mapping") or {}).get("is_keyframe")}
    selected={c["step"] for c in cap}|{source_times[c["source"]] for c in cap}
    env,item,data=semantic_scene(doc)
    targets=set(map(int,item.target_object_ids)); saved={}
    gt_folder=PRIVATE/"evaluator_only"/arm
    gt_folder.mkdir(exist_ok=True,parents=True,mode=0o700)
    try:
        f=_reset_environment(env,item,data)
        for t,s in enumerate(steps):
            assert s["step"]==t
            if t in selected:
                assert np.linalg.norm(state(env)[0]-s["position_before"])<.002
                mask=np.asarray(f["semantic"]).squeeze().copy()
                near=sorted(map(int,_near_target_ids(tuple(state(env)[0]),item.target_object_ids,item.target_positions,
                            threshold=.25,pathfinder=env.sim.pathfinder,target_view_points=item.view_points)))
                rgb=np.asarray(f["rgb"])[...,:3]
                small=cv2.resize(rgb,(64,64),interpolation=cv2.INTER_AREA)
                assert mask.shape==(320,320) and rgb.shape==(1280,1280,3)
                saved[t]=(mask,near,small)
                np.savez_compressed(gt_folder/f"step_{t}.npz",semantic=mask,near=near)
            if t>=max(selected):
                break
            if s["action_role"]=="navigate":
                f=env.step(_env_action(s["action"]))
    finally:
        env.close()
    def thumb(path):
        img=np.asarray(Image.open(path).convert("RGB"))
        return cv2.resize(img,(64,64),interpolation=cv2.INTER_AREA).astype(np.float32)
    rows=[]
    for c,file in zip(cap,captures):
        t=c["step"]; source=c["source"]; st=source_times[source]
        cur,near,native=saved[t]; src,_,reg=saved[st]
        folder=file.parent
        live_error=float(np.abs(thumb(folder/"live.png")-native).mean())
        source_file=folder/f"anchor_{source}.png"
        source_error=float(np.abs(thumb(source_file)-reg).mean()) if source_file.exists() else None
        assert live_error<1. and (source_error is None or source_error<5.), (t,source,source_error,live_error)
        # Resolve original clicks independently of the predicted current index.
        sameframe=[o for o in obs if o["frame_id"]==source and o["step"]<=t and len(o.get("bbox") or [])==4]
        roi=np.asarray(c["source_normalized_roi"])
        winner=sorted(sameframe,key=lambda o:float(np.max(np.abs(roi-np.asarray(o["bbox"])/518))))
        original_box=winner[0] if winner else None
        if original_box is not None:
            nearest=float(np.max(np.abs(roi-np.asarray(original_box["bbox"])/518)))
            tied=[o for o in winner if float(np.max(np.abs(roi-np.asarray(o["bbox"])/518)))<1e-5]
        else:
            nearest=None;tied=[]
        pts=sorted({tuple(o["pixel"]) for o in tied if len(o.get("pixel") or [])==2})
        guesses=[]
        for x,y in pts:
            xi,yi=np.clip(np.round(np.array([x,y])*.32),0,319).astype(int)
            guesses.append(int(src[yi,xi]))
        region_src=region(src,roi,1.,targets)
        options=[region(cur,box,c["work_width"],targets) for box in c["live_boxes_work_width"]]
        rows.append(dict(step=t,source=source,source_step=st,registered_roi=region_src,click_ids=guesses,
            reference_distance_to_observation_bbox=nearest,n_observations_on_keyframe=len(sameframe),
            selected_distinct_clicks=len(pts),live=options,official_near=near,targets=sorted(targets),
            source_image_l1=source_error,live_image_l1=live_error))
    save(gt_folder/"request_labels.json",rows)
    return rows


def first(rows):
    pred=read(P/"identity_stage4_geometry/provider/first_joint_anchored.json")
    r,=[r for r in rows if r["step"]==pred["step"] and r["source"]==pred["source"]]
    index=pred["selected_box"]
    select=r["live"][index]; rt=set(r["targets"]); near=set(r["official_near"])
    clicks=r["click_ids"]
    visible_ids={int(i) for i,n in select["task_pixels"].items() if n>0}
    single=clicks[0] if len(set(clicks))==1 else None
    same_pixels=sum(int(n) for i,n in select["task_pixels"].items() if int(i)==single)
    frozen_mask=np.load(PRIVATE/"evaluator_only/keyframes_only"/f"step_{r['step']}.npz")["semantic"]
    outcome=dict(step=r["step"],selected_zero_based=index,n_current_options=len(r["live"]),
                source_observations_on_frame=r["n_observations_on_keyframe"],source_distinct_clicks=r["selected_distinct_clicks"],
                source_box_to_recorded_box_max_frac=r["reference_distance_to_observation_bbox"],
                source_click_on_goal=bool(clicks) and set(clicks)<=rt,
                source_region_task_fraction=round(r["registered_roi"]["fraction"],4),
                all_candidate_centers_on_task=[b["center"] in rt for b in r["live"]],
                all_candidate_task_fractions=[round(b["fraction"],4) for b in r["live"]],
                selected_center_on_task=select["center"] in rt,
                selected_center_matches_source_click=bool(clicks) and set(clicks)=={select["center"]} and select["center"] in rt,
                selected_center_in_official_range=select["center"] in near,
                selected_contains_registered_goal_pixels=same_pixels>0,
                selected_target_pixel_count=same_pixels,
                selected_target_only_registered_goal=single in rt and visible_ids=={single},
                selected_fraction_of_visible_registered_goal=round(float(same_pixels/max(1,(frozen_mask==single).sum())),4),
                registered_goal_in_official_range=single in near,
                any_official_target_in_range=bool(near))
    save(BASE/"paid_first_pair_truth.json",outcome)
    print(outcome,flush=True)


if __name__=="__main__":
    from sys import argv
    if argv[1:]==["cached_first"]:
        first(read(PRIVATE/"evaluator_only/keyframes_only/request_labels.json"))
        raise SystemExit(0)
    for arm in argv[1:] or ("keyframes_only","anchored_pnp"):
        rows=truth(arm)
        print(arm,"queries",len(rows),"unique_sources",len({r["source"] for r in rows}),
              "source_l1_max",max(r["source_image_l1"] or 0. for r in rows),
              "live_l1_max",max(r["live_image_l1"] for r in rows),flush=True)
        if arm=="keyframes_only": first(rows)
