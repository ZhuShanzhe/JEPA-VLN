"""Validate closed-run painting proposals by overlay matching and private semantic replay."""
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

P = Path(os.environ["NAV_ROOT"])
sys.path[:0] = [str(P/"jepa_exec_v14"), str(P/"jepa_completion_v17/physical")]
from verify_habitat import deny_network, state
sys.addaudithook(deny_network)
from check_routes import semantic_scene
from evaluation.main.evaluator import _reset_environment, _env_action


def load(p):
    return json.loads(p.read_bytes())


def rows(p):
    return [json.loads(line) for line in p.read_bytes().splitlines() if line]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def top_categories(mask, selector, names):
    counts = Counter(names.get(int(i), "unannotated") for i in mask[selector])
    return dict(counts.most_common(4))


def measure(frame, proposal, private, target_ids, names):
    mask=np.asarray(frame["semantic"]).squeeze()
    assert mask.shape==(320,320)
    raw=Image.fromarray(np.asarray(frame["rgb"])[...,:3])
    assert raw.size==(1280,1280)
    # Square camera. The mapper rescales to 518x518 before grounding.
    # The archived candidate panel starts with the full image at width 320,
    # tinting its native mask by 55% red and drawing a crosshair at the click.
    source=cv2.resize(np.asarray(raw.resize((518,518),Image.Resampling.BICUBIC)),(320,320),
                      interpolation=cv2.INTER_AREA).astype(np.float32)
    path=private/"candidate_evidence_assets"/proposal["panel"]
    assert sha(path)==proposal["panel"]
    image=np.asarray(Image.open(path).convert("RGB"),dtype=np.float32)
    assert image.shape[0]==320 and image.shape[1]>326
    overlay=image[:,:320]
    blend=source*.45 + np.array([255,32,32])*.55
    plain=np.abs(source-overlay).mean(2)
    tinted=np.abs(blend-overlay).mean(2)
    fit=np.minimum(plain,tinted)
    median=float(np.median(fit)); p75=float(np.percentile(fit,75))
    if median > 12 or p75>22:
        raise ValueError(f"Source RGB does not match archived panel: median={median}, p75={p75}")
    box=np.zeros(mask.shape,dtype=bool)
    x,y,u,v=np.clip(np.round(np.asarray(proposal["bbox"])*320/518).astype(int),0,320)
    assert u>x and v>y
    box[y:v,x:u]=True
    cx,cy=np.clip(np.round(np.asarray(proposal["pixel"])*.32).astype(int),0,319)
    assert box[cy,cx]
    # High-confidence interior of the delivered red selection; reject blends
    # the original RGB could explain and suppress JPEG/crosshair artifacts.
    interior=box & (tinted+12<plain) & (tinted<18)
    interior=cv2.erode(interior.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    interior[max(0,cy-5):cy+6,max(0,cx-5):cx+6]=False
    assert interior.sum()>20
    task=np.isin(mask,list(target_ids))
    pic_ids={i for i,name in names.items() if name in ("picture", "painting", "wall art", "artwork", "framed picture")}
    # The exact task instance IDs remain authoritative even if a raw category
    # is more specific than the generic picture-category vocabulary.
    pic_ids |= target_ids
    pic=np.isin(mask,list(pic_ids))
    nearby=cv2.dilate(box.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    return dict(source_match_median_rgb_mae=round(median,2),source_match_p75_rgb_mae=round(p75,2),
                bbox_area=int(box.sum()),identified_mask_interior_pixels=int(interior.sum()),
                entire_frame_task_pixels=int(task.sum()),entire_frame_picture_pixels=int(pic.sum()),
                task_pixels_in_box=int(np.count_nonzero(task & box)),
                task_pixels_in_expanded_box=int(np.count_nonzero(task & nearby)),
                task_pixels_in_selection=int(np.count_nonzero(task & interior)),
                any_picture_pixels_in_box=int(np.count_nonzero(pic & box)),
                any_picture_pixels_in_selection=int(np.count_nonzero(pic & interior)),
                click_on_task=bool(task[cy,cx]),click_category=names.get(int(mask[cy,cx]),"unannotated"),
                mask_interior_categories=top_categories(mask,interior,names))


def main():
    pilot=P/"mapping_runtime/pilots/jepa_reid_v24_first2"
    assert (pilot/"closed.json").exists()
    eid=load(pilot/"panel.json")["rows"][1]["episode_id"]
    receipt=load(pilot/"comparison/accepted/second"/(eid+".json"))
    path=Path(receipt["result"])
    assert sha(path)==receipt["result_sha256"]
    doc=load(path); ep,=doc["episodes"]
    assert "landscape" in ep["goal_text"].lower() and ep["steps_taken"]==300
    private=path.parent/"agent"
    erows=rows(private/"candidate_evidence.jsonl")
    proposals=[r for r in erows if r["event"]=="proposal"]
    assert len(proposals)==len(set(r["candidate_id"] for r in proposals))==5
    trace=rows(private/"action_trace.jsonl")
    steps=ep["paper"]["steps"]
    frames=defaultdict(list)
    for proposal in proposals:
        fid=proposal["frame_id"]
        k,=[r for r in trace if r.get("mapping_frame_id")==fid and (r.get("mapping") or {}).get("is_keyframe")]
        frames[k["step"]].append(proposal)
    reviews={r["candidate_id"]:r.get("status") for r in erows if r["event"]=="review_receipt" and r.get("status")}
    env,item,data=semantic_scene(doc)
    try:
        scene=env.sim.semantic_annotations()
        names={int(o.id.split("_")[-1]):o.category.name().lower().strip() for o in scene.objects
               if o is not None and o.category is not None}
        ids=set(map(int,item.target_object_ids)); assert ids and ids<=names.keys()
        target_names=sorted(set(names[i] for i in ids))
        frame=_reset_environment(env,item,data)
        records=[]
        for i,row in enumerate(steps):
            pos,rot=state(env)
            assert row["step"]==i and np.linalg.norm(pos-row["position_before"])<.002
            for p in frames.get(i,[]):
                records.append(dict(review=reviews.get(p["candidate_id"],"unreviewed"),
                                    **measure(frame,p,private,ids,names)))
            if i==max(frames):
                break
            if row["action_role"]=="navigate":
                frame=env.step(_env_action(row["action"]))
    finally:
        env.close()
    assert len(records)==5
    result=dict(complete=True,paid_requests=0,distinct_source_frames=len(frames),verified_candidates=len(records),
                task_category_names=target_names, scope="Only proposals generated in the closed painting episode",
                by_review={})
    for review in sorted({r["review"] for r in records}):
        group=[r for r in records if r["review"]==review]
        result["by_review"][review]=dict(total=len(group),
            task_under_click=sum(r["click_on_task"] for r in group),
            any_task_in_box=sum(r["task_pixels_in_box"]>0 for r in group),
            any_task_in_selection=sum(r["task_pixels_in_selection"]>0 for r in group),
            any_picture_in_selection=sum(r["any_picture_pixels_in_selection"]>0 for r in group),
            records=group)
    out=P/"jepa_reid_v24/verification/closed_painting_candidate_audit.json"
    out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result))


if __name__=="__main__":
    main()
