"""Score semantic localization and the actual views after navigation. All labels remain evaluator-only."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT/"committed_inspect_20261128"
sys.path.insert(0,str(OLD))
from visibility_review import (DATA, PIXELS, THRESHOLDS, configured_sim, hs, instance_lut,
    lines, load_scene_instances, _load_goal_sets, quat_from_coeffs, reference_proximity)

VIEWS=("right","center","left")


def box_match(mask,box):
    a=np.clip(np.asarray(box,dtype=np.float64)*192,0.,192.)
    assert len(a)==4 and np.all(a[2:]>a[:2])
    x,y=np.clip(np.floor((a[:2]+a[2:])*.5).astype(int),0,191)
    sid=int(mask[y,x])
    px=set(int(v) for v in np.unique(mask[math.floor(a[1]):math.ceil(a[3]),math.floor(a[0]):math.ceil(a[2])]) if v)
    if not sid:
        return None,0.,bool(px)
    yy,xx=np.where(mask==sid)
    b=np.asarray([xx.min(),yy.min(),xx.max()+1,yy.max()+1])
    overlap=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-overlap
    return sid,float(overlap/union),bool(px)


def run_house(i,spec):
    dest=HERE/"private_events"
    events=lines(dest/f"{i}.jsonl")
    trajectory=lines(dest/f"{i}.poses.jsonl")
    assert len(trajectory)==301 and all(t["step"]==j for j,t in enumerate(trajectory))
    with gzip.open(DATA/"eval/hssdov/content"/(spec["scene"]+".json.gz")) as f:
        ep=next(r for r in json.load(f)["episodes"] if str(r["episode_id"])==str(spec["id"]))
    classes=_load_goal_sets(DATA)["scoring_class_ids"]["hssdov"][ep["object_category"]]
    rows=[r for r in load_scene_instances(DATA/"protocol/instance_viewpoints/hssd","val",ep["scene_id"])
          if r["class_id"] in classes and r["view_points"]]
    close=reference_proximity(ep,trajectory,rows)
    sim=configured_sim(ep)
    checks=Counter()
    try:
        lut=instance_lut(sim,"hssd",DATA/"vocab")
        manager=sim.get_rigid_object_manager()
        objects={obj.semantic_id:obj for obj in (manager.get_object_by_handle(h) for h in manager.get_object_handles())}
        target_ids=set()
        for row in rows:
            sid=row["semantic_id"]
            assert lut[sid]==row["class_id"]+1
            assert Path(objects[sid].creation_attributes.handle).name.removesuffix(".object_config.json")==row["template_name"]
            target_ids.add(sid)
        mask_frames=set()
        inspections=[e for e in events if e.get("cue_selected")]
        for e in inspections:
            tv=int(e["cue_view_index"])
            assert tv in (0,1,2) and e.get("cue_checked") and e.get("gpt_called")
            assert [t["action"] for t in trajectory[e["step"]-3:e["step"]+1]]==["turn_right","turn_left","turn_left","turn_right"]
            mask_frames.add(e["step"]-(3,4,1)[tv])
        kept,counts,first_clear={},[],{}
        for step,t in enumerate(trajectory):
            state=hs.AgentState()
            state.position=np.float32(t["xyz"])
            state.rotation=quat_from_coeffs(np.asarray(t["quat_xyzw"]))
            sim.get_agent(0).set_state(state)
            raw=sim.get_sensor_observations()["instance_mask"]
            n=np.bincount(raw.ravel().clip(min=0))
            frame={sid:int(n[sid]) for sid in target_ids if sid<len(n) and n[sid]}
            if step in mask_frames:
                kept[step]=np.where(np.isin(raw,list(target_ids)),raw,0)
            for sid,pixels in frame.items():
                if pixels>=math.ceil(.01*PIXELS):
                    first_clear.setdefault(sid,step)
            counts.append(frame)
        hits={str(thresh):len({sid for frame in counts for sid,p in frame.items()
                              if p>=math.ceil(thresh*PIXELS)}) for thresh in THRESHOLDS}
        simultaneous={str(thresh):len({sid for frame,near in zip(counts,close) for sid,p in frame.items()
                                      if p>=math.ceil(thresh*PIXELS) and sid in near}) for thresh in THRESHOLDS}
        term={}
        for e in events:
            rel=e if e.get("event")=="arrived" else e.get("released")
            if rel:term[rel["goal_id"]]=(rel["event"],e["step"])
        acquisitions=[]
        for e in inspections:
            view=int(e["cue_view_index"])
            start=e["step"]
            snap=start-(3,4,1)[view]
            sid,iou,any_pixels=box_match(kept[snap],e["cue_bbox_norm"])
            good=bool(sid and iou>=.25)
            acquired=e.get("event")=="acquired" and e.get("action_source")=="local"
            checks["selected_boxes"]+=1
            checks["selected_side_boxes"]+=int(view!=1)
            checks["boxes_overlapping_target_pixels"]+=int(any_pixels)
            checks["valid_selected_boxes"]+=int(good)
            checks["valid_selected_side_boxes"]+=int(good and view!=1)
            checks["correct_boxes_with_projected_anchor"]+=int(good and e["anchors"]>0)
            checks["correct_boxes_with_candidate_site"]+=int(good and e["sites"]>0)
            checks["correct_boxes_with_acquisition"]+=int(good and acquired)
            detail=dict(view=VIEWS[view],valid_box=good,center_on_target=bool(sid),
                        has_target_pixels=any_pixels,iou_same_instance=round(iou,4),anchored=e["anchors"]>0,
                        site_offered=e["sites"]>0,acquired=acquired,status=e["exec_status"])
            if acquired:
                release,at=term.get(e["goal_id"],("episode_end",300))
                detail.update(release=release, new_one_percent_instances_during_period=sum(start<j<=at for j in first_clear.values()))
                if sid:
                    detail.update(same_instance_frame_percent_at_cue=round(100*counts[snap].get(sid,0)/PIXELS,3),
                                  same_instance_frame_percent_at_release=round(100*counts[at].get(sid,0)/PIXELS,3))
                checks["valid_acquired_and_arrived"]+=int(good and release=="arrived")
            acquisitions.append(detail)
        result_path,=list((HERE/"runs").glob(f"{i}_*.json"))
        result=json.loads(result_path.read_text())
        assert result["distinct_approached"]==len(set().union(*close))
        return dict(goal=spec["goal"],counts=dict(checks),selected_cues=acquisitions,
                    visible=hits,visible_and_close=simultaneous,approached=result["distinct_approached"])
    finally:
        sim.close()


if __name__=="__main__":
    panel=json.loads((ROOT/"local_target_20261128/eval_family/panel.json").read_text())
    houses=[run_house(i,spec) for i,spec in enumerate(panel)]
    report=dict(frames=len(houses)*301,houses=houses)
    for field in ("counts","visible","visible_and_close"):
        sums=Counter()
        for h in houses:sums.update(h[field])
        report[field]=dict(sums)
    (HERE/"interface_review.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report["counts"],report["visible"],report["visible_and_close"],flush=True)
