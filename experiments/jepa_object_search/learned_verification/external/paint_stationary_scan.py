"""Run a fixed yaw/pitch scan at the first archived painting pose in official report range."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

P=Path(os.environ["NAV_ROOT"])
for sub in ("jepa_exec_v14","jepa_completion_v17/physical","jepa_reid_v21/detector"):
    sys.path.insert(0,str(P/sub))
from verify_habitat import deny_network,replay_until,state
sys.addaudithook(deny_network)
from check_routes import semantic_scene
from evaluation.main.evaluator import _env_action,_near_target_ids
from evaluation.main.benchmark_api import Action
from proposal_worker import ProposalWorker


def near(pos,item,env):
    return _near_target_ids(tuple(pos),item.target_object_ids,item.target_positions,
                            threshold=.25,pathfinder=env.sim.pathfinder,target_view_points=item.view_points)


def measure(frame,boxes,target_ids):
    mask=np.asarray(frame["semantic"]).squeeze()
    height,width=frame["rgb"].shape[:2]
    pixels=[]; overlaps=[]
    for target in target_ids:
        ys,xs=np.where(mask==int(target)); pixels.append(len(ys))
        if not len(ys):
            continue
        gt=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1],float)*np.array(
                 [width/mask.shape[1],height/mask.shape[0]]*2)
        for d in boxes:
            box=np.array(d["bbox"],dtype=float)
            inter=np.maximum(np.minimum(gt[2:],box[2:])-np.maximum(gt[:2],box[:2]),0).prod()
            union=np.prod(gt[2:]-gt[:2])+np.prod(box[2:]-box[:2])-inter
            overlaps.append(float(inter/union) if union else 0.)
    return dict(max_task_pixels=max(pixels,default=0),max_iou=max(overlaps,default=0.))


def main():
    def load(p):return json.loads(p.read_bytes())
    pilot=P/"mapping_runtime/pilots/jepa_reid_v24_first2"
    row=load(pilot/"panel.json")["rows"][1]
    ticket=load(pilot/"comparison/accepted/second"/(row["episode_id"]+".json"))
    path=Path(ticket["result"]);assert hashlib.sha256(path.read_bytes()).hexdigest()==ticket["result_sha256"]
    doc=load(path); ep,=doc["episodes"]
    assert ep["action_protocol"]=="goat" and "landscape" in ep["goal_text"].lower()
    env,item,data=semantic_scene(doc)
    right=int(Action.TURN_RIGHT); up=int(Action.LOOK_UP); down=int(Action.LOOK_DOWN)
    try:
        names={str(k).lower() for k in env.task.actions}
        assert _env_action(right).lower() in names
        pitch=all(_env_action(a).lower() in names for a in (up,down))
        moves=[right]*2+[up]+[right]*6+[down]+[right]*2 if pitch else [right]*12
        steps=ep["paper"]["steps"]
        first=next(i for i,step in enumerate(steps) if near(step["position_before"],item,env))
        frame=replay_until(env,item,data,steps,first)
        pos0=state(env)[0]
        assert near(pos0,item,env)
        # Save new inputs before inspecting any target ID, pixel evidence, or detections.
        frames=[{k:np.asarray(frame[k]).copy() for k in ("rgb","semantic")}]
        max_shift=0.
        for a in moves:
            frame=env.step(_env_action(a))
            max_shift=max(max_shift,float(np.linalg.norm(pos0-state(env)[0])))
            frames.append({k:np.asarray(frame[k]).copy() for k in ("rgb","semantic")})
        assert max_shift<.002
        distinct_rgb=len({hashlib.sha256(np.ascontiguousarray(f["rgb"]).tobytes()).digest() for f in frames})
        assert distinct_rgb>=10
        worker=ProposalWorker(P/"b_verification/env/bin/python",P,"cuda")
        try:
            readings=[measure(f,worker.detect(np.ascontiguousarray(f["rgb"][...,:3]),"painting"),item.target_object_ids) for f in frames]
        finally:
            worker.close()
    finally:
        env.close()
    result=dict(paid_requests=0,source="First pose in range from a previously accepted HM3D painting trajectory",
                start_selected_retrospectively=True,physics_actions=len(moves),look_up_down_enabled=pitch,
                action_counts=dict(Counter(Action(a).name for a in moves)),full_scored_views=len(readings),
                distinct_rendered_images=distinct_rgb,
                max_base_displacement_m=round(max_shift,5),start_task_pixels=readings[0]["max_task_pixels"],
                best_task_pixels_in_scan=max(r["max_task_pixels"] for r in readings[1:]),
                views_after_movement_with_at_least_103_task_pixels=sum(r["max_task_pixels"]>=103 for r in readings[1:]),
                start_best_detector_iou=round(readings[0]["max_iou"],4),peak_best_detector_iou=round(max(r["max_iou"] for r in readings[1:]),4),
                newly_scanned_views_with_detector_iou_at_least_half=sum(r["max_iou"]>=.5 for r in readings[1:]))
    (Path(__file__).parent/"paint_scan_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
