"""Exercise the production duplicate-review entry on two preselected HSSD test pairs, CPU only."""
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from evaluate import label_boxes
from identity import guarded_instance_memory
from map_views import metric_view, point_in_box, rgb_aligned
from trace_agent import TraceFrontier


def no_internet(event, args):
    if event == "socket.connect" and args[0].family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError("Offline check")


def view(root, metadata, detections):
    bgr = cv2.imread(str(root / metadata["rgb"]))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(root / metadata["depth"], allow_pickle=False)[::8,::8]
    u,v = np.meshgrid(np.arange(depth.shape[1])*8+.5, np.arange(depth.shape[0])*8+.5)
    K, T = np.array(metadata["K"]), np.array(metadata["T_world_from_camera"])
    local = np.stack([(u-K[0,2])/K[0,0]*depth,(v-K[1,2])/K[1,1]*depth,depth],axis=-1).reshape(-1,3)
    snap = dict(rows=(v-.5).ravel().astype(int), points=local@T[:3,:3].T+T[:3,3], pose=T,
                colors=rgb[::8,::8].reshape(-1,3), frame_id=metadata["frame_id"])
    geo = metric_view(snap,1.,8)
    if geo is None or not rgb_aligned(snap, rgb, 8):
        return None
    boxes = [r["xyxy"] for r in detections]
    return dict(meta=metadata, snap=snap, raw=(root/metadata["rgb"]).read_bytes(),
                boxes=boxes, labels=label_boxes(boxes,metadata["boxes"]),
                points=[point_in_box(geo, b) for b in boxes])


def preselect(root, predictions):
    groups = defaultdict(list)
    for line in (root/"frames.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["split"]=="test":
            groups[(r["house"],r["sequence"])].append(r)
    picked = {}
    for key in sorted(groups):
        rows = sorted(groups[key],key=lambda r:(r["view"],r["frame_id"]))
        frames = [view(root, r, predictions[r["frame_id"]]) for r in rows]
        for l,r in combinations(frames,2):
            if l is None or r is None or l["meta"]["category"]!=r["meta"]["category"]:
                continue
            A,B = l["snap"]["pose"],r["snap"]["pose"]
            baseline = np.linalg.norm(A[:3,3]-B[:3,3])
            angle = np.degrees(np.arccos(np.clip(A[:3,2]@B[:3,2],-1,1)))
            if not (0.02<baseline<=1.25 and angle<=45):
                continue
            for i, ll in enumerate(l["labels"]):
                for j, rr in enumerate(r["labels"]):
                    if not ll or not rr or l["points"][i] is None or r["points"][j] is None:
                        continue
                    distance = np.linalg.norm(np.subtract(l["points"][i],r["points"][j]))
                    kind = "same" if ll["instance"]==rr["instance"] else "distinct_same_class"
                    if distance<=.60 and kind not in picked:
                        picked[kind]=l,i,r,j,float(distance),float(baseline),float(angle)
        if len(picked)==2:
            break
    return picked


def run(root):
    sys.addaudithook(no_internet)
    project=root.resolve()
    kit=project/"jepa_reid_v21"
    ck=kit/"hard_cases/pose_jepa_near_review.pt"
    os.environ.update(NAV_REID_POSE_WEIGHTS=str(ck), NAV_REID_DEVICE="cpu", NAV_COVISIBLE_GUARD="1")
    os.environ.pop("NAV_REID_WEIGHTS",None)
    pred={row["frame_id"]:row["boxes"] for row in map(json.loads,(kit/"detector/predictions.jsonl").read_text().splitlines())}
    picked=preselect(kit/"data",pred)
    answers=[]
    a=TraceFrontier.__new__(TraceFrontier)
    try:
        for kind,(left,i,right,j,distance,baseline,angle) in picked.items():
            a._init_report_memory()
            a.now=2
            a._metric_snapshot={"scale":1.}
            a.instance_store=guarded_instance_memory()
            a.instance_dup_radius_m=.6
            a._instance_goal_index={}
            a.dup_review_deliver_max=6
            events=[]
            a._audit=lambda event,**fields:events.append(dict(event=event,**fields))
            a._note=lambda event,*_,**fields:events.append(dict(event=event,**fields))
            snapshots=[dict(v["snap"],frame_id=fid) for fid,v in ((1,left),(2,right))]
            a.client=SimpleNamespace(get_frame_points=lambda stride:snapshots,
                          get_frame_image=lambda fid:({"found":True},(left if fid==1 else right)["raw"]))
            # Raw RGB is the supplied context of this two-frame fixture. The
            # prediction separately crops each real box using the production path.
            a._tool_view_instance=lambda *args,**kwargs:left["raw"]
            old = a.instance_store.new_observation(np.asarray(left["points"][i])[[0,2,1]],
                             left["meta"]["category"], frame_id=1, step=1, bbox=left["boxes"][i])
            node = a.instance_store.create_instance(old)
            new = a.instance_store.new_observation(np.asarray(right["points"][j])[[0,2,1]],
                             right["meta"]["category"], frame_id=2, step=2, bbox=right["boxes"][j])
            ds,n = a.instance_store.nearby(new.point,1.,.6)[0]
            assert n.iid==node.iid
            a._last_dup_reviews=[dict(observation_id=new.oid,frame_id=2,neighbors=[dict(instance_id=node.iid,dist_m=ds,reported=False)])]
            rows,_,_=a._dup_review_payload()
            assert len(rows)==1 and a.instance_store.instance_for_observation(new.oid) is None
            event,=[e for e in events if e["event"]=="pose_review"]
            reported="pose_jepa_review" in rows[0]
            selection=[left["meta"]["frame_id"],i,right["meta"]["frame_id"],j]
            answers.append(dict(label=kind, fixture_sha256=sha256(json.dumps(selection).encode()).hexdigest(),
                         category=left["meta"]["category"], baseline_m=round(baseline,3), optical_angle_deg=round(angle,2),
                         point_separation_m=round(distance,3), metric_source="HSSD RGB-D + known camera pose",
                         gate=event.get("gate"), score=event.get("score"), threshold=event.get("threshold"),
                         hint_in_native_response=reported, remained_pending=True))
    finally:
        if hasattr(a,"_pose_worker"):
            a._pose_worker.pipe.close()
    return dict(split="test",device="cpu",paid_requests=0,
                preselection="first qualified pair of each label in sorted house/sequence/view/box order; model scores excluded",
                checkpoint_sha256=sha256(ck.read_bytes()).hexdigest(),examples=answers)


if __name__=="__main__":
    print(json.dumps(run(Path(sys.argv[1])),indent=2,allow_nan=False))
