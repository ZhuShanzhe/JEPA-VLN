"""Check the actual two-process visual/metric interface on untouched HSSD RGB-D."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import cv2
import numpy as np

from multiview_agent import MultiviewFrontier
from pose_worker import PoseWorker


def check(project):
    here=project/"jepa_reid_v21"
    data=here/"data"
    frames=[json.loads(l) for l in (data/"frames.jsonl").read_text().splitlines()]
    preds={r["frame_id"]:r["boxes"] for r in map(json.loads,(here/"detector/predictions.jsonl").read_text().splitlines())}
    # Only the existence of a detection determines the test views; do not select by ID or score.
    selected={}
    for r in frames:
        if r["split"]=="test" and preds[r["frame_id"]]:
            selected.setdefault(r["sequence"],[]).append(r)
    group=next(v[:2] for v in selected.values() if len(v)>=2)
    snapshots=[]
    img={}
    agent=MultiviewFrontier.__new__(MultiviewFrontier)
    agent.now=1
    agent._map_step=None
    for r in group:
        fid=r["frame_id"]
        rgb=cv2.imread(str(data/r["rgb"]))[:,:,::-1].copy()
        img[fid]=(data/r["rgb"]).read_bytes()
        depth=np.load(data/r["depth"],allow_pickle=False)[::8,::8]
        u,v=np.meshgrid(np.arange(depth.shape[1])*8+.5,np.arange(depth.shape[0])*8+.5)
        K=np.array(r["K"])
        pose=np.asarray(r["T_world_from_camera"])
        pts=np.stack([(u-K[0,2])/K[0,0]*depth,(v-K[1,2])/K[1,1]*depth,depth],axis=-1).reshape(-1,3)
        pts=pts@pose[:3,:3].T+pose[:3,3]
        snapshots.append(dict(frame_id=fid,rows=(v-.5).ravel().astype(int),points=pts,pose=pose,
                              colors=rgb[::8,::8].reshape(-1,3)))
    agent.client=SimpleNamespace(get_frame_points=lambda stride:snapshots,
                                 get_frame_image=lambda fid:({"found":True},img[fid]))
    packets=[]
    for r in group:
        packet=None
        for b in preds[r["frame_id"]]:
            obs=SimpleNamespace(frame_id=r["frame_id"],bbox=b["xyxy"])
            packet=agent._metric_observation(obs,1.)
            if packet is not None:
                break
        if packet is None:
            raise ValueError("Predetermined smoke view lacks aligned support")
        packets.append(packet)
    ck=here/"model/pose_jepa.pt"
    worker=PoseWorker(project.parent/"jepa_instance_bench_20261016/policy_env/bin/python",project,ck,"cpu")
    try:
        out=worker.compare(*packets)
        assert 0<=out["score"]<=1 and out["match"]==(out["score"]>=out["threshold"])
    finally:
        worker.pipe.close()
    return dict(passed=True,paid_requests=0,source="HSSD RGB-D converted to mapper snapshot format; real RGB detections, actual frozen model",weights_sha256=hashlib.sha256(ck.read_bytes()).hexdigest())


if __name__=="__main__":
    print(json.dumps(check(Path(sys.argv[1]))))
