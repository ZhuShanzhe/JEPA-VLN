"""Paired interface test on every previously granted inspection, with the old upper intent held fixed."""
import json
import math
import os
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np

from run_multiview import HERE, ROOT, PRIOR, FAMILY, ScanRecorder, environment, physical, read
from cue_service import CueSelector
from multi_policy import located_cue
from multi_worker import start_worker
from local_runtime import propose_sites, request


def check_case(i,spec,worker):
    trajectory=[json.loads(s) for s in (PRIOR/f"private_events/{i}.poses.jsonl").read_text().splitlines()]
    log=[json.loads(s) for s in (PRIOR/f"private_events/{i}.jsonl").read_text().splitlines()]
    selected={e["step"]:e for e in log if e["event"]=="acquired"}
    if not selected:
        return []
    ep,sim=environment(spec)
    scan=ScanRecorder(sim)
    cue=CueSelector(HERE/"private_fixed"/f"cue_{i}",allow_new=False)
    cases=[]
    try:
        state=physical.hs.AgentState()
        state.position=np.float32(ep["start_position"])
        state.rotation=physical.quat_from_coeffs(ep["start_rotation"])
        sim.get_agent(0).set_state(state)
        obs=scan.get_sensor_observations()
        pose=physical.policy_pose(sim)
        grid=physical.DepthGrid(pose[:2])
        grid.update(pose,obs["depth"])
        for step,t in enumerate(trajectory[1:max(selected)+1],1):
            previous=pose
            obs=scan.step(t["action"])
            pose=physical.policy_pose(sim)
            assert np.allclose(sim.get_agent(0).state.position,t["xyz"],rtol=0,atol=1.e-5)
            assert abs(math.remainder(pose[2]-t["yaw"],math.tau))<1.e-5
            grid.update(pose,obs["depth"])
            if t["action"]=="move_forward" and (obs.get("collided",False) or math.dist(previous[:2],pose[:2])<.08):
                grid.record_contact(previous)
            if step not in selected:
                continue
            frames=list(scan.frames)
            views=[frames[j]["rgb"] for j in (1,0,3)]
            geometry,_=scan.packet(views,frames[0]["depth"],pose)
            choice=cue.select(views,ep["object_category"])
            assert cue.last_receipt["event"]=="prior_cache"
            result=dict(goal=ep["object_category"],accepted_cue=bool(choice),candidates=cue.last_receipt["candidates"])
            if choice is not None:
                anchors,source,yaw=located_cue(choice,views,grid,pose,1.,geometry,90.)
                idx=choice["view_index"]
                assert source=="selected_view_aligned_rgbd" and abs(yaw-geometry[idx][0][2])<1.e-8
                sites=propose_sites(grid,pose,1.,anchors,min(300-step,24))
                observed=request(worker,mode="observe",views=views,plans=[],goal=ep["object_category"])
                assert len(observed["current"])==3
                scores=request(worker,mode="selected_box",view_index=idx,bbox_norm=choice["bbox_norm"],
                               plans=[s["actions"] for s in sites]) if sites else []
                assert len(sites)==len(scores) and all(math.isfinite(x) for x in scores)
                result.update(source_view=("right","center","left")[idx],candidate_id=choice["candidate_id"],
                              projected=bool(anchors),reachable_sites=len(sites),finite_jepa_scores=len(scores))
            cases.append(result)
        return cases
    finally:
        cue.close()
        sim.close()


if __name__=="__main__":
    os.umask(0o077)
    folder=HERE/"private_fixed"
    folder.mkdir(mode=0o700,exist_ok=True)
    ch,proc=start_worker(ROOT,folder/"worker.log")
    try:
        rows=[r for i,spec in enumerate(read(FAMILY/"panel.json")) for r in check_case(i,spec,ch)]
    finally:
        ch.close()
        proc.wait(timeout=60)
    assert len(rows)==4
    result=dict(fixed_grants=len(rows),source="all granted sites in previous 8-house committed run",
                source_pose_match=True, ordinary_gate="historical inspect held fixed", new_paid_calls=0,rows=rows)
    (HERE/"fixed_handoffs.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
