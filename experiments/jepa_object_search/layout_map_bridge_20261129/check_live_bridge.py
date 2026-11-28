"""Read-only acceptance against the native geometry service, with no semantic or policy API."""
import argparse
import atexit
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from layout_memory import DX, DZ, LayoutMap
from most_layout_bridge import MostLayoutBridge, layout_for_history


def run(port):
    from agents.nav_agent import NavAgent
    from mapping.client import MappingClient
    remote = MappingClient(port=port, timeout=120)
    try:
        frames = remote.get_frame_points(stride=8)
        revision = dict(remote.last_frame_snapshot_revision)
    finally:
        remote.close()
    assert len(frames) >= 8
    class GeometrySnapshot:
        last_frame_snapshot_revision = revision
        def get_frame_points(self, stride):
            return frames
        def get_captioned_frame_ids(self):
            return False, []
        def get_state(self):
            return revision
    os.environ["NAV_POSE_FREE"] = "1"
    with patch("agents.mapping_agent.MappingClient", return_value=GeometrySnapshot()), \
         patch("agents.nav_agent.VLMDecisionClient.from_env", return_value=SimpleNamespace(
             enabled=False, set_trace_path=lambda path:None)):
        agent = NavAgent()
    atexit.unregister(agent._finalize)
    obs = SimpleNamespace(episode_id="live-geometry-inspection", step_count=int(revision["num_frames"]))
    agent.episode_id = obs.episode_id
    # This rebuild calls the native agent's map snapshot and scale calibration in offline,
    # pose-free mode from the same live keyframe point/pose bundle that powers exploration.
    agent.decision_map_point_stride=8
    provider=MostLayoutBridge()
    p=provider.snapshot(agent,obs,(0.,0.,0.))
    assert provider.last_status == "observed", provider.last_status
    counts=np.bincount(p.cells.ravel(),minlength=3)
    assert np.all(counts>0),counts
    grid=agent._frontier_grid
    gx,gy,yaw=agent._frontier_pose
    scale=agent._frontier_scale
    # Check an analytic +right / +forward local stencil against official world_to_cell.
    samples=0
    for x,z in zip(DX[::5,::5].ravel(),DZ[::5,::5].ravel()):
        xy=(gx+(np.cos(yaw)*z+np.sin(yaw)*x)/scale,
            gy+(np.sin(yaw)*z-np.cos(yaw)*x)/scale)
        cell=grid.world_to_cell(xy)
        original=0
        if grid.in_bounds(cell):
            col,row=cell
            original=2 if grid.obstacle[row,col] else int(grid.free[row,col] and grid.geometry_observed[row,col])
        assert p.query(x,z)==original
        samples+=1
    labeled=dict(episode=obs.episode_id,floor=0,frame_id=obs.step_count)
    local=dict(x=0.,z=0.,yaw=0.,**labeled)
    merged=layout_for_history([dict(local,layout_patch=p)])
    assert np.array_equal(merged.cells,p.cells)
    blank=layout_for_history([local])
    assert not blank.cells.any()
    old=layout_for_history([dict(local,frame_id=obs.step_count+1,layout_patch=p)])
    other=layout_for_history([dict(local,episode="separate",layout_patch=p)])
    assert not old.cells.any() and not other.cells.any()
    assert not provider.snapshot(agent,obs,(0.,0.,0.),floor=1).cells.any()
    assert not provider.snapshot(agent,obs,None).cells.any()
    assert not provider.snapshot(agent,SimpleNamespace(episode_id=obs.episode_id,step_count=obs.step_count+1),
                                 (0.,0.,0.),refresh=False).cells.any()
    same_revision=SimpleNamespace(episode_id=obs.episode_id,pose_free=True,
             _last_frontier_step=obs.step_count+1,_frontier_revision=agent._frontier_revision,
             _frontier_grid=grid,_frontier_pose=agent._frontier_pose,_frontier_scale=scale)
    assert not provider.snapshot(same_revision,SimpleNamespace(episode_id=obs.episode_id,
             step_count=obs.step_count+1),(0.,0.,0.),refresh=False).cells.any()
    assert provider.last_status=="stale_revision"
    depth=np.full((192,192),2.,dtype=np.float32)
    start=dict(x=0.,z=0.,yaw=0.,depth=depth)
    before=LayoutMap.from_depth_history([start],episode="rgbd")
    after=LayoutMap.from_depth_history([start,dict(start,bump=True)],episode="rgbd")
    assert before.query(0.,.25)==1 and after.query(0.,.25)==2
    return dict(native_map="single atomic snapshot of a live RGB-only VGGT-SLAM mapper; pose-free metric ruler",
                native_keyframe_records=len(frames),distinct_keyframes=len({f["frame_id"] for f in frames}),
                checked_world_to_cell=samples,local_native_free_cells=int(counts[1]),local_native_occupied_cells=int(counts[2]),
                verified=["native NavAgent map refresh", "pose-free native geometry without raw Habitat depth",
                          "exact occupancy parity on sampled cells", "missing pose, empty map and stale revision",
                          "cross-episode isolation","floor change rejects 2-D map", "depth-only fallback",
                          "new contact marks the attempted motion blocked"],
                publication="Counts only; no frame images, map arrays, trajectory or episode IDs.")


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--port",type=int,required=True)
    cli.add_argument("--out",type=Path,required=True)
    opt=cli.parse_args()
    report=run(opt.port)
    opt.out.write_text(json.dumps(report,indent=2)+"\n")
    print(report)
