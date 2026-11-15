"""Habitat to mapping RPC integration probe. The scripted motion has no navigation score."""
import json
import os
from pathlib import Path
import numpy as np
from mapping.client import MappingClient


class PhysicalMapProbe:
    def __init__(self, **kwargs):
        self.client=MappingClient(port=int(os.environ["VGGT_SLAM_PORT"]),timeout=240)
        self.reset()

    def reset(self):
        self.moves=[]
        self.displacements=[]
        self.first=None

    def act(self, observation):
        frame=np.ascontiguousarray(observation.rgb[:,:,:3],dtype=np.uint8)
        assert frame.shape==(1280,1280,3) and np.ptp(frame)>0
        if self.first is None:
            self.client.reset_map()
            self.client.set_episode("licensed_scene_engineering_probe")
            accepted=self.client.set_task("description",goal_text=observation.goal_text)
            assert accepted.get("accepted"), accepted.get("reason")
            self.first=frame.copy()
        self.client.feed_frame(frame)
        if observation.gps is not None:
            self.displacements.append(np.asarray(observation.gps).copy())
        sequence=[1]*4+[2]*3+[1]*4+[3]*3+[1]*4+[2]*4+[1]*4+[3]*3
        if len(self.moves)<len(sequence):
            action=sequence[len(self.moves)]
            self.moves.append(action)
            return action
        self.client.flush_map()
        assert self.client.wait_idle(timeout=200)
        state=self.client.get_state()
        poses, ids=self.client.get_all_poses()
        points, colors=self.client.get_map_points(max_points=100000)
        assert state["num_submaps"]>=1 and state["loop_closure_enabled"]
        assert state["sam_available"] and state["semantic"]["caption_enabled"]
        assert poses is not None and len(ids)==len(poses)>=2 and np.isfinite(poses).all()
        assert len(points)==len(colors)>100 and np.isfinite(points).all()
        assert np.any(frame!=self.first)
        pose_span=float(np.max(np.linalg.norm(poses[:,:3,3]-poses[0,:3,3],axis=1)))
        assert pose_span>0
        summary=dict(status="pass",scope="scripted real-scene mapping I/O; no navigation score",
                     observations=len(self.moves)+1,actions_before_finish=len(self.moves),
                     submaps=state["num_submaps"],estimated_poses=len(ids),sampled_map_points=len(points),
                     loop_detector_enabled=state["loop_closure_enabled"],sam_available=state["sam_available"],
                     caption_worker=state["semantic"]["caption_enabled"],embedding_enabled=state["semantic"]["embedding_enabled"])
        if self.displacements:
            summary["physical_displacement_span_m"]=float(max(np.linalg.norm(x-self.displacements[0]) for x in self.displacements))
        Path(os.environ["MOST_MAP_PROBE_RECEIPT"]).write_text(json.dumps(summary,indent=2)+"\n")
        return 0

    def close(self):
        self.client.close()
