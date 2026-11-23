"""Spot-check stored HSSD reference views after rebinding render IDs independently."""
from collections import Counter
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import habitat_sim as hs
from habitat_sim.utils.common import quat_from_coeffs
import numpy as np

from derive_viewpoints import EYE_HEIGHT, load_scene


def order(house, item):
    return hashlib.sha256((house+"|"+str(item)).encode()).digest()


def audit(path, raw):
    with gzip.open(path,"rt") as stream:
        data=json.load(stream)
    house=data["house"]
    groups={}
    for g in data["goals"]:
        if g["view_points"]:
            groups.setdefault(g["object_category"],[]).append(g)
    checked=Counter()
    max_area_error=0
    sim=load_scene(raw,house,0)
    try:
        objects=sim.get_rigid_object_manager().get_objects_by_handle_substring("")
        new_ids={}
        for i, handle in enumerate(sorted(objects,reverse=True),500):
            objects[handle].semantic_id=i
            new_ids[handle]=i
        agent=sim.get_agent(0)
        for cat, members in groups.items():
            goal=min(members,key=lambda g:order(house,g["object_name"]))
            views=sorted(goal["view_points"],key=lambda v:order(house,v["agent_state"]["position"]))[:3]
            for view in views:
                pose=view["agent_state"]
                state=hs.AgentState()
                state.position=np.array(pose["position"],dtype=np.float32)
                state.rotation=quat_from_coeffs(np.array(pose["rotation"]))
                assert sim.pathfinder.is_navigable(state.position)
                agent.set_state(state)
                state=agent.get_state()
                for sensor in state.sensor_states.values():
                    assert np.linalg.norm(sensor.position-state.position-[0,EYE_HEIGHT,0])<1e-4
                    sensor.rotation=quat_from_coeffs(np.array(view["sensor_rotation"]))
                agent.set_state(state,infer_sensor_states=False)
                obs=sim.get_sensor_observations()
                visible=obs["seg"]==new_ids[goal["object_name"]]
                area=int(visible.sum())
                near=int(np.sum(visible & (obs["depth"]>0)&(obs["depth"]<=2.0)))
                assert area>=48 and near>=32,(house,cat,area,near)
                max_area_error=max(max_area_error,abs(area-view["visible_pixels"]),abs(near-view["near_pixels"]))
                checked[cat]+=1
    finally:
        sim.close()
    return dict(views=sum(checked.values()),by_class=dict(checked),max_pixel_count_difference=max_area_error)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--catalog-dir",type=Path,required=True)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    parts=json.loads(a.plan.read_bytes())
    splits={"train":parts["train"],"calibration":parts["calibration"],"holdout":[r["house"] for r in parts["holdout_candidates"][:4]]}
    result={}
    for split,houses in splits.items():
        fixed=sorted(houses,key=lambda h:order("v7-view-audit",h))[:2]
        result[split]=[audit(a.catalog_dir/(h+".json.gz"),a.raw) for h in fixed]
    a.output.write_text(json.dumps(result,indent=2)+"\n")
