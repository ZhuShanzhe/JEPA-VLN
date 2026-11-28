"""Post-hoc visibility and first observed phase for the four controlled physical runs."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"committed_inspect_20261128"))
from visibility_review import (DATA, configured_sim, reference_proximity, lines, instance_lut,
                               _load_goal_sets, load_scene_instances, hs, quat_from_coeffs)


def review(traces):
    panel=json.loads((ROOT/"local_target_20261128/eval_family/panel.json").read_text())
    summary=json.loads((HERE/"loop_results.json").read_text())
    results=[]
    for row in summary["results"]:
        i,arm=row["house_index"],row["arm"]
        spec=panel[i]
        with gzip.open(DATA/"eval/hssdov/content"/(spec["scene"]+".json.gz")) as f:
            ep=next(e for e in json.load(f)["episodes"] if str(e["episode_id"])==str(spec["id"]))
        assert row["goal"]==ep["object_category"]
        classes=_load_goal_sets(DATA)["scoring_class_ids"]["hssdov"][ep["object_category"]]
        records=[r for r in load_scene_instances(DATA/"protocol/instance_viewpoints/hssd","val",ep["scene_id"])
                 if r["class_id"] in classes and r["view_points"]]
        states=lines(traces/f"{i}_{arm}.poses")
        decisions=lines(traces/f"{i}_{arm}.events")
        assert len(states)==301
        nearby=reference_proximity(ep,states,records)
        assert len(set.union(*nearby))==row["distinct_approached"]
        phases=["scan_or_tail"]*301
        phases[0]="start"
        for e in decisions:
            if "action_source" in e and "last_step" in e:
                for step in range(e["step"]+1,e["last_step"]+1):
                    phases[step]=e["action_source"]
        assert phases.count("local")==row["executed_and_verified"]
        sim=configured_sim(ep)
        try:
            lut=instance_lut(sim,"hssd",DATA/"vocab")
            manager=sim.get_rigid_object_manager()
            objects={o.semantic_id:o for o in (manager.get_object_by_handle(h) for h in manager.get_object_handles())}
            for rec in records:
                sid=rec["semantic_id"]
                assert lut[sid]==rec["class_id"]+1
                assert Path(objects[sid].creation_attributes.handle).name.removesuffix(".object_config.json")==rec["template_name"]
            first_visible,first_close={},{}
            for state,phase,local in zip(states,phases,nearby):
                cam=hs.AgentState()
                cam.position=np.asarray(state["xyz"],np.float32)
                cam.rotation=quat_from_coeffs(np.asarray(state["quat_xyzw"]))
                sim.get_agent(0).set_state(cam,reset_sensors=True)
                mask=sim.get_sensor_observations()["instance_mask"]
                threshold=math.ceil(mask.size*.01)
                ids,counts=np.unique(mask,return_counts=True)
                visible=set(ids[counts>=threshold]) & {r["semantic_id"] for r in records}
                for sid in visible:
                    first_visible.setdefault(sid,phase)
                for sid in visible & local:
                    first_close.setdefault(sid,phase)
            results.append(dict(arm=arm,house_index=i,goal=row["goal"],frames=len(states),
                                visible=len(first_visible),close_and_visible=len(first_close),
                                first_visible_phase=dict(Counter(first_visible.values())),
                                first_close_and_visible_phase=dict(Counter(first_close.values()))))
        finally:
            sim.close()
    return dict(threshold_frame_fraction=.01,reference_view_geodesic_m=.25,results=results)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("private_attempt",type=Path)
    args=parser.parse_args()
    result=review(args.private_attempt)
    (HERE/"loop_visibility.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
