"""Post-hoc 24-action physical intervention on the two logged plans at case 39, step 175."""
from functools import lru_cache
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import numpy as np
from evaluation.main.evaluator import _near_target_ids

D=Path(sys.argv[1])
B=D.parents[1]
def read(f):return json.loads(f.read_bytes())


def fork(env,frame,item,objects,rows,ep,plan):
    @lru_cache(maxsize=None)
    def near(loc):
        return set(map(int,_near_target_ids(loc,item.target_object_ids,item.target_positions,threshold=.25,
                        pathfinder=env.sim.pathfinder,target_view_points=item.view_points)))
    def state():
        st=env.sim.get_agent_state()
        return tuple(map(float,st.position)),st.sensor_states["rgb"].rotation.components
    def paint(frame):
        labels=np.asarray(frame["semantic"]).squeeze()
        assert labels.shape==(320,320)
        return {o:int(np.count_nonzero(labels==o)) for o in objects}
    from evaluation.main.evaluator import _env_action
    t=175
    initial_seen=set()
    credited=set()
    reports={r["step"]:set(map(int,r["new_targets"])) for r in ep["report_events"]}
    for r in rows[:t]:
        loc,_=state()
        assert np.linalg.norm(np.array(loc)-r["position_before"])<.002
        initial_seen|=near(loc)
        credited|=reports.get(r["step"],set())
        if r["action_role"]=="navigate":frame=env.step(_env_action(r["action"]))
    p0,q0=state()
    initial_seen|=near(p0)
    assert np.linalg.norm(np.array(p0)-rows[t]["position_before"])<.002
    signature=hashlib.sha256(np.ascontiguousarray(frame["rgb"]).tobytes()).hexdigest()
    collision=0;stalled=0;walk=0;distance=0.;exposures=set();moderate=set();maxpix=0
    prev_counts=paint(frame)
    last_eligible=near(p0)
    union=set()
    actual=[r["action"] for r in rows[t:t+len(plan)]]
    first=None
    latest_error=0.
    follows=plan==actual
    for k,a in enumerate(plan,1):
        last,qprev=state()
        frame=env.step(_env_action(a))
        p,newq=state();dist=np.linalg.norm(np.array(last)-p)
        if follows:
            latest_error=max(latest_error,float(np.linalg.norm(np.array(p)-rows[t+k]["position_before"])))
            assert latest_error<.002
        if a==1:
            walk+=1;distance+=float(dist);stalled+=int(dist<.001)
            collision+=int(bool(env.sim.previous_step_collided))
        last_eligible=near(p)
        union|=last_eligible
        if first is None and union-initial_seen:first=k
        counts=paint(frame)
        exposures.update(o for o,c in counts.items() if c>=1)
        if dist>1e-5 or abs(np.dot(qprev,newq))<1.-1.e-7:
            moderate.update(o for o,c in counts.items() if c>=103 and prev_counts[o]>=103)
        maxpix=max(maxpix,max(counts.values()));prev_counts=counts
    return dict(signature=(p0,q0.tolist(),signature,sorted(credited),sorted(initial_seen)),
        new_instances=union-initial_seen,moderate_instances=moderate,public=dict(
        action_budget=len(plan),forward_commands=walk,forward_collisions=collision,
        forward_without_displacement=stalled,distance_m=round(distance,5),all_eligible_on_branch=len(union),
        previously_unreached_eligible=len(union-initial_seen),as_yet_uncredited_eligible=len(union-credited),
        eligible_at_endpoint=len(last_eligible),any_visible_target_instances=len(exposures),
        two_distinct_consecutive_views_103px_instances=len(moderate),maximum_single_target_pixels=maxpix,
        first_new_eligible_action_offset=first,matched_logged_24_action_sequence=follows,
        max_position_error_m=latest_error if follows else None))


def main():
    r=next(item for item in read(D/"panel.json")["rows"] if item["ordinal"]==39)
    ref=read(D/"comparison/accepted/second"/(r["episode_id"]+".json"))
    p=Path(ref["result"])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==ref["result_sha256"]
    sel=next(row for line in (p.parent/"agent/jepa_selection.jsonl").read_text().splitlines()
        if (row:=json.loads(line))["step"]==175)
    assert sel["selected"]=="motion:straight" and sel["frozen_selected"]=="motion:right30"
    plans=dict(online=sel["selected_nominal_actions"],fixed=[a+1 for a in sel["supported_plans"][sel["frozen_selected"]]])
    assert plans["online"]==[1]*24 and plans["fixed"]==[3]+[1]*23
    srv=B/"runs"/read(D/"mapper_process.json")["run"]
    path=B/"analysis/visibility/visibility_replay.py"
    import importlib.util
    spec=importlib.util.spec_from_file_location("visibility",path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    src=inspect.getsource(mod.replay)
    cut=src.index("        for r, t in zip(rows, trace):")
    src=src[:cut]+'''        return fork(env,frame,item,objects,rows,ep,out)
    finally:
        env.close()
'''
    # Same CPU rendering backend for both physical interventions.
    src=src.replace('        cfg.habitat.simulator.scene = item.scene_id','        cfg.habitat.simulator.scene = item.scene_id\n        cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = -1')
    ns={**vars(mod),"fork":fork};exec(compile(src,"<posthoc physical two-way branch>", "exec"),ns)
    outcomes={k:ns["replay"](p,srv,v) for k,v in plans.items()}
    a,b=(outcomes[k]["signature"] for k in ("online","fixed"));assert a==b
    assert outcomes["online"]["public"]["matched_logged_24_action_sequence"]
    output=dict(ordinal=39,decision_step=175,role="exploratory post-hoc physical branch",paired_fork=True,
        exact_same_position_camera_quaternion_rgb_prior_eligible_and_credited=True,prior_eligible_count=len(a[4]),
        prior_credited_count=len(a[3]),additional_api_calls=0,actual_closed_loop_score_generated=False,
        actions={"online":"24 forwards","fixed":"one right 30 degree turn and 23 forwards"},
        arms={k:v["public"] for k,v in outcomes.items()})
    dest=D/"paired_fork175.json"
    if dest.exists():assert read(dest)==output
    else:dest.write_text(json.dumps(output,indent=2))
    both={}
    for name in ("new_instances","moderate_instances"):
        left=outcomes["online"][name]
        right=outcomes["fixed"][name]
        both[name]=dict(shared=len(left & right),online_only=len(left-right),fixed_only=len(right-left))
    overlap=D/"paired_fork175_instance_overlap.json"
    if overlap.exists():assert read(overlap)==both
    else:overlap.write_text(json.dumps(both,indent=2))
    print(json.dumps(output))


if __name__=="__main__":main()
