"""Post-episode physical receipt: planned observation views versus reached views."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np


def expected(actions):
    x=y=yaw=0.
    for action in actions:
        if action==1:
            x+=.25*math.cos(yaw)
            y+=.25*math.sin(yaw)
        else:
            yaw+=math.pi/6 if action==2 else -math.pi/6
    return np.array([x,y]),yaw


def run(project,result):
    sys.path.insert(0,str(project/"jepa_exec_v14"))
    from verify_habitat import scene
    from verify_rank import deny_ip_sockets
    from evaluation.main.evaluator import _reset_environment,_env_action
    import quaternion
    deny_ip_sockets()
    doc=json.loads(result.read_bytes())
    ep,=doc["episodes"]
    timeline=ep["paper"]["steps"]
    log=result.parent/"agent/v13_routes.jsonl"
    events=[json.loads(line) for line in log.read_bytes().splitlines() if line] if log.is_file() else []
    starts=[r for r in events if r["event"]=="view_track_start"]
    closed={r["start"]:r for r in events if r["event"]=="view_track_end"}
    counters=Counter(selected_views=len(starts))
    if not starts:
        return dict(counters)
    env,item,data=scene(doc)
    states=[]
    try:
        _reset_environment(env,item,data)
        for row in timeline:
            pose=env.sim.get_agent_state()
            assert np.linalg.norm(np.asarray(pose.position)-row["position_before"])<.002
            states.append((np.array(pose.position),quaternion.as_rotation_matrix(pose.rotation)))
            if row["action_role"]=="navigate":
                env.step(_env_action(row["action"]))
        end=env.sim.get_agent_state()
        states.append((np.array(end.position),quaternion.as_rotation_matrix(end.rotation)))
    finally:
        env.close()
    for row in starts:
        start=row["step"]
        prefix=row["prefix"]
        end=start+len(prefix)
        closure=closed.get(start)
        if closure and closure["reason"]!="observed_command_prefix_complete":
            counters["interrupted_"+closure["reason"]]+=1
            continue
        if end>len(timeline):
            counters["episode_ended_before_view"]+=1
            continue
        if [t["action"] for t in timeline[start:end]]!=prefix:
            counters["command_sequence_diverged"]+=1
            continue
        counters["matched_action_prefix"]+=1
        p,R=states[start]
        q,S=states[end]
        offset=R.T@(q-p)
        forward=R.T@S@np.array([0.,0.,-1.])
        actual=np.array([-offset[2],-offset[0]])
        actual_yaw=math.atan2(-forward[0],-forward[2])
        target,yaw=expected(prefix)
        distance=float(np.linalg.norm(target-actual))
        angle=abs(math.degrees(math.remainder(actual_yaw-yaw,2*math.pi)))
        if distance<=.125 and angle<=15:
            counters["physically_confirmed_view"]+=1
        else:
            counters["pose_missed_despite_matching_actions"]+=1
    return dict(counters)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("project",type=Path)
    p.add_argument("result",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    counts=run(a.project,a.result)
    a.output.write_text(json.dumps(counts,indent=2)+"\n")
    print(counts)
