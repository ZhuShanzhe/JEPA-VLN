"""Offline privileged diagnostic. The resulting routes are never given to the evaluated policy."""
import heapq
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import quaternion
from evaluation.main.evaluator import _env_action
from verify_habitat import scene, replay_until
from replay_views import measure
from point_executor import route_geometry


def solve(env, actions, budget):
    st=env.sim.get_agent_state()
    origin=st.position.copy()
    rotation=quaternion.as_rotation_matrix(st.rotation)
    desired,yaw=route_geometry((0.,0.,0.),1.,actions)
    x,y=desired[-1]
    goal=origin + rotation@[-y,0,-x]
    final=int(round(yaw/(math.pi/6)))%12
    bearings=[rotation@[-math.sin(h*math.pi/6),0,-math.cos(h*math.pi/6)] for h in range(12)]
    def cost(point,h):
        gap=np.linalg.norm((point-goal)[[0,2]])
        yawdiff=(final-h)%12
        return max(0,int(math.ceil((gap-.125001)/.25)))+min(yawdiff,12-yawdiff)
    serial=itertools.count()
    key=lambda p,h:(*(int(v) for v in np.rint(p/.035)),h)
    heap=[(cost(origin,0),0,next(serial),origin,0,())]
    visited={key(origin,0):0}
    expanded=0
    while heap and expanded<150000:
        _,negd,_,p,h,path=heapq.heappop(heap)
        d=-negd
        if d>visited[key(p,h)]:
            continue
        expanded+=1
        if cost(p,h)==0:
            for action in path:
                env.step(_env_action(action))
            err,angle=measure(env.sim,origin,rotation,desired[-1],yaw)
            return dict(physically_witnessed=err<=.12501 and angle<=15.001,
                        steps=len(path),physical_error_m=err,physical_yaw_deg=angle,expanded=expanded)
        for action in (1,2,3):
            nh=(h+int(action==2)-int(action==3))%12
            nxt=(np.asarray(env.sim.pathfinder.try_step(p,p+.25*bearings[h]),float) if action==1 else p)
            k=key(nxt,nh)
            if visited.get(k,budget+1)<=d+1:
                continue
            f=d+1+cost(nxt,nh)
            if f<=budget:
                visited[k]=d+1
                heapq.heappush(heap,(f,-d-1,next(serial),nxt,nh,path+(action,)))
    return dict(physically_witnessed=False,expanded=expanded,reason="discretized_search_found_no_witness")


def main(project, output):
    source=project/"mapping_runtime/pilots/jepa_confirm_v18_first2"
    history=json.loads((project/"route_execution_v1/paired_geometry.json").read_bytes())
    out=[]
    for panel in json.loads((source/"panel.json").read_bytes())["rows"]:
        ref=json.loads((source/"comparison/accepted/second"/(panel["episode_id"]+".json")).read_bytes())
        result=Path(ref["result"])
        doc=json.loads(result.read_bytes())
        ep,=doc["episodes"]
        recorded=[json.loads(x) for x in (result.parent/"agent/v13_routes.jsonl").read_bytes().splitlines()]
        env,item,data=scene(doc)
        try:
            for r in history["results"]:
                if r["episode_ordinal"]!=panel["ordinal"] or r["arms"]["selected_view"]["view_ok"] or r["audit_geometry"]["horizontal_gap_to_snapped_mesh_m"]>.125:
                    continue
                step=r["source_step"]
                event,=[e for e in recorded if e["event"]=="view_track_start" and e["step"]==step]
                replay_until(env,item,data,ep["paper"]["steps"],step)
                budget=min(300-step,len(event["prefix"])+8)
                row=dict(ordinal=panel["ordinal"],step=step,local_budget=budget,**solve(env,event["prefix"],budget))
                out.append(row)
                output.write_text(json.dumps(out,indent=2)+"\n")
                print(row,flush=True)
        finally:
            env.close()
    assert len(out)==3


if __name__=="__main__":
    main(*map(Path,sys.argv[1:]))
