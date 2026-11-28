"""Replay every selected box with the recorded actions and isolate the first failing site gate."""
from collections import Counter,deque
import json
import math

import numpy as np
from run_multiview import HERE,FAMILY,environment,physical,digest,hash_plane,read
from multi_policy import located_cue
import observation_sites as geometry

OFFSETS=((.95,0.),(.95,32.),(.95,-32.),(1.4,0.),(.72,0.),(1.20,55.),(1.20,-55.))


def classify(grid,pose,anchor,budget):
    online=geometry._Map(grid,1.)
    counts=Counter()
    used=set()
    detour_possible=0
    for radius,deg in OFFSETS:
        angle=math.atan2(pose[1]-anchor[1],pose[0]-anchor[0])+math.radians(deg)
        point=(anchor[0]+radius*math.cos(angle),anchor[1]+radius*math.sin(angle))
        cell=online.cell(point)
        if math.dist(pose[:2],point)<.34:
            counts["closer_than_minimum_translation"]+=1
            continue
        if not online.free(point):
            x,y=cell
            if not online.observed(cell): why="floor_unobserved"
            elif grid.blocked[y,x]: why="previous_forward_contact"
            elif grid.obstacle[y,x]: why="occupied_floor_cell"
            else: why="obstacle_clearance_margin"
            counts[why]+=1
            continue
        yaw=math.atan2(anchor[1]-point[1],anchor[0]-point[0])
        key=(round(point[0]/.22),round(point[1]/.22),round(yaw/geometry.TURN)%12)
        if key in used:
            counts["redundant_site"]+=1
            continue
        if not online.clear_sightline(point,anchor):
            dist=math.dist(point,anchor)
            n=max(1,math.ceil(dist/min(.10,.55*online.cell_m)))
            label=None
            for i in range(n+1):
                f=i/n
                if (1-f)*dist<=max(.18,1.1*online.cell_m): break
                c=online.cell(tuple(a+(b-a)*f for a,b in zip(point,anchor)))
                if not online.observed(c):
                    label="sightline_unobserved"
                elif grid.obstacle[c[1],c[0]]:
                    label="sightline_occupied"
                if label: break
            assert label
            counts[label]+=1
            continue
        direct=geometry._route(online,pose,point,yaw,min(32,budget),False)
        if direct and 1 in direct:
            used.add(key)
            counts["admissible"]+=1
        else:
            counts["no_direct_forward_route"]+=1
            detour=geometry._route(online,pose,point,yaw,min(32,budget),True)
            detour_possible+=bool(detour and 1 in detour)
    assert sum(counts.values())==7
    return counts,detour_possible


def replay(i,spec):
    private=HERE/"private_events"
    events=[json.loads(x) for x in (private/f"{i}.jsonl").read_text().splitlines()]
    selected={e["step"]:e for e in events if e.get("cue_selected")}
    if not selected: return []
    trace=[json.loads(x) for x in (private/f"{i}.poses.jsonl").read_text().splitlines()]
    contacts={e["last_step"] for e in events if e.get("contact")}
    ep,sim=environment(spec)
    record=[]
    try:
        sim.seed(1128)
        state=physical.hs.AgentState()
        state.position=np.float32(ep["start_position"])
        state.rotation=physical.quat_from_coeffs(ep["start_rotation"])
        sim.get_agent(0).set_state(state)
        frames=deque(maxlen=5)
        grid=physical.DepthGrid(physical.policy_pose(sim)[:2])
        for k in range(max(selected)+1):
            obs=sim.step(trace[k]["action"]) if k else sim.get_sensor_observations()
            pose=physical.policy_pose(sim)
            assert np.allclose(sim.get_agent(0).state.position,trace[k]["xyz"],atol=1e-5,rtol=0.)
            assert abs(math.remainder(pose[2]-trace[k]["yaw"],math.tau))<1e-5
            grid.update(pose,obs["depth"])
            if k in contacts: grid.record_contact(frames[-1][0])
            frames.append((pose,obs["depth"].copy(),obs["rgb"][:,:,:3].copy()))
            if k not in selected: continue
            e=selected[k]
            slots=[frames[1],frames[0],frames[3]]
            assert all(hash_plane(s[1])==f["depth_sha256"] and hash_plane(s[2])==f["rgb_sha256"]
                       for s,f in zip(slots,e["physical_scan"]))
            choice={"view_index":e["cue_view_index"],"bbox_norm":e["cue_bbox_norm"],
                    "candidate_id":e["cue_candidate_id"],"category":spec["goal"]}
            anchors,_,_=located_cue(choice,[s[2] for s in slots],grid,pose,1.,[s[:2] for s in slots],90.)
            assert len(anchors)==e["anchors"]==1 and np.allclose(anchors[0]["anchor_xy"],e["cue_anchor_xy"],atol=1e-5)
            budget=min(300-k,24)
            assert len(geometry.propose_sites(grid,pose,1.,anchors,budget=budget))==e["sites"]
            counts,detours=classify(grid,pose,anchors[0]["anchor_xy"],budget)
            assert counts["admissible"]==e["sites"]
            record.append(dict(goal=spec["goal"],view=("right","center","left")[e["cue_view_index"]],
                               accepted_sites=e["sites"],first_gate_counts=dict(counts),
                               additionally_routable_with_native_detour=detours))
    finally: sim.close()
    return record


if __name__=="__main__":
    rows=[r for i,s in enumerate(read(FAMILY/"panel.json")) for r in replay(i,s)]
    reference=read(HERE/"accepted.json")["cue_funnel"]
    assert len(rows)==reference["selected_box"] and sum(bool(r["accepted_sites"]) for r in rows)==reference["with_reachable_view"]
    failed=[r for r in rows if r["accepted_sites"]==0]
    def causes(cohort):
        count=Counter()
        for row in cohort: count.update(row["first_gate_counts"])
        return dict(count)
    result=dict(selected_grants=len(rows),empty_grants=len(failed),empty_grant_candidate_poses=len(failed)*7,
                first_rejection_on_empty_grants=causes(failed),first_rejection_all_grants=causes(rows),
                empty_grants_recoverable_using_same_geometry_and_detour=sum(r["additionally_routable_with_native_detour"]>0 for r in failed),
                all_source_images_and_depths_identical=True,all_generated_counts_identical=True,grants=rows)
    (HERE/"site_filters.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
