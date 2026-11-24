"""Independent local counterfactuals on archived HM3D junctions, with IP disabled."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from verify_habitat import difference, digest, read, rows, scene, state
from verify_rank import deny_ip_sockets


def write(path, obj):
    with path.open("x") as fp:
        json.dump(obj, fp, indent=2, allow_nan=False)
        fp.write("\n")


def motion_from_rgb(agent, obs):
    """Reproduce native MappingAgent RGB-only stall feedback, without external mapper I/O."""
    visual = obs.rgb[::16, ::16, :3].astype(np.int16)
    same = (agent._last_visual is not None and obs.previous_action == 1 and
            float(np.abs(visual-agent._last_visual).mean()) < float(os.getenv("MAPPING_STUCK_RGB_DELTA", "1.0")))
    agent._visual_stall_count = agent._visual_stall_count + 1 if same else 0
    agent._last_motion_failed = agent._visual_stall_count >= max(1,int(os.getenv("MAPPING_STUCK_CONFIRM_STEPS","2")))
    agent._last_visual = visual


def start_agent(kind, output, ep):
    agent = kind.__new__(kind)
    agent.output_dir = str(output)
    agent.channel = agent.worker = None
    agent._write_probe = agent._log_event = lambda *_, **kw: None
    agent.client = SimpleNamespace(get_all_poses=lambda: (None,None))
    agent._frontier_pose = (0.,0.,0.)
    agent._frontier_slam_pose = None
    agent._frontier_scale = 1.
    agent.reach_m = float(os.getenv("NAV_REACH_M","0.8"))
    agent._branch_outcomes = {}
    agent._frontier_failures = {}
    agent._recent_frontiers = []
    agent.decision_loop = object()
    agent._last_visual = None
    agent._visual_stall_count = 0
    agent._last_motion_failed = False
    obs = SimpleNamespace(step_count=0, max_steps=ep["max_steps_budget"], episode_id=ep["episode_id"],
                          goal_text=ep["goal_text"], previous_action=None)
    agent.init_episode(obs)
    return agent, obs


def observed(agent, obs, frame, action=None):
    if action is not None:
        agent.now += 1
        obs.step_count = agent.now
        obs.previous_action = int(action)
    obs.rgb = np.asarray(frame["rgb"])[..., :3].copy()
    agent._feed_frame(obs)


def clusters_from(activation):
    return [dict(path=r["path_forward_left_m"], world=r["endpoint_forward_left_m"],
                 key=r["id"], branch_id=None, scale=1., path_cost_m=r["path_cost_m"], utility=r["utility"])
            for r in activation["candidates"]]


def prefix(env, item, data, agent, obs, timeline, start, old_ranks):
    from evaluation.main.evaluator import _env_action, _reset_environment
    from v13_inputs import camera
    frame = _reset_environment(env, item, data)
    observed(agent, obs, frame)
    for row in timeline[:start]:
        assert row["step"] == agent.now and np.linalg.norm(state(env)[0]-row["position_before"]) < .002
        if row["action_role"] == "navigate":
            frame = env.step(_env_action(row["action"]))
        observed(agent, obs, frame, row["action"])
        # Replay prior *archived* scans to initialize memory for this one independent intervention.
        for old in old_ranks:
            if agent.now in old["frame_steps"]:
                last = agent.history[-1]
                agent.history[-1] = (*last[:3], camera(obs.rgb,agent.pitch))
            if agent.now == old["step"]:
                views = [r[3] for r in agent.history if r[0] in old["frame_steps"]]
                if agent.channel is None:
                    agent._connect()
                agent.channel.send(dict(history=agent.history,step=agent.now,views=views,plans=old["plans_native"],
                                        remaining=obs.max_steps-agent.now))
                assert agent.channel.poll(240)
                agent.channel.recv()
                agent.history = []
    assert np.linalg.norm(state(env)[0]-timeline[start]["position_before"]) < .002


def simulate(env, agent, obs, action):
    from evaluation.main.evaluator import _env_action
    before = state(env)
    frame = env.step(_env_action(action))
    distance, angle = difference(before,state(env))
    observed(agent, obs, frame, action)
    return dict(action=int(action), dist=distance, deg=angle, native_stall=bool(agent._last_motion_failed),
                collision=bool(action==1 and env.sim.previous_step_collided))


def replay(env,item,data,agent,obs,ep,act,old_ranks,output,forced=False):
    from evaluation.main.evaluator import _env_action
    from verify_habitat import replay_until
    timeline = ep["paper"]["steps"]
    start = act["step"]
    prefix(env,item,data,agent,obs,timeline,start,[r for r in old_ranks if r["step"] < start])
    origin = state(env)
    clusters = clusters_from(act)
    agent._last_frontier_clusters = clusters
    assert [r["nominal_action_ids"] for r in agent._rows()] == [r["nominal_action_ids"] for r in act["candidates"]]
    assert agent._activate_frontier(obs,clusters[int(act["selected_id"][1:])],act["source"])
    events_path = output/"v13_routes.jsonl"
    eligibility, = [e for e in rows(events_path) if e["event"]=="eligibility"]
    result = dict(step=start, input_candidates=len(clusters), reason=eligibility["reason"], forced=forced,
                  candidate_ids=eligibility["ids"],
                  delayed_watch_started=any(e["event"]=="deferred_start" for e in rows(events_path)))
    if agent.scan is None:
        assert eligibility["reason"]
        assert agent.commit is None
        action = agent._explore_follow(obs)
        native = act["candidates"][int(act["selected_id"][1:])]["nominal_action_ids"]
        assert (action is None and not native) or action==native[0]
        if action is not None:
            result["native_motion"]=simulate(env,agent,obs,action)
        return dict(result,status="native_fallback")
    plans = [list(p) for p in agent.scan["plans"]]
    assert len(plans) >= 2 and all(p for p in plans)
    if forced:
        class Last:
            def send(self,msg):
                assert list(map(list,msg["plans"])) == plans
            def poll(self,t): return True
            def recv(self): return {"index":len(plans)-1}
        assert agent.channel is None
        agent.channel = Last()
    physical=[]
    for expected in (3,2,2,3):
        action = agent._explore_follow(obs)
        assert action==expected
        physical.append(simulate(env,agent,obs,action))
    assert difference(origin,state(env))[0]<.002 and difference(origin,state(env))[1]<.05
    cmd = agent._explore_follow(obs)
    events = rows(events_path)
    rank, = [e for e in events if e["event"]=="rank"]
    fork = rank["first_difference"]
    if not forced:
        scores = np.asarray(rank["final_scores"])
        assert scores.shape == (len(plans),) and np.isfinite(scores).all()
        assert int(rank["index"])==int(scores.argmax())
    assert rank["original"]==act["selected_id"] and rank["selected"]==rank["ids"][rank["index"]]
    if forced:
        assert rank["selected"] == rank["ids"][-1]
    chosen=plans[rank["index"]]
    for i in range(1 if fork is None else fork+1):
        if i:
            cmd=agent._explore_follow(obs)
        if cmd is None:
            break
        assert cmd in (1,2,3)
        physical.append(simulate(env,agent,obs,cmd))
        if agent.commit is None:
            break
    chosen_state = state(env)
    events=rows(events_path)
    ends=[e for e in events if e["event"]=="commit_end"]
    post=physical[4:]
    if fork is None:
        assert not ends and chosen==plans[0] and len(post)==1 and post[0]["action"]==chosen[0]
        status="kept_native"
    else:
        end,=ends
        status=end["reason"]
        assert status in ("first_difference_observed","forward_blocked","native_path_ended")
        if status=="first_difference_observed":
            assert [a["action"] for a in post]==chosen[:fork+1]
            assert end["confirmed_steps"] == fork+1
    assert agent.commit is None
    replay_until(env,item,data,timeline,start)
    for action in plans[0][:len(post)]:
        env.step(_env_action(action))
    dist,angle = difference(chosen_state,state(env))
    changed = dist>.005 or angle>1.
    if status=="kept_native": assert dist<.002 and angle<.05
    return dict(result, status=status, selected=rank["selected"], original=rank["original"], fork=fork,
                lengths=list(map(len,plans)), physical_divergence=bool(changed), separation_m=dist,
                separation_deg=angle, physical=physical)


def main(project, out, controls=False):
    deny_ip_sockets()
    from most_frontier_probe import ProbeAgent
    from variable_agent import VariableFrontier
    pilot=project/"mapping_runtime/pilots/jepa_evidence_v15_first2"
    out.mkdir(parents=True,exist_ok=False)
    cases=[]
    for row in read(pilot/"panel.json")["rows"]:
        if controls and row["ordinal"]!=14:
            continue
        ref=read(pilot/"comparison/accepted/second"/(row["episode_id"]+".json"))
        result=Path(ref["result"])
        assert digest(result)==ref["result_sha256"]
        ep,=read(result)["episodes"]
        logs=result.parent/"agent"
        archived=rows(logs/"v13_routes.jsonl")
        old_ranks=[r for r in archived if r["event"]=="rank"]
        gates=[r for r in archived if r["event"]=="eligibility"]
        pipe=rows(logs/"frontier_probe.jsonl")
        env,item,data=scene(read(result))
        try:
            for event in gates:
                if controls and event["step"] not in (62,237):
                    continue
                act,=[r for r in pipe if r["event"]=="activation" and r["activated"] and r["step"]==event["step"]
                      and r["selected_id"]==event["original"] and r["source"]==event["source"]]
                modes=[controls]
                for force in modes:
                    target=out/str(row["ordinal"])/(str(event["step"])+("-forced" if force else ""))
                    target.mkdir(parents=True)
                    agent,obs=start_agent(VariableFrontier,target,ep)
                    try:
                        with patch.object(ProbeAgent,"_feed_frame", motion_from_rgb):
                            case=replay(env,item,data,agent,obs,ep,act,old_ranks,target,force)
                    finally:
                        if agent.worker:
                            agent.channel.close()
                            agent.worker.wait(timeout=30)
                    case.update(ordinal=row["ordinal"], v15_reason=event["reason"])
                    write(target/"result.json",case)
                    cases.append(case)
                    print({k:case.get(k) for k in ("ordinal","step","forced","status","lengths","physical_divergence")},flush=True)
        finally:
            env.close()
    natural=[c for c in cases if not c["forced"]]
    assert len(cases)==(2 if controls else 12)
    if controls:
        assert all(c["status"]!="native_fallback" for c in cases)
    result=dict(passed=True, production_interventions=len(natural), forced_controls=len(cases)-len(natural),
                results=[{k:v for k,v in c.items() if k not in ("native_motion","physical")} for c in cases],
                statuses=dict(Counter(c["status"] for c in natural)), ip_sockets_denied=True)
    write(out/"receipt.json",result)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("project",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--controls",action="store_true")
    args=parser.parse_args()
    main(args.project,args.output,args.controls)
