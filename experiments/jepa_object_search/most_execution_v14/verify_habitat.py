"""Offline physical replay of every archived V13 rank; private traces stay on the data disk."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deny_network(event, args):
    if event == "socket.connect" and args[0].family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError("This physical verification disables network connections")


class RecordedScore:
    def __init__(self, rank):
        self.rank = rank
        self.requests = 0

    def send(self, msg):
        assert [list(p) for p in msg["plans"]] == self.rank["plans_native"]
        assert len(msg["views"]) == 3 and msg["step"] == self.rank["step"]
        self.requests += 1

    def poll(self, timeout):
        return True

    def recv(self):
        return {"index": self.rank["ids"].index(self.rank["selected"])}


def state(env):
    st = env.sim.get_agent_state()
    return np.array(st.position, dtype=float), np.array(st.rotation.components, dtype=float)


def difference(before, after):
    delta = float(np.linalg.norm(after[0] - before[0]))
    dot = abs(np.dot(before[1], after[1]) / (np.linalg.norm(before[1])*np.linalg.norm(after[1])))
    return delta, math.degrees(2*math.acos(float(np.clip(dot,-1,1))))


def replay_until(env, item, data, steps, start):
    from evaluation.main.evaluator import _env_action, _reset_environment
    frame = _reset_environment(env, item, data)
    for step, row in enumerate(steps[:start]):
        assert row["step"] == step and np.linalg.norm(state(env)[0]-row["position_before"]) < .002
        if row["action_role"] == "navigate":
            frame = env.step(_env_action(row["action"]))
    assert np.linalg.norm(state(env)[0]-steps[start]["position_before"]) < .002
    return frame


def scene(doc):
    from habitat.config import read_write
    from evaluation.main import run_eval
    ep, = doc["episodes"]
    arg = doc["manifest"]["arguments"]
    group = Path(arg["dataset_dir"]) / Path(ep["scene_id"]).name.removesuffix(".basis.glb")
    data = run_eval.load_custom_dataset(*(str(group/(n+".json")) for n in ("episodes", "queries", "goals")),
                                        goal_type="description")
    item, = [e for e in data if e.episode_id == ep["episode_id"]]
    data.episodes = [item]
    cfg = run_eval.load_habitat_config(arg["config"])
    root = Path(arg["scene_root"])
    run_eval.configure_hm3d_scene_root(cfg, root)
    run_eval.attach_resolved_scene_paths(data, root)
    with read_write(cfg.habitat.simulator):
        cfg.habitat.simulator.scene = item.scene_id
        cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
    return run_eval.Env(cfg), item, data


def replay_case(env, item, data, rank, activation, ep, output):
    from evaluation.main.evaluator import _env_action
    from agents.navigator import PathFollower
    from jepa_exec_agent import JepaExecutionFrontier
    from most_frontier_probe import ProbeAgent
    step = activation["step"]
    timeline = ep["paper"]["steps"]
    frame = replay_until(env, item, data, timeline, step)
    camera = lambda f: np.asarray(f["rgb"])[..., :3].copy()
    observation = SimpleNamespace(step_count=0, max_steps=300, episode_id=ep["episode_id"],
                                  goal_text="clock" if rank["model_target"]=="clock.n.01" else "painting",
                                  rgb=camera(frame), previous_action=None)
    agent = JepaExecutionFrontier.__new__(JepaExecutionFrontier)
    agent.output_dir = str(output)
    agent.channel = agent.worker = None
    agent._write_probe = lambda *args, **kw: None
    agent._log_event = lambda *args: None
    agent.client = SimpleNamespace(get_all_poses=lambda: (None, None))
    agent._frontier_pose = (0., 0., 0.)
    agent._frontier_slam_pose = None
    agent._frontier_scale = 1.
    agent.reach_m = float(os.environ.get("NAV_REACH_M","0.8"))
    agent._branch_outcomes = {}
    agent._frontier_failures = {}
    agent._recent_frontiers = []
    agent.decision_loop = object()
    agent._last_motion_failed = False
    clusters=[]
    for i,row in enumerate(activation["candidates"]):
        assert row["id"] == f"f{i}"
        clusters.append(dict(path=row["path_forward_left_m"], world=row["endpoint_forward_left_m"],
                             key=row["id"], branch_id=None, scale=1., path_cost_m=row["path_cost_m"],
                             utility=row["utility"]))
    agent._last_frontier_clusters = clusters
    agent.init_episode(observation)
    assert agent.word == rank["model_target"]
    agent.channel = channel = RecordedScore(rank)
    agent.observed_step = step
    agent.now = observation.step_count = step
    agent.pitch = sum(int(r["action"]==4)-int(r["action"]==5) for r in timeline[:step])
    agent.yaw = sum(int(r["action"]==2)-int(r["action"]==3) for r in timeline[:step]) % 12
    events = []
    native_emit = agent.emit
    def record(event, **values):
        events.append(dict(event=event, step=agent.now, **values))
        native_emit(event, **values)
    agent.emit = record
    idx = int(rank["original"][1:])
    native = rank["plans_native"][0]
    selected = rank["plans_native"][rank["ids"].index(rank["selected"])]
    fork = rank["first_difference"]
    commands = []
    physical = []
    with patch.object(ProbeAgent, "_feed_frame", lambda *_: None):
        assert agent._activate_frontier(observation, clusters[idx], activation["source"])
        assert agent.scan is not None, events
        for i in range(4 + (1 if fork is None else fork+1)):
            before = state(env)
            if i == 4:
                origin = before
            cmd = agent._explore_follow(observation)
            assert cmd in (1,2,3), (cmd, events)
            commands.append(cmd)
            frame = env.step(_env_action(cmd))
            after = state(env)
            dist, deg = difference(before,after)
            collided = cmd == 1 and bool(env.sim.previous_step_collided)
            blocked = cmd == 1 and dist < .005
            physical.append(dict(step=agent.now, action=cmd, distance_m=dist, angle_deg=deg,
                                 collided=collided, blocked=blocked, xyz=after[0].tolist(),
                                 rotation=after[1].tolist()))
            observation.previous_action=cmd
            agent.now += 1
            observation.step_count=agent.now
            observation.rgb=camera(frame)
            agent._last_motion_failed=blocked
            agent._feed_frame(observation)
            if i>=4 and any(e["event"]=="commit_end" for e in events):
                break
    assert channel.requests==1 and commands[:4]==[3,2,2,3]
    chosen_state = state(env)
    ends = [e for e in events if e["event"]=="commit_end"]
    if fork is None:
        assert not ends and not any(e["event"]=="commit_start" for e in events)
        status = "unchanged_native"
    else:
        ended, = ends
        status = ended["reason"]
        if status == "first_difference_observed":
            assert commands[4:] == selected[:fork+1]
            assert all(x["distance_m"]>.005 if x["action"]==1 else abs(x["angle_deg"]-30)<.2
                       for x in physical[4:])
        else:
            assert status=="forward_blocked" and physical[-1]["blocked"], ended
    assert agent.commit is None
    post = []
    if status=="forward_blocked":
        post_status = "aborted_at_branch"
    else:
        post_status = "three_native_actions_observed"
        prior_dispatches=sum(e["event"]=="commit_dispatch" for e in events)
        with patch.object(ProbeAgent, "_feed_frame", lambda *_: None):
            for _ in range(3):
                before=state(env)
                cmd=agent._explore_follow(observation)
                if cmd is None:
                    post_status="native_follow_ended"
                    break
                assert cmd in (1,2,3)
                frame=env.step(_env_action(cmd))
                after=state(env)
                dist,deg=difference(before,after)
                blocked=cmd==1 and dist<.005
                if cmd in (2,3):
                    assert abs(deg-30)<.2
                post.append(dict(step=agent.now, action=cmd, distance_m=dist, angle_deg=deg,
                                 xyz=after[0].tolist(), rotation=after[1].tolist(),
                                 collided=cmd==1 and bool(env.sim.previous_step_collided),
                                 blocked=blocked))
                observation.previous_action=cmd
                agent.now+=1
                observation.step_count=agent.now
                observation.rgb=camera(frame)
                agent._last_motion_failed=blocked
                agent._feed_frame(observation)
                if blocked:
                    post_status="native_forward_blocked"
                    break
        assert agent.commit is None and prior_dispatches==sum(e["event"]=="commit_dispatch" for e in events)
    post_state = state(env)
    # Independent counterfactual path from exactly the archived post-scan state.
    replay_until(env,item,data,timeline,rank["step"])
    pos_error, rot_error = difference(origin,state(env))
    assert pos_error < .002 and rot_error < .03,(pos_error,rot_error)
    reference = PathFollower(scale=1., reach_m=agent.reach_m)
    reference.set_path(list(clusters[idx]["path"]))
    def original_step(expected):
        action,arrived=reference.next_action()
        assert not arrived and action==expected,(action,expected,arrived)
        reference.dead_reckon(action)
        env.step(_env_action(action))
    n=len(commands)-4
    for action in native[:n]:
        original_step(action)
    cross_dist, cross_angle = difference(chosen_state,state(env))
    if status=="first_difference_observed":
        assert cross_dist>.005 or cross_angle>1, (cross_dist,cross_angle)
    if status=="unchanged_native":
        assert commands[4:]==native[:1] and cross_dist<.002 and cross_angle<.03
    for action in native[n:n+len(post)]:
        original_step(action)
    assert len(native[n:n+len(post)])==len(post)
    post_dist, post_deg=difference(post_state,state(env))
    if status=="unchanged_native":
        assert post_dist<.002 and post_deg<.03
    return dict(step=rank["step"], archived_selection_changed=fork is not None, fork_offset=fork,
                selected=rank["selected"], original=rank["original"], status=status,
                common_start_position_error_m=pos_error, common_start_angle_error_deg=rot_error,
                separation_m=cross_dist, separation_deg=cross_angle,
                handback_status=post_status, handback_steps=len(post),
                handback_separation_m=post_dist, handback_separation_deg=post_deg,
                post_branch_physical=post, simulated_decisions=channel.requests, native_prefix=native[:n+len(post)],
                scan_and_selected_actions=commands, physical=physical, events=events)


def main(project, output):
    output.mkdir(parents=True, exist_ok=False)
    pilot = project/"mapping_runtime/pilots/jepa_round13_horizon_five"
    cases = []
    inputs = []
    for row in read(pilot/"panel.json")["rows"]:
        ep_id = row["episode_id"]
        ref = read(pilot/"comparison/accepted/second"/(ep_id+".json"))
        result=Path(ref["result"])
        assert digest(result)==ref["result_sha256"]
        roots=result.parent/"agent"
        ranks=[r for r in rows(roots/"v13_routes.jsonl") if r["event"]=="rank"]
        inputs.append(dict(ordinal=row["ordinal"], ranks=len(ranks), result_sha256=digest(result),
                           rank_log_sha256=digest(roots/"v13_routes.jsonl"),
                           frontier_log_sha256=digest(roots/"frontier_probe.jsonl")))
        if not ranks:
            continue
        probe=rows(roots/"frontier_probe.jsonl")
        doc=read(result)
        env,item,data=scene(doc)
        try:
            for rank in ranks:
                activations=[p for p in probe if p["step"]==rank["step"]-4 and
                             p["event"]=="activation" and p["selected_id"]==rank["original"]]
                activation=activations[-1]
                dest=output/(ep_id+"_"+str(rank["step"]))
                dest.mkdir()
                report=replay_case(env,item,data,rank,activation,doc["episodes"][0],dest)
                (dest/"trace.json").write_text(json.dumps(report,indent=2))
                cases.append(dict(ordinal=row["ordinal"],**{k:v for k,v in report.items()
                                  if k not in ("events","physical","post_branch_physical","native_prefix","scan_and_selected_actions")}))
                print("TRACE",row["ordinal"],rank["step"],report["status"],flush=True)
        finally:
            env.close()
    assert len(cases)==5 and sum(c["archived_selection_changed"] for c in cases)==4
    status=Counter(c["status"] for c in cases)
    assert status=={"first_difference_observed":4,"unchanged_native":1}, status
    summary=dict(cases=cases, counts=dict(status), input_digests=inputs, new_paid_requests=0,
                 source="all ranks in the archived five V13 episodes; metric path geometry; mocked rank responses and mapper poses; real adapter, PathFollower, Habitat observations",
                 verified=__file__, verifier_sha256=digest(Path(__file__)),
                 adapter_sha256=digest(Path(__file__).with_name("jepa_exec_agent.py")))
    (output/"physical_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary["counts"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    args=parser.parse_args()
    sys.addaudithook(deny_network)
    main(args.project,args.output)
