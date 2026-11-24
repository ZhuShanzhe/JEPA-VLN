"""Exercise the frozen scorer and execution adapter in a real HM3D engineering scene."""
import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def deny_ip_sockets():
    """Block new IPv4/IPv6 sockets for this process and its scorer subprocess."""
    lib = ctypes.CDLL("libseccomp.so.2")
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
        ctypes.c_void_p,
    ]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]

    class Argument(ctypes.Structure):
        _fields_ = [("index", ctypes.c_uint), ("op", ctypes.c_uint),
                    ("a", ctypes.c_uint64), ("b", ctypes.c_uint64)]

    context = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("could not initialize network isolation")
    try:
        syscall = lib.seccomp_syscall_resolve_name(b"socket")
        for family in (socket.AF_INET, socket.AF_INET6):
            rule = Argument(0, 4, family, 0)  # SCMP_CMP_EQ
            status = lib.seccomp_rule_add_array(
                context, 0x00050000 | errno.EPERM, syscall, 1, ctypes.byref(rule))
            if status != 0:
                raise RuntimeError(f"socket-family rule failed: {status}")
        if lib.seccomp_load(context):
            raise RuntimeError("could not activate network isolation")
    finally:
        lib.seccomp_release(context)
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            test_socket = socket.socket(family)
        except PermissionError:
            continue
        test_socket.close()
        raise RuntimeError("IP sockets unexpectedly enabled")


class SimulatedChoice:
    """Change only the output index in the explicitly labeled simulated case."""
    def __init__(self, channel):
        self.channel = channel

    def __getattr__(self, name):
        return getattr(self.channel, name)

    def recv(self):
        answer = self.channel.recv()
        return dict(answer, index=1)


def make_scene(project):
    from habitat.config import read_write
    from evaluation.main import run_eval
    reference = project / "mapping_runtime/runs/branch_execution_03/result.json"
    receipt = json.loads(reference.read_bytes())
    sample, = receipt["episodes"]
    args = receipt["manifest"]["arguments"]
    group = Path(args["dataset_dir"]) / Path(sample["scene_id"]).name.removesuffix(".basis.glb")
    data = run_eval.load_custom_dataset(*(str(group/(name + ".json")) for name in
                                          ("episodes", "queries", "goals")), goal_type="description")
    episode, = [ep for ep in data if ep.episode_id == sample["episode_id"]]
    data.episodes = [episode]
    cfg = run_eval.load_habitat_config(args["config"])
    root = Path(args["scene_root"])
    run_eval.configure_hm3d_scene_root(cfg, root)
    run_eval.attach_resolved_scene_paths(data, root)
    with read_write(cfg.habitat.simulator):
        cfg.habitat.simulator.scene = episode.scene_id
        cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
    return run_eval.Env(cfg), episode, data


def run_case(env, episode, data, simulated, out):
    import numpy as np
    from evaluation.main.evaluator import _reset_environment
    from most_frontier_probe import ProbeAgent
    from jepa_exec_agent import JepaExecutionFrontier

    _reset_environment(env, episode, data)
    native = env.sim.get_agent(0)
    names = {s.name: k for k, s in native.agent_config.action_space.items()}
    action_names = {1: "move_forward", 2: "turn_left", 3: "turn_right"}

    def rgb():
        frame = env.sim.get_sensor_observations()
        key = next(k for k in frame if "rgb" in k.lower())
        return np.asarray(frame[key])[..., :3].copy()

    agent = JepaExecutionFrontier.__new__(JepaExecutionFrontier)
    agent.channel = agent.worker = None
    agent.output_dir = str(out)
    agent._write_probe = lambda *args, **kwargs: None
    agent._log_event = lambda *args: None
    agent.reach_m = .35
    agent._frontier_pose = (0., 0., 0.)
    agent._frontier_scale = 1.
    agent._recent_frontiers = []
    agent._last_motion_failed = False
    agent.decision_loop = object()
    # Only frontier proposal and the private mapper are replaced. Native
    # activation, follower, scored commands, RGB and feedback use real code.
    clusters = []
    for i, theta in enumerate((0, math.pi/5, -math.pi/5)):
        xy = (8*math.cos(theta), 8*math.sin(theta))
        clusters.append(dict(path=[(0., 0.), xy], world=xy, scale=1.,
                             key=f"k{i}", branch_id=None, path_cost_m=8., utility=1.))
    agent._last_frontier_clusters = clusters
    obs = SimpleNamespace(step_count=0, max_steps=300, episode_id="offline-rank-check",
                          goal_text="find all wall clocks", rgb=rgb(), previous_action=None)

    def logs():
        return [json.loads(line) for path in sorted(out.glob("*.jsonl"))
                for line in path.read_bytes().splitlines()]

    if simulated:
        connect = agent._connect
        def connect_fixture():
            connect()
            agent.channel = SimulatedChoice(agent.channel)
        agent._connect = connect_fixture

    try:
        with patch.object(ProbeAgent, "_feed_frame", lambda *_: None):
            agent.init_episode(obs)
            agent._feed_frame(obs)
            assert agent._activate_frontier(obs, clusters[0], "autonomous EXPLORE")
            initial = native.get_state()
            sent = []
            for index in range(28):
                cmd = int(agent._explore_follow(obs))
                if index < 4:
                    assert cmd == agent.scan_actions[index]
                before = native.get_state()
                env.sim.step(names[action_names[cmd]])
                after = native.get_state()
                displacement = float(np.linalg.norm(after.position-before.position))
                turn = abs(float(np.dot(before.rotation.components, after.rotation.components)))
                blocked = cmd == 1 and bool(env.sim.previous_step_collided)
                agent._last_motion_failed = blocked
                sent.append(dict(command=cmd, forward_m=displacement, cos_half_turn=turn, blocked=blocked))
                obs.previous_action = cmd
                obs.step_count = agent.now = index + 1
                obs.rgb = rgb()
                agent._feed_frame(obs)
                if index == 3:
                    assert np.linalg.norm(initial.position-after.position) < .0001
                    assert abs(np.dot(initial.rotation.components, after.rotation.components)) > .99999
                if index < 4:
                    continue
                rank, = [row for row in logs() if row["event"] == "rank"]
                assert rank["frame_steps"] == [1, 2, 3] and rank["step"] == 4
                assert len(rank["final_scores"]) == 3 and np.isfinite(rank["final_scores"]).all()
                selected = rank["selected"]
                original = rank["original"]
                if selected == original:
                    assert not simulated
                    break
                first = rank["first_difference"]
                assert first is not None
                if len(sent) - 4 < first + 1:
                    continue
                a = rank["plans_native"][rank["ids"].index(selected)]
                b = rank["plans_native"][rank["ids"].index(original)]
                assert [x["command"] for x in sent[4:]] == a[:first+1]
                assert sent[-1]["command"] != b[first]
                assert not blocked and (displacement > .001 or turn < .9999)
                assert any(row["event"] == "commit_end" and row["reason"] == "first_difference_observed"
                           for row in logs()), "missing execution feedback"
                break
            else:
                raise AssertionError("timed out before first different movement")
        return dict(selected_by="simulated_index_1" if simulated else "frozen_jepa",
                    original=rank["original"], selected=rank["selected"],
                    scan_steps=rank["frame_steps"], first_difference=rank["first_difference"],
                    executed_actions=[r["command"] for r in sent], physical_last_action=sent[-1],
                    reached_changed_action=selected != original)
    finally:
        if agent.channel is not None:
            agent.channel.close()
            agent.worker.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("select a new output path")
    deny_ip_sockets()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    adapter = args.project / "jepa_env_v13/most_integration"
    sys.path.insert(0, str(adapter))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    frozen = json.loads((adapter / "no_api_check.json").read_bytes())
    weights = {}
    for source in frozen["checkpoints"]:
        with open(source, "rb") as f:
            hasher = hashlib.sha256()
            for block in iter(lambda: f.read(1048576), b""):
                hasher.update(block)
        assert hasher.hexdigest() == frozen["sha256"][source], "frozen weights changed"
        weights[Path(source).name] = hasher.hexdigest()
    env, episode, data = make_scene(args.project)
    rows = []
    try:
        for simulated in (False, True):
            with tempfile.TemporaryDirectory() as out:
                rows.append(run_case(env, episode, data, simulated, Path(out)))
            if rows[-1]["reached_changed_action"]:
                break
    finally:
        env.close()
    receipt = dict(passed=True, inherited_ip_socket_denial=True, paid_requests=0,
                   environment="HM3D branch_execution_03", inputs="simulated frontier geometry",
                   verified_checkpoints=weights, cases=rows)
    with args.output.open("x") as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
