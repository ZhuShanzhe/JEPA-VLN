"""Replay the streaming interface in a live Habitat simulator, checking the independently saved closed-loop log."""
from collections import Counter
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "new_view_20261129"
sys.path.insert(0, str(PRIOR))


def simulator_loop(house, packet):
    from sample_views import ACTION, ASSETS, SOURCE_H, camera, inventory, simulator
    sim = simulator(house)
    try:
        inventory(sim, ASSETS)
        agent = sim.get_agent(0)
        for message in sys.stdin:
            command, value = message.strip().split()
            if command == "source":
                path = PRIOR / "private_render_blind60" / ("blind_" + house) / (value + "_sensor.npz")
                with np.load(path) as archive:
                    start, yaw = archive["position"], float(archive["yaw"])
                obs = camera(sim, start, yaw, SOURCE_H, 0.)
                contact = False
            else:
                assert command == "act"
                action = int(value)
                previous = agent.state.position.copy()
                hit = agent.act(ACTION[action])
                moved = np.linalg.norm((agent.state.position - previous)[[0, 2]])
                contact = action == 1 and (hit or moved < .23)
                obs = sim.get_sensor_observations()
            state = agent.state
            delta = state.position - start
            heading = 2 * math.atan2(state.rotation.y, state.rotation.w) - yaw
            pose = (math.cos(yaw)*delta[0] - math.sin(yaw)*delta[2],
                    -math.sin(yaw)*delta[0] - math.cos(yaw)*delta[2],
                    math.atan2(math.sin(heading), math.cos(heading)))
            np.savez(packet, rgb=obs["rgb"][..., :3], depth=obs["depth"].squeeze(),
                     pose=pose, bump=bool(contact))
            print("observation_ready", flush=True)
    finally:
        sim.close()


def receive(proc, packet, command):
    proc.stdin.write(command + "\n")
    proc.stdin.flush()
    for line in proc.stdout:
        if line.strip() == "observation_ready":
            break
    else:
        raise RuntimeError("simulator exited before supplying action result")
    with np.load(packet) as archive:
        x,z,yaw = map(float, archive["pose"])
        return dict(rgb=archive["rgb"], depth=archive["depth"], x=x, z=z, yaw=yaw, bump=bool(archive["bump"]))


def behaviors(row):
    at = row["new_confirmation"]
    end = at if at is not None else len(row["events"])-1
    collision = any(e["bump"] for e in row["events"][:end])
    return dict(initial=at == 0, moving_confirmation=at is not None and at > 2,
                confirmed_after_contact=collision and at is not None,
                budget_exhausted=at is None and end == row["budget"],
                options_exhausted=at is None and end < row["budget"])


def verify():
    from inspection_session import ReactiveInspection
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(2)
    inspection = ReactiveInspection.load()
    counts = Counter()
    coverage = Counter()
    houses = json.loads((PRIOR / "view_design.json").read_text())["cohorts"]["blind"]
    with tempfile.TemporaryDirectory(dir=HERE, prefix=".stream-test-") as folder, threadpool_limits(limits=2):
        base = Path(folder)
        for house in houses:
            directory = HERE / "private_roll" / house
            known = [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"), key=lambda x:int(x.stem))]
            chosen = {}
            for condition in behaviors(known[0]):
                for row in known:
                    if behaviors(row)[condition] and row["source"] not in chosen:
                        chosen[row["source"]] = row
                        break
            packet = base / "frame.npz"
            env = dict(os.environ, MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            for key in ("PYTHONPATH","LD_LIBRARY_PATH"):
                env.pop(key,None)
            py = os.environ.get("JEPA_HABITAT_PYTHON", str(HERE.parent/"simulation/habitat-env/bin/python"))
            proc = subprocess.Popen([py,"-u",str(Path(__file__).resolve()),"--house",house,"--packet",str(packet)],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
            try:
                for n,row in chosen.items():
                    stream = []
                    events = row["events"]
                    def read(command, action):
                        t = len(stream)
                        if t >= len(events) or events[t]["action"] != action:
                            raise AssertionError(f"action mismatch {house} source={n} step={t} expected={events[t]['action'] if t < len(events) else 'stop'} actual={action}")
                        frame = receive(proc,packet,command)
                        assert frame["bump"] == events[t]["bump"] and np.allclose(
                            [frame["x"],frame["z"],frame["yaw"]], events[t]["pose"], atol=1e-5)
                        stream.append(frame)
                        return frame
                    source = read(f"source {n}", 0)
                    with np.load(PRIOR/"private_render_blind60"/("blind_"+house)/(str(n)+"_sensor.npz")) as archive:
                        box = archive["box"].tolist()
                    out = base / f"{house}_{n}"
                    verdict = inspection.run(source, box, row["budget"]-18, row["budget"],
                        lambda a: read(f"act {a}",a), out)
                    at = row["new_confirmation"]
                    expected_end = at if at is not None else len(events)-1
                    assert verdict["actions_used"] == len(stream)-1 == expected_end
                    if at is not None:
                        assert verdict["status"] == "confirmed"
                        bundle = Path(verdict["evidence_directory"])
                        receipt = json.loads((bundle/"evidence.json").read_text())
                        assert receipt["step"] == at and receipt["box"] == events[at]["box"]
                        assert receipt["p_clear"] == events[at]["p_clear"] and receipt["p_same"] == events[at]["p_same"]
                        for a,b in (("frame.png",f"{n}_step{at}.png"),("crop.png",f"{n}_step{at}_crop.png")):
                            with Image.open(bundle/a) as produced, Image.open(directory/b) as archived:
                                assert np.array_equal(produced,archived)
                        counts["frame_and_crop_pairs_exact"] += 1
                    else:
                        assert verdict["status"] == ("budget_exhausted" if expected_end == row["budget"] else "no_supported_view")
                        assert not (out/"confirmed").exists()
                    for name, matches in behaviors(row).items():
                        if matches: coverage[name] += 1
                    counts.update(requests=1,actions=expected_end,
                                  post_contact_actions=sum(f["bump"] for f in stream[:-1]))
            finally:
                proc.stdin.close()
                proc.wait(timeout=75)
                if proc.returncode:raise RuntimeError("simulator did not exit cleanly")
    assert all(coverage[k]>0 for k in ("initial","moving_confirmation","confirmed_after_contact","budget_exhausted","options_exhausted"))
    result = dict(houses=len(houses), counts=counts, behavior_coverage_nonexclusive=coverage,
                  criteria="Exact primitive-by-primitive agreement to first accepted frame or stop against frozen JEPA closed-loop logs; independently executed Habitat simulator, RGB-D/odometry only to loaded policy. Identical confirmation step, rectangle, confidences and full-frame/crop pixels. Behavior-stratified interface smoke test only.")
    (HERE/"stream_interface_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    os.umask(0o077)
    parser=argparse.ArgumentParser()
    parser.add_argument("--house");parser.add_argument("--packet")
    args=parser.parse_args()
    if args.house:simulator_loop(args.house,args.packet)
    else:verify()
