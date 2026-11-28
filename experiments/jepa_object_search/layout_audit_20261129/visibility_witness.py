"""Offline-only positive witnesses for the frozen panel. A negative search is inconclusive."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "new_view_20261129"
OLD = ROOT / "reactive_view_20261129"
sys.path.insert(0, str(SOURCE))
from sample_views import ACTION, ASSETS, SOURCE_H, camera, inventory, simulator
from view_geometry import turns


def search(sim, old, pose, yaw, target, center):
    agent = sim.get_agent(0)
    budget = old["budget"]

    def restore(state):
        agent.set_state(state, reset_sensors=False, infer_sensor_states=False)

    def walk(action):
        before = agent.state.position.copy()
        contact = agent.act(ACTION[action])
        return action != 1 or (not contact and np.linalg.norm((agent.state.position-before)[[0, 2]]) >= .23)

    def pixels():
        return int((sim.get_sensor_observations()["ids"] == target).sum())

    camera(sim, pose, yaw, SOURCE_H, 0)
    start = agent.get_state()

    def certify(actions, kind):
        restore(start)
        for action in actions:
            walk(action)
        count = pixels()
        assert count >= 369 and len(actions) <= budget
        return {"kind": kind, "steps": len(actions), "pixels": count, "actions": actions}

    for i, event in enumerate(old["events"]):
        if event["px"] >= 369:
            return certify([e["action"] for e in old["events"][1:i+1]], "previous_route")

    for a in [2, 3]:
        walk(a)
    after_scan = agent.get_state()

    def test_view(path, heading):
        state = agent.get_state()
        dx, _, dz = center - state.position
        toward = round((math.atan2(-dx, -dz) - yaw) / (math.pi/6))
        for final in dict.fromkeys((heading, toward, toward-1, toward+1)):
            turn = list(turns(final-heading))
            if len(path) + len(turn) > budget:
                continue
            restore(state)
            for a in turn:
                walk(a)
            if pixels() >= 369:
                return path + turn
        restore(state)
        return None

    branches = []
    for heading in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6):
        restore(after_scan)
        opening = list(turns(heading))
        for a in opening:
            walk(a)
        path = [2, 3] + opening
        for i in range(13):
            winner = test_view(path, heading)
            if winner is not None:
                return certify(winner, "one_heading")
            if i in (1, 2, 4, 6, 8):
                branches.append((agent.get_state(), path.copy(), heading))
            if i == 12 or len(path) >= budget or not walk(1):
                break
            path.append(1)

    for state, first_leg, heading in branches:
        for offset in (2, -2, 3, -3, 1, -1, 4, -4):
            restore(state)
            bend = list(turns(offset))
            for a in bend:
                walk(a)
            path = first_leg + bend
            for _ in range(min(10, budget-len(path))):
                if not walk(1):
                    break
                path.append(1)
                winner = test_view(path, heading + offset)
                if winner is not None:
                    return certify(winner, "two_headings")
    return {"kind": "unverified", "steps": None}


def run(index):
    houses = json.loads((SOURCE/"view_design.json").read_text())["cohorts"]["blind"]
    house = houses[index]
    destination = HERE/"private_witness"/house
    destination.mkdir(parents=True, exist_ok=True)
    sim = simulator(house)
    try:
        objects = inventory(sim, ASSETS)
        centers = {int(sid): center for group in objects.values() for sid, _, center in group}
        for f in sorted((OLD/"private_roll"/house).glob("*.json"), key=lambda f:int(f.stem)):
            prev = json.loads(f.read_text())
            path = destination/f.name
            if not prev["focus"] or path.exists():
                continue
            source = SOURCE/"private_render_blind60"/("blind_"+house)/(f.stem+"_sensor.npz")
            with np.load(source) as s:
                target = int(s["oid"])
                witness = search(sim, prev, s["position"], float(s["yaw"]), target, centers[target])
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(witness))
            tmp.replace(path)
    finally:
        sim.close()


def summary():
    counts = Counter(json.loads(f.read_text())["kind"] for f in (HERE/"private_witness").glob("*/*.json"))
    assert sum(counts.values()) == 94
    result = dict(groups=94, by_witness=counts, scope="First check the historical route. Only if it never met the area criterion, try physically executing fixed one-heading and then two-heading paths, initial two scan actions included, within the original 22/26/30-step cap. Replay each positive witness from the original start. Counts indicate witnessed visibility of the target, not successful detection. Search misses have unknown feasibility. Target labels used by this offline evaluator alone.")
    (HERE/"witness_counts.json").write_text(json.dumps(result, indent=2)+"\n")
    print(result)


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--house", type=int, choices=(0, 1, 2))
    index = parser.parse_args().house
    if index is None:
        summary()
    else:
        run(index)
