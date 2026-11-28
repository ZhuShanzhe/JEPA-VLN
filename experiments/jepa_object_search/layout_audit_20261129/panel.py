"""Independent render, verification and scoring for the frozen local 2 x 2 panel."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parent
GRID = os.environ.get("JEPA_PANEL_REVISION","first")=="grid"
HERE = EXPERIMENT/"grid" if GRID else EXPERIMENT
SOURCE = ROOT/"new_view_20261129"
BASE = ROOT/"refined_confirmation_20261129"
ORIGINAL = ROOT/"reactive_view_20261129"
ROUTE = ROOT/("layout_detour_v2_20261129" if GRID else "layout_detour_20261129")
AREA = ROOT/"clarity_fix_20261129"
sys.path[:0] = list(map(str, (AREA, BASE, ORIGINAL, SOURCE)))


def read(path):
    return json.loads(path.read_text())


def houses():
    return read(SOURCE/"view_design.json")["cohorts"]["blind"]


def panel(house):
    for p in sorted((ORIGINAL/"private_roll"/house).glob("*.json"), key=lambda p:int(p.stem)):
        row = read(p)
        if row["focus"]:
            yield p.stem, row


def render(index):
    from sample_views import ACTION, ASSETS, SOURCE_H, camera, inventory, simulator
    house = houses()[index]
    obs_out = HERE/"private_inputs"/house
    label_out = HERE/"private_scoring"/house
    for directory in (obs_out, label_out):
        directory.mkdir(parents=True, exist_ok=True)
    sim = simulator(house)
    try:
        inventory(sim, ASSETS)
        agent = sim.get_agent(0)
        for src, old in panel(house):
            row = read(ROUTE/"private_motion"/house/(src+".json"))
            packets = row["events"]
            assert row["budget"] == old["budget"] and len(packets)-1 <= old["budget"]
            with np.load(SOURCE/"private_render_blind60"/("blind_"+house)/(src+"_sensor.npz")) as s:
                pos, yaw, oid, cue = s["position"].copy(), float(s["yaw"]), int(s["oid"]), s["box"].copy()
            obs = camera(sim, pos, yaw, SOURCE_H, 0.)
            rgb, depth, odom, actions, masks = [], [], [], [], []
            for step, ev in enumerate(packets):
                a = int(ev["action"])
                if step:
                    prev = agent.state.position.copy()
                    hit = agent.act(ACTION[a])
                    stalled = np.linalg.norm((agent.state.position-prev)[[0,2]]) < .23
                    assert bool(a == 1 and (hit or stalled)) == ev["bump"]
                    obs = sim.get_sensor_observations()
                else:
                    assert a == 0
                d = agent.state.position-pos
                q = agent.state.rotation
                theta = 2*math.atan2(q.y,q.w)-yaw
                local = (math.cos(yaw)*d[0]-math.sin(yaw)*d[2],
                         -math.sin(yaw)*d[0]-math.cos(yaw)*d[2],
                         math.atan2(math.sin(theta),math.cos(theta)))
                assert np.allclose(local, ev["pose"], atol=1e-5)
                mask = obs["ids"] == oid
                assert abs(int(mask.sum())-ev["px"]) <= 1
                rgb.append(obs["rgb"][...,:3].copy())
                depth.append(obs["depth"].squeeze().copy())
                odom.append(local)
                actions.append(a)
                masks.append(np.packbits(mask.reshape(-1)))
            np.savez(obs_out/(src+".npz"), rgb=rgb, depth=depth, odom=odom, action=actions, cue=cue, budget=old["budget"])
            np.savez(label_out/(src+".npz"), packed=masks)
    finally:
        sim.close()


def decisions(arm, index):
    from memory_refit import RefinedMemory
    from inspection_session import save_confirmation
    from threadpoolctl import threadpool_limits
    if arm.endswith("new"):
        if GRID:
            from strict_rescue import StrictRescueMemory as AreaMemory
        else:
            from area_verifier import AreaMemory
    house = houses()[index]
    source_root = BASE if arm.startswith("old") else HERE
    paths = sorted((source_root/"private_inputs"/house).glob("*.npz"))
    assert len(paths) == len(list(panel(house)))
    with threadpool_limits(limits=1):
        for path in paths:
            dest = HERE/"private_decisions"/arm/house/path.stem
            dest.mkdir(parents=True, exist_ok=True)
            with np.load(path) as s:
                rgb, depths, odom = s["rgb"], s["depth"], s["odom"]
                source = (rgb[0], depths[0], s["cue"])
                if arm.endswith("new"):
                    verifier = AreaMemory(*source)
                else:
                    verifier = RefinedMemory(*source, use_memory=True, use_clarity=True)
                assert len(rgb)-1 <= int(s["budget"])
            step = box = image_hash = None
            for t, (image, z, pose) in enumerate(zip(rgb, depths, odom)):
                decision = verifier.observe(image, z, pose)
                if decision["accept"]:
                    assert decision["p_clear"]*decision["p_same"] >= .835-1e-9
                    save_confirmation(dest, {"rgb":image}, decision, t)
                    receipt = read(dest/"confirmed/evidence.json")
                    step, box, image_hash = t, receipt["box"], receipt["rgb_sha256"]
                    break
            result = dict(actions_used=t, step=step, box=box, frame_sha256=image_hash)
            (dest/"decision.json").write_text(json.dumps(result))


def grade(arm, house, src, old):
    from score_paired import box_grade
    from PIL import Image
    if arm == "old_old":
        directory = BASE/"private_paired/full"/house/src
    else:
        directory = HERE/"private_decisions"/arm/house/src
    d = read(directory/"decision.json")
    source_root = BASE if arm.startswith("old") else HERE
    with np.load(source_root/"private_scoring"/house/(src+".npz")) as s:
        masks = np.unpackbits(s["packed"], axis=1).reshape(-1,192,192)
    t = d["step"]
    assert 0 <= d["actions_used"] <= old["budget"]
    if t is None:
        assert d["actions_used"] == len(masks)-1
        status = "unreported"
    else:
        assert t == d["actions_used"] and 0 <= t < len(masks)
        l,u,r,b = d["box"]
        assert 0<=l<r<=192 and 0<=u<b<=192
        n, inside, iou = box_grade(masks[t], d["box"])
        status = "correct" if n>=369 and inside>=.5*n and iou>=.30 else "false"
        evidence = directory if arm=="old_old" else directory/"confirmed"
        with Image.open(evidence/"frame.png") as raw, Image.open(evidence/"crop.png") as crop, np.load(source_root/"private_inputs"/house/(src+".npz")) as inputs:
            frame = np.asarray(raw)
            assert np.array_equal(frame,inputs["rgb"][t])
            assert np.array_equal(frame[u:b,l:r], np.asarray(crop))
            assert hashlib.sha256(frame.tobytes()).hexdigest() == d["frame_sha256"]
    clear_steps = np.flatnonzero(masks.sum(axis=(1,2))>=369)
    return status, d["actions_used"], bool(len(clear_steps)), bool(len(clear_steps) and clear_steps[0]<=d["actions_used"])


def summarize():
    arms = ("old_old","old_new","new_old","new_new")
    totals = {a:Counter() for a in arms}
    per_house = []
    transitions = {a:Counter() for a in arms}
    groups = {}
    routing = Counter()
    for house in houses():
        counts = {a:Counter() for a in arms}
        for src, row in panel(house):
            status = {}
            for a in arms:
                s, cost, seen, seen_before_stop = grade(a,house,src,row)
                status[a] = s
                for tally in (totals[a], counts[a]):
                    tally[s] += 1
                    tally["actions"] += cost
                    tally["ever_clear_on_no_stop_route"] += int(seen)
                    tally["clear_before_stop"] += int(seen_before_stop)
            old_event = row["new_confirmation"]
            if old_event is not None:
                cohort = "historical_correct" if row["events"][old_event]["correct"] else "historical_false"
            else:
                cohort = "historical_missed_clear" if any(e["px"]>=369 for e in row["events"]) else "historical_no_clear"
            witness = read(EXPERIMENT/"private_witness"/house/(src+".json"))["kind"]
            labels = [cohort, "witness_"+witness]
            if witness != "unverified":
                labels.append("witnessed_in_budget_before_test")
            if cohort != "historical_false" and status["old_old"] == "false":
                labels.append("previous_five_new_false")
            for label in labels:
                band = groups.setdefault(label,{a:Counter() for a in arms})
                for a in arms:
                    band[a][status[a]] += 1
            for a in arms:
                transitions[a][status["old_old"]+"->"+status[a]] += 1
            current = read(ROUTE/"private_motion"/house/(src+".json"))
            old_seen = any(e["px"]>=369 for e in row["events"])
            new_seen = any(e["px"]>=369 for e in current["events"])
            routing[str(old_seen)+"->"+str(new_seen)] += 1
        per_house.append(counts)
    assert all(sum(t[a] for a in ("correct","false","unreported"))==94 for t in totals.values())
    assert (totals["old_old"]["correct"], totals["old_old"]["false"], totals["old_old"]["actions"])==(30,9,1783)
    assert sum(groups["previous_five_new_false"]["old_old"].values()) == 5
    result = dict(groups=94, coding=("arm = route_verifier; new = grid route or strict multi-view verifier" if GRID else "arm = route_verifier; new = first layout detour or first corrected area verifier"), results=totals, subset=groups, transitions_vs_prior=transitions, by_house=per_house, route_opportunity_transition=routing)
    (HERE/"factorial_results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(totals)


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser()
    p.add_argument("stage",choices=("render","decide","summarize"))
    p.add_argument("--house",type=int,choices=(0,1,2))
    p.add_argument("--arm",choices=("old_new","new_old","new_new"))
    args = p.parse_args()
    if args.stage == "summarize":
        summarize()
    elif args.stage == "render":
        render(args.house)
    else:
        decisions(args.arm,args.house)
