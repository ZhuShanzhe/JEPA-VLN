"""Paired equal-allowance opportunity audit at all scored historical HM3D junctions."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys
from unittest.mock import patch
import numpy as np
from verify_rank import deny_ip_sockets
from verify_habitat import scene, replay_until


def load(p):
    return json.loads(p.read_bytes())


def lines(p):
    return [json.loads(line) for line in p.read_bytes().splitlines() if line]


def semantic_scene(doc):
    from evaluation.main import run_eval
    from habitat.config import read_write
    from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig
    from omegaconf import OmegaConf
    create = run_eval.Env
    def augmented(cfg):
        sim = cfg.habitat.simulator
        rgb = sim.agents.main_agent.sim_sensors.rgb
        sem = OmegaConf.structured(HabitatSimSemanticSensorConfig)
        for field in ("hfov","position","orientation"):
            sem[field] = rgb[field]
        sem.width,sem.height = rgb.width//4,rgb.height//4
        assert sem.width==sem.height==320
        with read_write(sim):
            sim.agents.main_agent.sim_sensors["semantic"] = sem
        return create(cfg)
    with patch.object(run_eval,"Env",augmented):
        return scene(doc)


def trial(env,item,data,ep,step,actions,threshold,remaining_ids):
    from evaluation.main.evaluator import _env_action, _near_target_ids
    frame = replay_until(env,item,data,ep["paper"]["steps"],step)
    def measure(frame):
        pos = tuple(env.sim.get_agent_state().position)
        near = set(map(int,_near_target_ids(pos,item.target_object_ids,item.target_positions,
                   threshold=threshold,pathfinder=env.sim.pathfinder,target_view_points=item.view_points)))
        pixels = np.bincount(np.asarray(frame["semantic"]).ravel().clip(min=0).astype(int))
        visible = {i for i in remaining_ids if i < len(pixels) and pixels[i]>=103}
        return near & remaining_ids,visible
    initial_near,initial_seen = measure(frame)
    near,seen = set(initial_near),set(initial_seen)
    collision = False
    for action in actions:
        frame = env.step(_env_action(action))
        n,v = measure(frame)
        near |= n
        seen |= v
        if action==1 and env.sim.previous_step_collided:
            collision = True
            break
    return dict(new_in_range=bool(near-initial_near), new_in_view=bool(seen-initial_seen),
                in_range=bool(near), in_view=bool(seen), collision=bool(collision))


def run(project):
    deny_ip_sockets()
    certified = project / "jepa_variable_v16/checks"
    public = (certified/"public_summary.json").read_bytes()
    private, = [d.parent for d in certified.glob("private.*/public_summary.json") if d.read_bytes()==public]
    records = load(private/"junctions/receipt.json")["results"]
    pilot = project/"mapping_runtime/pilots/jepa_evidence_v15_first2"
    counts = {arm:Counter() for arm in ("native","v16","any_offered_with_scan")}
    comparisons = Counter()
    n = choices = 0
    for row in load(pilot/"panel.json")["rows"]:
        ref = load(pilot/"comparison/accepted/second"/(row["episode_id"]+".json"))
        path = Path(ref["result"])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==ref["result_sha256"]
        doc = load(path)
        ep, = doc["episodes"]
        threshold = doc["manifest"]["arguments"]["threshold"]
        assert threshold == .25
        env,item,data = semantic_scene(doc)
        try:
            for event in records:
                if event["ordinal"]!=row["ordinal"] or "selected" not in event:
                    continue
                step = event["step"]
                budget = min(24,ep["max_steps_budget"]-step)
                rank, = [r for r in lines(private/"junctions"/str(row["ordinal"])/str(step)/"v13_routes.jsonl") if r["event"]=="rank"]
                orig = rank["ids"].index(rank["original"])
                chosen = rank["ids"].index(rank["selected"])
                assert rank["selected"]==event["selected"]
                credited = {int(i) for rep in ep["report_events"] if rep["step"] < step for i in rep["new_targets"]}
                ids = set(map(int,item.target_object_ids)) - credited
                native_log, = [p for p in lines(path.parent/"agent/frontier_probe.jsonl") if p["event"]=="activation" and p["step"]==step and p["selected_id"]==rank["original"] and p["activated"]]
                native_plan, = [r["nominal_action_ids"] for r in native_log["candidates"] if r["id"]==rank["original"]]
                assert rank["plans_native"][orig]==native_plan[:len(rank["plans_native"][orig])]
                native = trial(env,item,data,ep,step,native_plan[:budget],threshold,ids)
                group = [trial(env,item,data,ep,step,([3,2,2,3]+p)[:budget],threshold,ids) for p in rank["plans_native"]]
                oracle = {k:any(r[k] for r in group) for k in group[0]}
                for arm,result in (("native",native),("v16",group[chosen]),("any_offered_with_scan",oracle)):
                    counts[arm].update(result)
                for metric in ("new_in_range","new_in_view"):
                    comparisons[metric+"_v16_only"] += group[chosen][metric] and not native[metric]
                    comparisons[metric+"_native_only"] += native[metric] and not group[chosen][metric]
                n += 1
                choices += len(group)
                print(row["ordinal"],step,native,group[chosen],flush=True)
        finally:
            env.close()
    assert n==9 and choices==28
    return dict(junctions=n,offered_routes=choices,arms=counts,paired=comparisons,
                unit="Number of archived junctions with a previously uncredited target meeting each condition; junctions share episodes.",
                protocol="Max 24 total actions or remaining official budget. V16 pays four scan actions. "
                         "Test nominal chosen route until budget, route end, or first simulated collision. "
                         "new_in_range and new_in_view exclude targets already in range/in view at the decision start. "
                         "Sight threshold 103 semantic pixels in one 320px frame. Reach uses official .25m viewpoint geodesic rule. "
                         "No VLM replanning, official report or extra trajectory after a branch ends; offline local opportunity only.")


if __name__=="__main__":
    output=Path(sys.argv[2])
    content=json.dumps(run(Path(sys.argv[1])),indent=2,allow_nan=False)+"\n"
    output.write_text(content)
    print(content)
