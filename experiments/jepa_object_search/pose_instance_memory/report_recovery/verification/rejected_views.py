"""Inspect at most three rejected reports per completed case, after the policy exits."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def main(project, dest):
    sys.path[:0] = [str(project/"jepa_exec_v14"), str(project/"jepa_completion_v17/physical")]
    from verify_habitat import deny_network, replay_until
    from check_routes import semantic_scene
    from evaluation.main.evaluator import _near_target_ids, _target_approach_distance
    sys.addaudithook(deny_network)
    pilot = project/"mapping_runtime/pilots/jepa_reid_v23_first2"
    result = []
    for row in read(pilot/"panel.json")["rows"]:
        receipt = pilot/"comparison/accepted/second"/(row["episode_id"]+".json")
        if not receipt.is_file():
            continue
        ticket = read(receipt)
        fp = Path(ticket["result"])
        assert hashlib.sha256(fp.read_bytes()).hexdigest() == ticket["result_sha256"]
        doc = read(fp)
        ep, = doc["episodes"]
        log = fp.parent/"agent/confirm_events.jsonl"
        if not log.exists():
            assert not ep["report_events"] and not ep["paper"]["agent_trace"]["observations"]
            continue
        events = [json.loads(r) for r in log.read_bytes().splitlines()]
        recovery = [r for r in events if r["event"]=="report_recovery_action"]
        if not recovery:
            continue
        # First and last refusal, plus first different reason when available.
        points = {r["step"]:r for r in (recovery[0],recovery[-1])}
        other = next((r for r in recovery if r["reason"]!=recovery[0]["reason"]),None)
        if other:
            points[other["step"]]=other
        env,item,data = semantic_scene(doc)
        try:
            for step,r in sorted(points.items()):
                frame = replay_until(env,item,data,ep["paper"]["steps"],step)
                pos = tuple(env.sim.get_agent_state().position)
                near = set(_near_target_ids(pos,item.target_object_ids,item.target_positions,
                    threshold=doc["manifest"]["arguments"]["threshold"],pathfinder=env.sim.pathfinder,
                    target_view_points=item.view_points))
                distances = [_target_approach_distance(env.sim.pathfinder,pos,obj,item.target_positions,item.view_points)
                             for obj in item.target_object_ids]
                credited = {v for e in ep["report_events"] if e["step"] < step for v in e["new_targets"]}
                s = np.asarray(frame["semantic"]).squeeze()
                assert s.shape == (320,320)
                visible = {int(obj) for obj in item.target_object_ids if (s==int(obj)).sum()>=103}
                would_score = set(map(int,near)) - set(map(int,credited))
                result.append(dict(task="clock" if row["ordinal"]==1 else "wall_art", step=step,
                    reason=r["reason"],recovery=r["recovery"], physical_action=ep["paper"]["steps"][step]["action"],
                    nearest_valid_view_m=round(min(d for d in distances if d is not None),3),
                    eligible_new_targets=len(would_score), visible_eligible_new_targets=len(would_score & visible)))
        finally:
            env.close()
    dest.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]))
