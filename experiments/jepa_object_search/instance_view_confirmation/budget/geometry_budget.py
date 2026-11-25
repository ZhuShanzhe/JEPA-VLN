"""Construct a full-map witness for movement budget, never importing the tested policy."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import habitat_sim
import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def route(doc, quota):
    from verify_habitat import scene
    from evaluation.main.evaluator import (_reset_environment, _env_action, _near_target_ids,
                                           _update_claim_quotas, _classify_target_claims)
    from habitat_sim.errors import GreedyFollowerError
    env, item, data = scene(doc)
    ep, = doc['episodes']
    counts = Counter()
    credited = set()
    claims, claim_quotas = {}, {}
    timeline = []
    excluded = set()
    fixed_goal = None
    try:
        _reset_environment(env, item, data)
        sim = env.sim
        assert np.linalg.norm(np.asarray(sim.get_agent_state().position) - ep['paper']['steps'][0]['position_before']) < .002
        assert ep['max_steps_budget'] == 300 and doc['manifest']['arguments']['threshold'] == .25
        follower = sim.make_greedy_follower(0, .17, stop_key=0, forward_key=1, left_key=2, right_key=3)
        while sum(counts.values()) < 300 and len(credited) < quota:
            here = np.asarray(sim.get_agent_state().position)
            near = set(_near_target_ids(here, item.target_object_ids, item.target_positions, .25, sim.pathfinder, item.view_points))
            if near - credited:
                counts['report'] += 1
                _update_claim_quotas(claim_quotas,near)
                newly_credited, _, _ = _classify_target_claims(near,claims,claim_quotas)
                assert set(newly_credited)==near-credited
                credited.update(newly_credited)
                timeline.append(dict(action_count=sum(counts.values()), total=len(credited)))
                fixed_goal = None
                continue
            if fixed_goal is None:
                candidates = []
                for iid in set(item.target_object_ids) - credited:
                    for index, pt in enumerate(item.view_points.get(iid, [])):
                        if (iid, index) in excluded:
                            continue
                        path = habitat_sim.ShortestPath()
                        path.requested_start, path.requested_end = here, np.asarray(pt, dtype=np.float32)
                        if sim.pathfinder.find_path(path) and np.isfinite(path.geodesic_distance):
                            candidates.append((float(path.geodesic_distance), iid, index))
                if not candidates:
                    break
                _, iid, index = min(candidates)
                fixed_goal = (iid, index)
                follower.reset()
            iid, index = fixed_goal
            try:
                act = follower.next_action_along(np.asarray(item.view_points[iid][index], dtype=np.float32))
            except GreedyFollowerError:
                act = 0
            if act not in (1, 2, 3):
                excluded.add(fixed_goal)
                fixed_goal = None
                continue
            env.step(_env_action(act))
            counts[{1:'forward',2:'turn',3:'turn'}[act]] += 1
        if len(credited)>=quota and sum(counts.values())<300:
            counts['finish']+=1
        return dict(quota=quota, credited=len(credited), action_counts=dict(counts), total=sum(counts.values()), evidence=timeline)
    finally:
        env.close()


def main(project):
    from verify_rank import deny_ip_sockets
    deny_ip_sockets()
    pilot=project/'mapping_runtime/pilots/jepa_confirm_v18_first2'
    for row in read(pilot/'panel.json')['rows']:
        ref=read(Path(row['ref']))
        file=Path(ref['result'])
        assert hashlib.sha256(file.read_bytes()).hexdigest()==ref['result_sha256']==row['result_sha256']
        print('GEOMETRY',json.dumps(dict(ordinal=row['ordinal'],**route(read(file),row['quota']))),flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]))
