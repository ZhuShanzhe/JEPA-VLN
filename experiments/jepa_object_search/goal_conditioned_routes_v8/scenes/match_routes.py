"""Build same-start target contrasts and audit nominal versus executed actions."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import numpy as np


def load_labels(paths):
    groups = defaultdict(lambda: defaultdict(dict))
    state_paths = {}
    for path in paths:
        for r in json.loads(path.read_text(encoding="utf-8")):
            sid, h, q = r["sid"], int(r["horizon"]), r["query"]
            if q in groups[sid][h]:
                raise ValueError(f"Duplicate goal within one state and horizon: {sid}, {h}, {q}")
            groups[sid][h][q] = r["gain_newly_eligible"]
            if sid in state_paths and state_paths[sid] != r["raw"]:
                raise ValueError(f"Conflicting observation path for {sid}")
            state_paths[sid] = r["raw"]
    return groups, state_paths


def match(paths, data_root):
    groups, locations = load_labels(paths)
    matches, kinds, alignment = [], Counter(), Counter()
    for sid, by_h in groups.items():
        raw = data_root / locations[sid]
        with np.load(raw / "context.npz", allow_pickle=False) as z:
            planned = z["planned_actions"]
        positions, actual = [], []
        for k in range(len(planned)):
            with np.load(raw / f"branch{k:03d}.npz", allow_pickle=False) as z:
                positions.append(z["relative_pose_xyyaw"][:, :2])
                actual.append(z["actions"])
        pos, actual = np.stack(positions), np.stack(actual)
        if actual.shape != planned.shape or pos.shape[:2] != planned.shape:
            raise ValueError(f"Inconsistent trajectory length in {sid}")
        prefix = np.cumprod(planned == actual, axis=1).sum(axis=1)
        radial = np.linalg.norm(pos, axis=-1)
        delta = np.diff(pos, axis=1, prepend=np.zeros((len(pos), 1, 2)))
        travel = np.linalg.norm(delta, axis=-1).cumsum(axis=1)
        alignment["routes"] += len(pos)
        for h, gains in sorted(by_h.items()):
            if h < 1 or h > planned.shape[1] or any(len(x) != len(pos) for x in gains.values()):
                raise ValueError(f"Invalid branch count or horizon in {sid}")
            alignment[f"exact_at_{h}"] += int((prefix >= h).sum())
            forwards = (planned[:, :h] == 0).sum(axis=1)
            for i,j in combinations(range(len(pos)), 2):
                if np.array_equal(planned[i,:h], planned[j,:h]):
                    continue
                separation = np.linalg.norm(pos[i,:h]-pos[j,:h], axis=1).max()
                if separation < .249:
                    continue
                mismatch = abs(int(forwards[i])-int(forwards[j]))
                gap = max(abs(radial[i,h-1]-radial[j,h-1]), abs(travel[i,h-1]-travel[j,h-1]))
                if mismatch == 0 and gap <= .251:
                    tier = "strict"
                elif mismatch <= 1 and gap <= .501:
                    tier = "near"
                else:
                    continue
                values = {q:[v[i],v[j]] for q,v in gains.items()}
                left = any(a>b for a,b in values.values())
                right = any(a<b for a,b in values.values())
                kind = "reversal" if left and right else "single" if left or right else "equal"
                exact = bool(min(prefix[i],prefix[j]) >= h)
                kinds[f"{tier}/{kind}/{'exact' if exact else 'follower'}"] += 1
                matches.append({"sid":sid, "horizon":h, "branches":[i,j], "match":tier,
                    "kind":kind, "both_prefixes_exact":exact, "query_gains":values})
    return {"states":len(groups), "matched_pairs":len(matches), "alignment":dict(alignment),
            "pairs_by_condition":dict(kinds)}, matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-rows", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--private-pairs", type=Path)
    args = parser.parse_args()
    summary, pairs = match(args.query_rows, args.data_root)
    for dest, obj in ((args.summary,summary),(args.private_pairs,pairs)):
        if dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("x", encoding="utf-8") as f:
                json.dump(obj,f,ensure_ascii=False,indent=2)
