"""Paired real-data engineering run, fixed holdout, no validation-based checkpoint selection."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import platform
import random
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

if __package__:
    from .policy import FrontierRanker, load_ranker
else:
    from policy import FrontierRanker, load_ranker


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent / "a_exploration" / "paired_branches_v2"


@dataclass
class State:
    sid: str
    scene: str
    goal: str
    context: torch.Tensor
    xy: torch.Tensor
    actions: torch.Tensor
    future: torch.Tensor
    label: torch.Tensor


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_states(root):
    split = {"train": [], "dev": []}
    audit = read_json(root / "audit.json")
    assert audit["complete"] and audit["cohort_sha256"] == sha(root / "cohort.json")
    seen = set()
    for line in (root / "index.jsonl").read_text().splitlines():
        idx = json.loads(line)
        sid, group, scene, goal = (idx[k] for k in ("id", "split", "scene", "target"))
        assert sid not in seen
        seen.add(sid)
        path = root / sid
        manifest = read_json(path / "manifest.json")
        assert group == manifest["split"] and scene == manifest["source_scene"]
        assert Path(sid).parts[:2] == (group, scene)
        # ONLY these context.npz fields enter policy; no frozen poses/GT/executed actions.
        with np.load(path / "context.npz") as inputs:
            xy = torch.from_numpy(inputs["frontiers_xy"]).float()
            actions = torch.from_numpy(inputs["planned_actions"]).long()
        with np.load(path / "features_vjepa21_b384.npz") as cache:
            context = torch.from_numpy(cache["context_embedding"]).float()
            future = torch.from_numpy(cache["future_targets"]).float()
        teacher = read_json(path / "teacher" / "labels.json")
        assert teacher["synset"] == goal and teacher["visible_pixel_threshold"] == 64
        rows = teacher["branch_labels"]
        assert [row["index"] for row in rows] == list(range(len(xy)))
        label = torch.tensor([bool(row["newly_visible_ids"]) for row in rows]).float()
        assert future.shape == (len(xy), 768) and actions.shape == (len(xy), 24)
        assert int(label.sum()) == idx["positive"] and torch.isfinite(future).all()
        split[group].append(State(sid, scene, goal, context, xy, actions, future, label))
    train_houses = {s.scene for s in split["train"]}
    dev_houses = {s.scene for s in split["dev"]}
    assert train_houses.isdisjoint(dev_houses)
    for name, states in split.items():
        counts = audit["counts"][name]
        assert counts["states"] == len(states)
        assert counts["branches"] == sum(len(s.label) for s in states)
        assert counts["positive_branches"] == sum(int(s.label.sum()) for s in states)
    return split


def fit_projection(model, train):
    # Fit on one *past* feature per training state, shared unchanged across both arms.
    contexts = torch.stack([g.context for g in train])
    center = contexts.mean(0)
    _, singular, vh = torch.linalg.svd(contexts - center, full_matrices=False)
    k = model.basis.shape[1]
    model.center.copy_(center)
    model.basis.copy_(vh[:k].T)
    model.scales.copy_((singular[:k] / math.sqrt(len(train)-1)).clamp(min=1e-3))


def rankloss(logits, label):
    positive = logits[label == 1]
    negative = logits[label == 0]
    if len(positive) and len(negative):
        return F.softplus(negative[None, :] - positive[:, None]).mean()
    return logits.sum() * 0


def train_model(model, states, mode, epochs, seed):
    generator = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    report = []
    for epoch in range(epochs):
        order = list(range(len(states)))
        generator.shuffle(order)
        totals = []
        for i in range(0, len(order), 8):
            losses = []
            for idx in order[i:i+8]:
                s = states[idx]
                goal_id = torch.tensor(model.goal_names.index(s.goal))
                logits, predicted = model(s.context, s.xy, s.actions, goal_id)
                loss = F.binary_cross_entropy_with_logits(logits, s.label) + rankloss(logits, s.label)
                if mode == "predictive":
                    loss = loss + F.smooth_l1_loss(predicted, model.project(s.future))
                losses.append(loss)
            loss = torch.stack(losses).mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            totals.append([float(loss.detach()), float(norm)])
        report.append({"epoch":epoch+1, "mean_objective":float(np.mean(totals,axis=0)[0]),
                       "max_preclip_grad_norm":max(t[1] for t in totals)})
    return report


@torch.inference_mode()
def evaluate(model, states):
    selected = positives = oracle = states_with_contrast = correct_pairs = pairs = 0
    error = brier = 0.
    choices = []
    for s in states:
        goal_id = torch.tensor(model.goal_names.index(s.goal))
        logits, predicted = model(s.context, s.xy, s.actions, goal_id)
        scores = logits.sigmoid()
        choice = int(scores.argmax())
        y = s.label
        if (y == 1).any() and (y == 0).any():
            states_with_contrast += 1
            delta = scores[y == 1][:, None] - scores[y == 0][None, :]
            correct_pairs += float((delta > 0).sum() + 0.5 * (delta == 0).sum())
            pairs += delta.numel()
        selected += int(y[choice]); oracle += int(y.max()); positives += int(y.sum())
        error += float(F.smooth_l1_loss(predicted, model.project(s.future)).item())
        brier += float((scores - y).square().mean())
        choices.append({"state":s.sid, "selected_index":choice, "selected_proxy":int(y[choice]),
                        "best_available_proxy":int(y.max())})
    metrics = dict(states=len(states), houses=len({s.scene for s in states}),
                   branches=sum(len(s.xy) for s in states), positive_branches=positives,
                   chosen_positive=selected, oracle_positive_states=oracle,
                   state_mean_regret=(oracle - selected)/len(states),
                   informative_states=states_with_contrast, comparable_pairs=pairs,
                   pair_accuracy=(correct_pairs/pairs if pairs else None),
                   state_mean_projected_huber=error/len(states), state_mean_brier=brier/len(states))
    return metrics, choices


def acceptance(rank, held):
    s = next(s for s in held if len(s.xy) >= 2)
    scores = rank.score(s.context, s.xy, s.actions, s.goal)
    rev = rank.score(s.context, s.xy.flip([0]), s.actions.flip([0]), s.goal)
    assert np.allclose(scores, rev[::-1], atol=1e-6)
    opposite = s.actions.clone()
    opposite[s.actions == 1] = 2
    opposite[s.actions == 2] = 1
    # Hold observation/XY fixed to isolate action dependence; these reflected commands
    # are a numerical sensitivity check, not physically verified evaluation branches.
    diff = max(abs(x-y) for x,y in zip(scores,rank.score(s.context,s.xy,opposite,s.goal)))
    assert diff > 1e-8
    bad = s.actions.clone(); bad[0,0] = 3
    try:
        rank.score(s.context,s.xy,bad,s.goal)
    except ValueError:
        pass
    else:
        raise ValueError("Accepted illegal action code")
    return {"candidate_permutation_equivariant":True,"illegal_action_rejected":True,
            "mirrored_plan_max_score_delta":diff,"selected_state":s.sid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--output",type=Path,required=True,help="New results directory")
    parser.add_argument("--epochs",type=int,default=5)
    parser.add_argument("--seed",type=int,default=17)
    args = parser.parse_args()
    if args.epochs <= 0 or args.output.exists():
        parser.error("epochs must be positive and output must be a fresh path")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    split = read_states(args.data)
    train, dev = split["train"], split["dev"]
    goals = sorted({s.goal for s in train})
    assert set(goals).issuperset(s.goal for s in dev)
    receipt = {"dataset":"HSSD v2 short-window proxy, offline engineering validation",
               "label":"unseen in start turn sequence at 64 pixels, visible on branch at 64 pixels",
               "feature_target":"4-frame pooled V-JEPA2.1, rank-32 train-context PCA",
               "python":platform.python_version(),"torch":str(torch.__version__),"device":"cpu",
               "epochs":args.epochs,"seed":args.seed,"basis_dimension":32,
               "train_dev_houses_disjoint":True,"parameters_optimized_each_arm":None,
               "data_hashes":{f:sha(args.data / f) for f in ("cohort.json","index.jsonl","feature_protocol.json")},
               "implementation_hashes":{f:sha(HERE / f) for f in ("policy.py","run_pilot.py")},
               "results":{}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as folder:
        stage = Path(folder) / "run"
        stage.mkdir()
        checksums = {}
        for mode in ("direct","predictive"):
            torch.manual_seed(args.seed)
            model = FrontierRanker(goals)
            fit_projection(model, train)
            numparams = sum(p.numel() for p in model.parameters())
            receipt["parameters_optimized_each_arm"] = numparams
            history = train_model(model,train,mode,args.epochs,args.seed)
            model.eval()
            filename = f"{mode}.pt"
            torch.save({"format":"hssd-visible-v2/v1","goal_names":goals,"basis_dim":32,
                        "mode":mode,"state":model.state_dict()},stage / filename)
            checksums[filename] = sha(stage / filename)
            restored = load_ranker(stage / filename)
            metrics, choices = evaluate(restored,dev)
            receipt["results"][mode] = {"training":history, "development":metrics,
                                        "checks":acceptance(restored,dev)}
            (stage / f"{mode}_choices.jsonl").write_text("".join(json.dumps(x)+"\n" for x in choices))
        receipt["checkpoint_sha256"] = checksums
        (stage / "receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        stage.rename(args.output)
    print(json.dumps({key:value["development"] for key,value in receipt["results"].items()}))


if __name__ == "__main__":
    main()
