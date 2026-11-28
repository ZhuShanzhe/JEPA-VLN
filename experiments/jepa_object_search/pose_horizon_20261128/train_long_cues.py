"""Train matched local-camera heads for genuinely long base plans with a fixed observed instance."""
import copy
import json
import os
import numpy as np
import torch

from train_joint import (ROOT, SEED, JointPredictor, binary, cosine, file_hash,
                         load, loss, scored)


HORIZONS = (12, 24, 52, 80, 108)
EPOCHS = 30
GROUPS_PER_BATCH = 12


def read_data():
    d = load("long_cue")
    masks = {s: np.flatnonzero(d["split"] == s) for s in ("train", "val", "test")}
    houses = [set(d["house"][ids]) for ids in masks.values()]
    cohort = {s:len(list((ROOT/"private_long_cue").glob(s+"_*.npz"))) for s in masks}
    assert cohort == {"train":12,"val":6,"test":12} and list(map(len, houses)) == [11, 6, 12]
    assert all(not houses[i].intersection(houses[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    assert set(d["horizon"]) == set(HORIZONS)
    assert np.array_equal(d["future_valid"], d["reach"] == 1)
    assert (d["visible"][d["reach"] == 0] == -1).all()
    d["group"]=np.char.add(np.char.add(d["house"],"/"),d["group"])
    return d, masks


def train_batches(d, ids):
    groups = {}
    for i in ids:
        groups.setdefault((d["house"][i], d["group"][i]), []).append(i)
    assert all(len(v) == 9 for v in groups.values())
    return np.array(list(groups.values()), dtype=int)


def labels(d, ids):
    return ((d["reach"][ids] == 1) & (d["visible"][ids] == 1)).astype(np.float32)


def prediction_report(d, ids, values, has_latent):
    future, reach, cond, joint, _ = values
    result = {}
    panels = {"all": np.ones(len(ids), bool), "native": d["native_mask"][ids]}
    for steps in HORIZONS:
        mask = d["horizon"][ids] == steps
        panels[str(steps)] = mask
        panels[f"{steps}_native"] = mask & d["native_mask"][ids]
    for key, mask in panels.items():
        idx = ids[mask]
        y = labels(d, idx)
        reached = d["reach"][idx] == 1
        row = dict(joint=binary(y, joint[mask]), reach=binary(reached, reach[mask]),
                   visible_if_reached=binary(d["visible"][idx][reached], cond[mask][reached]),
                   subset_reached=int(reached.sum()))
        if has_latent and reached.any():
            targets = d["future"][idx][reached]
            row["future_cosine_on_reached"] = float(cosine(future[mask][reached], targets).mean())
            row["source_copy_cosine_on_reached"] = float(cosine(d["context"][idx, :1024][reached], targets).mean())
        result[key] = row
    return result


def do_arm(d, masks, initial, output, device):
    base = ROOT/"private_train"/(initial + ".pt")
    original_hash = file_hash(base)
    saved = torch.load(base, map_location="cpu", weights_only=True)
    model = JointPredictor(saved["dims"]).to(device)
    model.load_state_dict(saved["weights"])
    stats = saved["stats"]["pose"].numpy()
    groups = train_batches(d, masks["train"])
    train_count = len(masks["train"])
    class_y = labels(d, masks["val"])
    y_test = labels(d, masks["test"])
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    best = float("inf")
    best_weights = None
    curve = []
    # Include zero-epoch transfer as a calibration checkpoint; explicit transfer scores also reported.
    for epoch in range(EPOCHS + 1):
        if epoch:
            model.train()
            order = rng.permutation(len(groups))
            for start in range(0, len(groups), GROUPS_PER_BATCH):
                ids = groups[order[start:start+GROUPS_PER_BATCH]].reshape(-1)
                opt.zero_grad()
                objective = loss(model, "pose", d, ids, stats, initial, device)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                opt.step()
        model.eval()
        val = scored(model, "pose", d, masks["val"], stats, initial, device)
        score = float(np.mean((val[3]-class_y)**2))
        curve.append(score)
        if score < best:
            best, chosen = score, epoch
            best_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    destination = ROOT/"private_train"/(output + ".pt")
    torch.save(dict(weights={k:v.cpu() for k,v in best_weights.items()}, dims=saved["dims"],
                    variant=initial, stats=saved["stats"]), destination)
    rows = masks["test"]
    outputs = scored(model, "pose", d, rows, stats, initial, device)
    per_candidate = {k:d[k][rows] for k in ("house","group","source_file","source_row","native_mask","horizon","reach","visible")}
    np.savez_compressed(ROOT/"private_train"/("test_"+output+"_predictions.npz"), **per_candidate,
                        p_reach=outputs[1], p_visible_if_reach=outputs[2], p_joint=outputs[3])
    tuned = prediction_report(d, rows, outputs, initial != "direct_outcome")
    per_arm = {"epoch":chosen, "training_rows_per_epoch":train_count, "updates_per_epoch":int(np.ceil(len(groups)/GROUPS_PER_BATCH)),
               "val_joint_brier_curve":curve, "test":tuned, "checkpoint_sha256":file_hash(destination),
               "initial_checkpoint_sha256":original_hash}
    model.load_state_dict(saved["weights"])
    transfer = scored(model, "pose", d, rows, stats, initial, device)
    per_arm["untuned_same_inputs"] = prediction_report(d, rows, transfer, initial != "direct_outcome")
    assert original_hash == file_hash(base)
    p=outputs[3]
    print(output, chosen, "joint_brier", float(np.mean((p-y_test)**2)), "initial",float(np.mean((transfer[3]-y_test)**2)),flush=True)
    return per_arm, p, transfer[3]


def paired(y, a, b, house, rng):
    hs=np.unique(house)
    delta=(a-y)**2 - (b-y)**2
    d=np.array([delta[house==h].mean() for h in hs])
    boot=rng.integers(len(hs),size=(4000,len(hs)))
    return dict(houses=len(hs),houses_favouring_first=int((d<0).sum()),
                house_macro_brier_difference=float(d.mean()),
                house_bootstrap_95=np.quantile(d[boot].mean(1),[.025,.975]).tolist())


def main():
    os.umask(0o077)
    torch.set_num_threads(3)
    device="cuda" if torch.cuda.is_available() else "cpu"
    d,masks=read_data()
    trained={}
    outputs={}
    transfers={}
    for origin,variant in (("joint_long","long_cue_joint"),("direct_outcome","long_cue_direct")):
        trained[variant],outputs[variant],transfers[variant]=do_arm(d,masks,origin,variant,device)
    rows=masks["test"]
    house=d["house"][rows]
    y=labels(d,rows)
    rng=np.random.default_rng(SEED)
    summary={}
    for which,mask in (("all",np.ones(len(rows),bool)),("native",d["native_mask"][rows])):
        choices=dict(jepa_vs_direct=(outputs["long_cue_joint"],outputs["long_cue_direct"]),
                     jepa_vs_untuned=(outputs["long_cue_joint"],transfers["long_cue_joint"]),
                     direct_vs_untuned=(outputs["long_cue_direct"],transfers["long_cue_direct"]))
        summary[which]={k:paired(y[mask],a[mask],b[mask],house[mask],rng) for k,(a,b) in choices.items()}
    result=dict(seed=SEED,epochs=EPOCHS,selection="minimum pooled validation joint Brier; epoch zero eligible",
                lr=.0003,groups_per_batch=GROUPS_PER_BATCH,rows={s:len(i) for s,i in masks.items()},
                allocated_houses={"train":12,"val":6,"test":12},effective_houses={s:len(set(d["house"][i])) for s,i in masks.items()},
                condition="Same initially seen instance; three nominal base plans per start, reached requires no blocked move. Only fixed 1.31m/0 native; two rig probes comprise other rows.",
                trained=trained,paired=summary)
    (ROOT/"long_cue_trained_metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
