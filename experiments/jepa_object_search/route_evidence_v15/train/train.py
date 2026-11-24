"""Train actor-only evidence forecasts against hidden-instance HSSD route outcomes."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch.nn import functional as F

from evidence_head import EvidenceNet

CHECKPOINTS = (120, 240, 360)
SEED = 1511


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load(project, split, teachers):
    v11 = project/"jepa_hidden_v11"
    v12 = project/"jepa_route_v12"
    sys.path.insert(0,str(v11/"models"))
    from train_compare import route_inputs
    root = (v12 if split in ("extra_train", "fresh2") else v11)/"models/cache/hssd"/split
    keys, fields = [], defaultdict(list)
    for actor in sorted((root/"actor").glob("*.npz")):
        with np.load(actor) as z:
            ctx = z["ctx"].astype(np.float32)
            x = route_inputs(ctx, z["actions"])
            house, origin, words, text = str(z["house"]), str(z["origin"]), z["words"].tolist(), z["text"]
        with np.load(root/"truth"/actor.name) as z:
            y = z["y"].T
        if teachers:
            with np.load(v12/"data/local_targets"/split/(origin+".npz")) as z:
                assert z["words"].tolist() == words and np.array_equal(z["y"].T, y)
                first = z["first_tap"].T
                local = z["local_z"].astype(np.float32).swapaxes(0, 1)
                assert np.array_equal(first >= 0, y > 0)
            with np.load(root/"teacher"/actor.name) as z:
                future = z["future"].astype(np.float32)
        for q, word in enumerate(words):
            keys.append((house, origin, word))
            fields["x"].append(x)
            fields["text"].append(text[q])
            fields["y"].append(y[q])
            if teachers:
                # Teacher only: local observed region most aligned to the goal at
                # each tap. At exact first discovery use the real instance mask.
                region = (future @ text[q]).argmax(-1)
                z = np.take_along_axis(future,region[...,None,None],axis=2).squeeze(2)
                valid = np.where(first[q] >= 0)[0]
                z[valid, first[q,valid]] = local[q,valid]
                fields["future"].append(z)
                fields["first"].append(first[q])
    if not keys or len(set(keys)) != len(keys):
        raise ValueError(f"invalid split {split}")
    return keys, {k:np.stack(v) for k,v in fields.items()}


def macro(keys, hits):
    origins = defaultdict(list)
    for (house,origin,_), hit in zip(keys,hits):
        origins[house,origin].append(float(hit))
    perhouse = defaultdict(list)
    for (house,_), vals in origins.items():
        perhouse[house].append(np.mean(vals))
    return {h:float(np.mean(v)) for h,v in perhouse.items()}


@torch.inference_mode()
def evaluate(net, keys, data):
    net.eval()
    probs = torch.cat([net(x, t)[0] for x,t in zip(data["x"].split(96),data["text"].split(96))]).cpu().numpy()
    y = data["y"].cpu().numpy()
    choices = probs.argmax(-1)
    correct = y[np.arange(len(y)),choices] > 0
    scores = macro(keys,correct)
    return dict(queries=len(y),selected_new_evidence=int(correct.sum()),
                house_macro=float(np.mean(list(scores.values()))),
                chance_house_macro=float(np.mean(list(macro(keys,(y>0).mean(-1)).values()))),
                house_scores=scores,brier=float(np.mean((probs-(y>0))**2)))


def losses(net, batch, arm):
    x, text, y, first, future = (batch[k] for k in ("x","text","y","first","future"))
    p, hazard, pred = net(x, text)
    binary = (y>0).float()
    discovery = F.binary_cross_entropy(p.clamp(1e-6,1-1e-6), binary)
    # Only at-risk taps have a first-event hazard label.
    taps = torch.arange(4, device=x.device)
    at_risk = (first[...,None] < 0) | (taps <= first[...,None])
    hit = taps == first[...,None]
    timed = F.binary_cross_entropy_with_logits(hazard[at_risk],hit[at_risk].float())
    pref = binary[:,:,None] > binary[:,None,:]
    logits = torch.logit(p, eps=1e-6)
    pair = F.softplus(logits[:,None,:]-logits[:,:,None])[pref].mean()
    loss = discovery + .4*timed + .4*pair
    if arm == "jepa":
        departure = F.normalize(x[...,:pred.shape[-1]],dim=-1)
        # Predict the *change* from the corresponding departure observation.
        # Emphasize target-bearing first observations rather than just scene identity.
        teacher_change = future-departure
        changed = teacher_change.norm(dim=-1) > .08
        err = 1-F.cosine_similarity(pred-departure,teacher_change,dim=-1)
        w = (1+4*hit.float()) * changed.float()
        loss += .6*(w*err).sum()/w.sum().clamp_min(1)
        # Within the same house/fork, predictions must distinguish branches
        # with different actual future evidence despite identical departure RGB.
        pd = pred - pred.mean(1,keepdim=True)
        td = future - future.mean(1,keepdim=True)
        mask = td.norm(dim=-1) > .08
        ctr = 1-F.cosine_similarity(pd,td,dim=-1)
        loss += .25*(ctr*mask).sum()/mask.sum().clamp_min(1)
    return loss


def main(project):
    dest = project/"jepa_evidence_v15/train"
    if (dest/"selection.json").exists():
        raise FileExistsError("The model is already frozen.")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def tensorize(d):
        return {k:torch.as_tensor(v,device=device,dtype=torch.float32) for k,v in d.items()}
    keys, train = load(project, "train", True)
    extra, more = load(project, "extra_train", True)
    assert {k[0] for k in keys}.isdisjoint({k[0] for k in extra})
    keys += extra
    train = tensorize({k:np.concatenate([train[k],more[k]]) for k in train})
    ck, cal = load(project,"calibration",False)
    cal = tensorize(cal)
    train_houses = set(k[0] for k in keys)
    cal_houses = set(k[0] for k in ck)
    assert train_houses.isdisjoint(cal_houses)
    by_house = [[i for i,k in enumerate(keys) if k[0]==h] for h in sorted(train_houses)]
    dim = train["text"].shape[-1]
    chosen, nets, table = {}, {}, {}
    for arm in ("jepa","direct"):
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        net = EvidenceNet(dim).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=.00065, weight_decay=.02)
        best = (-1.,float("-inf"))
        table[arm] = []
        for step in range(1,CHECKPOINTS[-1]+1):
            net.train()
            ix = [rng.choice(by_house[h]) for h in rng.integers(len(by_house),size=64)]
            loss = losses(net,{k:v[ix] for k,v in train.items()},arm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            opt.step()
            if step in CHECKPOINTS:
                result = evaluate(net,ck,cal)
                row = {"step":step,**result}
                table[arm].append(row)
                score = (result["house_macro"],-result["brier"])
                if score>best:
                    best = score
                    torch.save(net.state_dict(),dest/f"{arm}.pt")
                    chosen[arm] = step
        net.load_state_dict(torch.load(dest/f"{arm}.pt",map_location=device,weights_only=True))
        nets[arm]=net
    frozen=dest/"frozen.pt"
    shutil.copyfile(dest/"jepa.pt",frozen)
    sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
    write(dest/"selection.json",dict(sha256=sha,dim=dim,steps=chosen,seed=SEED,
          selector="highest calibration within-house/origin macro hit, lower Brier tie-break",
          selected_before_holdout=True))
    manifest = dict(training_houses=len(train_houses),training_queries=len(keys),
                    training_origins=len(set((k[0],k[1]) for k in keys)),
                    positive_route_queries=int((train["y"]>0).sum()),negative_route_queries=int((train["y"]==0).sum()),
                    calibration_houses=len(cal_houses),calibration_queries=len(ck),calibration=table,
                    disjoint=True,steps=chosen,seed=SEED,checkpoint_sha256=sha,heldout={})
    for split in ("test","fresh2"):
        kk, dd = load(project,split,False)
        hh=set(k[0] for k in kk)
        assert hh.isdisjoint(train_houses|cal_houses)
        dd=tensorize(dd)
        results={a:evaluate(n,kk,dd) for a,n in nets.items()}
        pd=np.array([results["jepa"]["house_scores"][h]-results["direct"]["house_scores"][h] for h in sorted(hh)])
        boot=np.random.default_rng(111).choice(pd,(10000,len(pd)),replace=True).mean(-1)
        manifest["heldout"][split]=dict(houses=len(hh),queries=len(kk),results=results,
                        delta_house_macro=float(np.mean(pd)),delta_ci95=np.quantile(boot,[.025,.975]).tolist())
    write(dest/"training_results.json",manifest)
    print(json.dumps({s:[r["houses"],r["queries"],r["delta_house_macro"]] for s,r in manifest["heldout"].items()}),flush=True)


if __name__=="__main__":
    main(Path(sys.argv[1]))
