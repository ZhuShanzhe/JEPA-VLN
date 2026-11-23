"""Fit once on HSSD, freeze the chosen model, and separately evaluate untouched houses."""
import argparse
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F

from report_head import ReportOpportunity, forecast_inputs, load_v7, route_scores
from route_heads import load_heads, route_logits


SOURCE = Path(__file__).resolve().parent
HERE = SOURCE
DATA = AUX = FEATURES = None
SPLITS = ("train", "calibration", "holdout")
SEED = 37


def write_new(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(split, heads):
    fname = DATA / f"{split}.rows.json"
    records = json.loads(fname.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"invalid {split}")
    official = {r["query"] for r in records}
    if len(official)!=6:
        raise ValueError("expected six official ObjectNav categories")
    extra = json.loads((AUX / f"{split}.rows.json").read_text())
    if not isinstance(extra, list) or not extra:
        raise ValueError(f"invalid auxiliary {split}")
    if official & {r["query"] for r in extra}:
        raise ValueError("overlapping official and derived goal categories")
    official_states={(r["sid"],r["horizon"]) for r in records}
    if official_states!={(r["sid"],r["horizon"]) for r in extra}:
        raise ValueError("the two tasks must share all physical forks and budgets")
    records=records+extra
    with np.load(FEATURES/"words.npz") as f:
        words={q:torch.tensor(f[q].astype(np.float32)) for q in f}
    missing={r["query"] for r in records} - words.keys()
    if missing:
        raise KeyError(f"Missing frozen text embeddings: {sorted(missing)}")
    by_start = defaultdict(list)
    for row in records:
        assert not row["reported_ids_before"], "natural panel expected empty prior reports"
        by_start[row["sid"]].append(row)
    groups = []
    for sid, rows in sorted(by_start.items()):
        name = hashlib.sha256(sid.encode()).hexdigest()+".npz"
        cache=FEATURES/"views"/name
        with np.load(cache) as f:
            ctx = F.normalize(torch.tensor(f["ctx"].astype(np.float32)), dim=-1)
            ck_actions = torch.tensor(f["actions"].astype(np.int64))
            if split=="train":
                teacher = F.normalize(torch.tensor(f["teacher"].astype(np.float32)), dim=-1)
                sampled = f["samples"].tolist()
        per_h = defaultdict(list)
        for row in rows:
            per_h[int(row["horizon"])].append(row)
        for horizon, rr in sorted(per_h.items()):
            rr.sort(key=lambda r:r["query"])
            q = [r["query"] for r in rr]
            if len(q) != len(set(q)) or len(q) < 2:
                raise ValueError("expected distinct official queries for wrong-target control")
            a = torch.as_tensor(rr[0]["action"]).long()
            if not torch.equal(ck_actions[:, :horizon], a):
                raise ValueError(f"action/cache mismatch in {sid!r}")
            if any(r["action"] != rr[0]["action"] for r in rr):
                raise ValueError(f"incompatible same-start actions in {sid!r}")
            y = torch.tensor([r["gain_newly_eligible"] for r in rr], dtype=torch.float32).T.contiguous()
            strict = torch.tensor([r["gain"] for r in rr], dtype=torch.float32).T.contiguous()
            initial = torch.tensor([len(r["start_eligible_ids"]) for r in rr],dtype=torch.float32)
            diff = (a[:,None] != a[None,:]).any(-1)
            if ((y[:,None] != y[None,:]) & ~diff[...,None]).any():
                raise ValueError("identical planned actions have conflicting labels")
            modes = []
            # Full panorama is acquired for the training cache only. The sparse condition
            # uses exclusively recorded contiguous views ending at the starting heading.
            flip = hashlib.sha256(f"23|{sid}".encode()).digest()[0] & 1
            sign = 1 if flip else -1
            for bins in (list(range(12)), *[sorted((sign*i) % 12 for i in range(n)) for n in (1,2,3)]):
                mode = "full" if len(bins) == 12 else "sparse"
                yaw = torch.tensor(bins).float() * math.pi / 6
                modes.append(forecast_inputs(heads, mode, ctx[bins], yaw, a,
                                             float(rr[0]["step"]), float(rr[0]["turn"])))
            # Circularly permute all goal texts at this physical state, preserving all rows.
            wrong = {}
            domain = ["official" if x in official else "derived" for x in q]
            for label in ("official","derived"):
                order = [i for i,v in enumerate(domain) if v==label]
                if len(order)<2:
                    raise ValueError("each domain must supply distractor queries")
                random.Random(f"{SEED}|{sid}|{horizon}|{label}").shuffle(order)
                wrong.update(dict(zip(order,order[1:] + order[:1])))
            groups.append(dict(sid=sid,house=rr[0]["house"],horizon=horizon,mask=diff,
                               wrong=[wrong[i] for i in range(len(q))], query=q,domain=domain,
                               strict=strict,initial=initial,
                               text=torch.stack([words[k] for k in q]), y=y,
                               truth=(teacher[:,[sampled.index(horizon//2),sampled.index(horizon)]]
                                      if split=="train" else None),
                               modes=modes))
    return groups


def objective(model, group, mode, pos_weight):
    p, near, motion = group["modes"][mode]
    y = group["y"]
    logits, predicted = model(p, near, motion, group["text"])
    gap = (y[:,None] - y[None,:]).clamp_min(0) * group["mask"][:,:,None]
    score_diff = logits[:,None] - logits[None,:]
    ranking = (F.softplus(-score_diff) * gap).sum() / gap.sum().clamp_min(1)
    bce = F.binary_cross_entropy_with_logits(logits, (y>0).float(), pos_weight=pos_weight)
    error = predicted - group["truth"]
    relative = (error - error.mean(0,keepdim=True)).square().sum(-1).mean()
    cosine = (1 - (predicted*group["truth"]).sum(-1)).mean()
    return ranking + .35*bce + .30*cosine + .15*relative


def per_origin(scores, group, y):
    y = y.numpy()
    p = scores.detach().numpy()
    eligible = group["mask"].numpy()
    num, queries = p.shape
    # Alternate origins cycle uniformly. The actual MOST VLM is absent in these houses.
    ref = np.arange(num)[:,None]
    qi = np.arange(queries)[None,:]
    options = np.where(eligible[:,:,None], p[None,:,:], -np.inf)
    alt = options.argmax(1)
    switched = np.max(options,axis=1) - p > .01
    chosen = np.where(switched,alt,ref)
    picked = y[chosen,qi]
    advantage = (picked-y).mean()
    dy = y[:,None]-y[None,:]
    dp = p[:,None]-p[None,:]
    valid = (dy>0)&eligible[:,:,None]
    order = (np.sign(dp[valid])+1)/2
    return dict(instances=float(picked.mean()), gain=float(advantage),
                origin_instances=float(y.mean()), override=float(switched.mean()),
                pair_credit=float(order.sum()), strict_pairs=int(valid.sum()),
                no_opportunity=float((y.max(0)==0).mean()),
                no_distinction=float((y.max(0)==y.min(0)).mean()))


def aggregate(raw):
    per = {}
    for arm, houses in raw.items():
        home = {}
        for hid, starts in sorted(houses.items()):
            by_start = []
            for sid, vals in starts.items():
                # Each horizon/query/mask shares a physical start with all of its replicas.
                by_start.append({k:float(np.mean([x[k] for x in vals])) for k in vals[0]})
            mean = {k:float(np.mean([x[k] for x in by_start])) for k in by_start[0]}
            mean["pair_accuracy"] = mean["pair_credit"]/mean["strict_pairs"] if mean["strict_pairs"] else None
            mean["informative_starts"] = sum(x["strict_pairs"]>0 for x in by_start)
            mean["starts"] = len(by_start)
            home[hid] = mean
        metrics = {k:float(np.mean([v[k] for v in home.values()])) for k in
                   ("instances","gain","origin_instances","override","no_opportunity","no_distinction",
                    "already_eligible_at_start_rate")}
        valid = [v["pair_accuracy"] for v in home.values() if v["pair_accuracy"] is not None]
        metrics["house_mean_pair_accuracy_where_informative"] = float(np.mean(valid)) if valid else None
        per[arm] = dict(mean=metrics, houses=home)
    return per


@torch.inference_mode()
def evaluate(model, groups, alpha, only_primary=False, domain="official", outcome="y"):
    raw = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for g in groups:
        for forecast, nearest, motion in g["modes"]:
            text = g["text"]
            primary = route_scores(model,forecast,nearest,motion,text,alpha)
            arms = {"v7":primary}
            if not only_primary:
                displacement = torch.linalg.vector_norm(motion[:,-1,:2],dim=1)
                route_length = g["horizon"] * .25 / 6
                arms.update(v6=route_logits(forecast,text).sigmoid(),
                            observed=route_scores(model,nearest,nearest,motion,text,alpha,observed_only=True),
                            clip_observed=route_logits(nearest,text).sigmoid(),
                            planned_displacement=(displacement/route_length).clamp(0,1)[:,None].expand_as(primary),
                            wrong=primary[:,g["wrong"]], retain=torch.zeros_like(primary))
            keep = [i for i,v in enumerate(g["domain"]) if v==domain]
            for k, value in arms.items():
                result=per_origin(value[:,keep],g,g[outcome][:,keep])
                result["already_eligible_at_start_rate"]=float((g["initial"][keep]>0).float().mean())
                raw[k][g["house"]][g["sid"]].append(result)
    out=aggregate(raw)
    if not only_primary:
        for other in ("v6","observed","clip_observed","planned_displacement","wrong","retain"):
            paired=[]
            for house,starts in raw["v7"].items():
                for sid,vals in starts.items():
                    refs=raw[other][house][sid]
                    paired.append(float(np.mean([a["instances"]-b["instances"] for a,b in zip(vals,refs)])))
            out["v7"].setdefault("paired_by_physical_start",{})[other]=dict(
                win=sum(x>1e-8 for x in paired),tie=sum(abs(x)<=1e-8 for x in paired),
                loss=sum(x < -1e-8 for x in paired))
    return out


@torch.inference_mode()
def categories(model, groups, alpha):
    # Same checkpoint and coverage, with a denominator for each actually curated category.
    raw=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for g in groups:
        for pred,near,motion in g["modes"]:
            v7=route_scores(model,pred,near,motion,g["text"],alpha)
            v6=route_logits(pred,g["text"]).sigmoid()
            for i,query in enumerate(g["query"]):
                for name,s in (("v7",v7),("v6",v6)):
                    m=per_origin(s[:,i:i+1],g,g["y"][:,i:i+1])
                    m["already_eligible_at_start_rate"]=float(g["initial"][i]>0)
                    raw[name+"|"+query][g["house"]][g["sid"]].append(m)
    out=aggregate(raw)
    return {k:dict(**v["mean"],houses=len(v["houses"]),
                   starts=sum(x["starts"] for x in v["houses"].values()),
                   informative_starts=sum(x["informative_starts"] for x in v["houses"].values()))
            for k,v in out.items()}


def fit(heads):
    import copy
    train, calib = [prepare(s, heads) for s in SPLITS[:2]]
    houses = lambda groups: sorted({g["house"] for g in groups})
    if set(houses(train)) & set(houses(calib)):
        raise ValueError("overlapping fit/calibration houses")
    yes = sum(int((g["y"]>0).sum()) for g in train)
    total = sum(g["y"].numel() for g in train)
    pos_weight = torch.tensor(min(15., (total-yes)/max(yes,1)))
    active = [g for g in train if (g["y"].amax(0) != g["y"].amin(0)).any()]
    if not active:
        raise ValueError("no training states distinguish reportable branches")
    table = dict(train_houses=houses(train),calibration_houses=houses(calib),groups=len(train),
                 total_branch_query_labels=total,positive_branch_query_labels=yes,active_groups=len(active))
    print("fit support", table, flush=True)
    rng = random.Random(SEED)
    model = ReportOpportunity(heads[0].center,heads[0].basis)
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    best = (-float("inf"),None,None,None)
    history = []
    for epoch in range(1,13):
        model.train()
        rng.shuffle(train)
        for j in range(0,len(train),12):
            batch = train[j:j+12]
            balanced = rng.choices(active,k=len(batch))
            loss = torch.stack([objective(model,g,rng.randrange(4),pos_weight)
                                for g in batch+balanced]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),3,error_if_nonfinite=True)
            opt.step()
        if epoch%3==0:
            model.eval()
            for alpha in (.5,1.):
                val = evaluate(model,calib,alpha,only_primary=True)["v7"]["mean"]["gain"]
                history.append(dict(epoch=epoch,alpha=alpha,gain=val))
                if (val,-alpha,-epoch) > (best[0],-(best[1] or 0),-(best[2] or 0)):
                    best=(val,alpha,epoch,copy.deepcopy(model.state_dict()))
            print("calibration",history[-2:],flush=True)
    checkpoint = HERE/"v7.pt"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    torch.save(dict(head=best[3],alpha=best[1]), checkpoint)
    model.load_state_dict(best[3])
    model.eval()
    write_new(HERE/"selection.json",dict(data=table,seed=SEED,epoch=best[2],alpha=best[1],
              history=history,calibration=evaluate(model,calib,best[1]), sha256=sha(checkpoint),
              gate_sha256=sha(SOURCE/"v7_protocol.json"),sources={p:sha(SOURCE/p) for p in
              ("report_head.py","train_report_head.py","route_heads.py")},
              label_sha256={f"{p.name}/{s}":sha(p/f"{s}.rows.json") for s in SPLITS[:2] for p in (DATA,AUX)}))


def holdout(heads):
    rec = json.loads((HERE/"selection.json").read_text())
    if sha(HERE/"v7.pt")!=rec["sha256"] or sha(SOURCE/"v7_protocol.json")!=rec["gate_sha256"]:
        raise RuntimeError("model or protocol changed")
    model, alpha = load_v7(HERE/"v7.pt")
    groups = prepare("holdout",heads)
    hids = sorted({g["house"] for g in groups})
    if len(hids)!=4 or set(hids)&set(rec["data"]["train_houses"]+rec["data"]["calibration_houses"]):
        raise ValueError("unexpected final house split")
    scores=evaluate(model,groups,alpha)
    gain=lambda k:scores[k]["mean"]["gain"]
    by=scores["v7"]["houses"]
    gates=dict(opportunities=sum(v["informative_starts"]>0 for v in by.values())>=2,
               positive=gain("v7")>0,better_v6=gain("v7")>gain("v6")+1e-12,
               better_observed=gain("v7")>gain("observed")+1e-12,
               better_wrong=gain("v7")>gain("wrong")+1e-12,
               generality=sum(v["gain"]>scores["v6"]["houses"][h]["gain"]+1e-12
                              for h,v in by.items())>=2)
    write_new(HERE/"result.json",dict(pass_gate=all(gates.values()),checks=gates,score=scores,
              derived=evaluate(model,groups,alpha,domain="derived"),per_available_class=categories(model,groups,alpha),
              official_strict_unseen=evaluate(model,groups,alpha,outcome="strict"),
              derived_strict_unseen=evaluate(model,groups,alpha,domain="derived",outcome="strict"),
              selection_sha256=sha(HERE/"selection.json"),holdout_labels_sha256=sha(DATA/"holdout.rows.json"),
              derived_holdout_sha256=sha(AUX/"holdout.rows.json"),
              frozen_checkpoint=rec["sha256"],groups=len(groups),physical_starts=len({g["sid"] for g in groups}),
              available_query_horizon_rows=sum(len(g["query"]) for g in groups)))
    print("holdout",gates,{k:gain(k) for k in scores},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["fit","holdout"])
    parser.add_argument("--full-checkpoint",type=Path,required=True)
    parser.add_argument("--sparse-checkpoint",type=Path,required=True)
    parser.add_argument("--feature-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--official",type=Path,required=True)
    parser.add_argument("--derived",type=Path,required=True)
    args=parser.parse_args()
    HERE=args.output.resolve()
    HERE.mkdir(parents=True,exist_ok=True)
    FEATURES=args.feature_dir.resolve()
    DATA=args.official.resolve()
    AUX=args.derived.resolve()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    frozen=load_heads(args.full_checkpoint,args.sparse_checkpoint)
    if args.stage=="fit":
        fit(frozen)
    else:
        holdout(frozen)
