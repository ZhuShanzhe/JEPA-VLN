"""Learn discounted first discovery over actual HSSD prefixes, without route padding."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F


SEED = 642217
CHECKS = (60, 120, 180, 240)
DISCOUNT = .82  # one interval = six actions


def distribution(logits, lens):
    n = logits.shape[-1]
    indices = torch.arange(n, device=logits.device)
    survival = F.logsigmoid(-logits)
    prior = F.pad(survival.cumsum(-1)[...,:-1], (1,0))
    mass = prior.exp() * logits.sigmoid() * (indices < lens[...,None])
    return mass.sum(-1), (mass * (DISCOUNT**indices)).sum(-1)


def labels(first,lens):
    hit = (first >= 0) & (first < lens)
    return hit.float(), hit.float() * DISCOUNT ** first.clamp_min(0)


def batch_loss(net,data,arm,ix,rng):
    x, text, first, future = (data[k][ix] for k in ("x","text","first","future"))
    _,logit,pred = net(x,text)
    # Lengths are sampled before using any route outcome. Unselected routes
    # and observations after the actual prefix have zero weight in all losses.
    lens = torch.tensor(rng.integers(1,5,size=first.shape),device=x.device)
    keep = np.ones(first.shape,dtype=bool)
    ko = rng.integers(0,3,len(ix))
    removed = rng.random(len(ix)) < .5
    keep[np.where(removed)[0],ko[removed]] = False
    active = torch.as_tensor(keep,device=x.device)
    ticks = torch.arange(4,device=x.device)
    good = active[...,None] & (ticks < lens[...,None])
    at_risk = good & ((first[...,None] < 0) | (ticks <= first[...,None]))
    event = ticks == first[...,None]
    chance,value = distribution(logit,lens)
    found,reward = labels(first,lens)
    pref = active[:,:,None] & active[:,None,:] & (reward[:,:,None] > reward[:,None,:])
    target_loss = F.binary_cross_entropy(chance[active].clamp(1e-6,1.-1e-6),found[active])
    time_loss = F.binary_cross_entropy_with_logits(logit[at_risk], event[at_risk].float())
    margin = (value[:,:,None]-value[:,None,:])[pref]
    contrast = F.softplus(-4*margin).mean() if margin.numel() else value.sum()*0
    loss = target_loss + .5*time_loss + .8*contrast + .5*F.mse_loss(value[active],reward[active])
    if arm == "jepa":
        depart = F.normalize(x[...,:pred.shape[-1]],dim=-1)
        delta = future - depart
        reliable = (delta.norm(dim=-1) > .08) & good
        importance = (1+4*event.float()) * reliable
        err = 1-F.cosine_similarity(pred-depart,delta,dim=-1)
        loss += .6*(importance*err).sum()/importance.sum().clamp_min(1)
        mask = good[...,None].float()
        count = mask.sum(1,keepdim=True).clamp_min(1)
        centered_p = pred - (pred*mask).sum(1,keepdim=True)/count
        centered_f = future - (future*mask).sum(1,keepdim=True)/count
        valid = (centered_f.norm(dim=-1) > .08) & good
        loss += .25*((1-F.cosine_similarity(centered_p,centered_f,dim=-1))*valid).sum()/valid.sum().clamp_min(1)
    return loss


def frozen_options(keys,n):
    lens, ids = [], []
    for house,origin,_ in keys:
        # Same origin has the same physical candidates for every target class.
        d = hashlib.sha256(("prefix-v1|" + house + "|" + origin).encode()).digest()
        order = np.argsort(np.frombuffer(d[:3],dtype=np.uint8),kind="stable")
        ids.append(order[:n])
        lens.append(1 + np.frombuffer(d[3:6],dtype=np.uint8).astype(int)%4)
    return torch.as_tensor(np.asarray(lens)), torch.as_tensor(np.asarray(ids))


def aggregate(houses,chosen_reward,chosen_hit):
    by_house = {h:float(np.mean(chosen_reward[houses==h])) for h in sorted(set(houses))}
    return dict(house_macro_discounted_utility=float(np.mean(list(by_house.values()))),
                mean_discounted_utility=float(chosen_reward.mean()), selected_discovery=int(chosen_hit.sum()),
                by_house=by_house)


@torch.inference_mode()
def evaluate(net,keys,d,origin_dense=None,legacy_cal=None):
    net.eval()
    logits = torch.cat([net(x,t)[1] for x,t in zip(d["x"].split(128),d["text"].split(128))]).cpu()
    first = d["first"].cpu()
    houses=np.array([k[0] for k in keys])
    report={}
    for n in (2,3):
        lens, ids = frozen_options(keys,n)
        chance, val = distribution(logits,lens)
        if legacy_cal:
            z = logits*legacy_cal["scale"] + legacy_cal["offset"]
            mask = torch.arange(4)<lens[...,None]
            val = -torch.expm1((F.logsigmoid(-z)*mask).sum(-1)/lens)
        found, reward = labels(first,lens)
        best = ids.gather(1,val.gather(1,ids).argmax(1,keepdim=True))
        chosen_reward=reward.gather(1,best).flatten().numpy()
        chosen_hit=found.gather(1,best).flatten().numpy()
        hit = found.gather(1,ids)
        r=aggregate(houses,chosen_reward,chosen_hit)
        r.update(queries=len(keys), random_discovery_expected=float(hit.sum()/n),
                 oracle_discovery=int(hit.any(1).sum()), mean_chosen_steps=float(6*lens.gather(1,best).float().mean()),
                 brier=float(((chance-found).square().gather(1,ids)).mean()) if not legacy_cal else None)
        if origin_dense is not None:
            visible = torch.as_tensor(origin_dense)
            any_step = (visible & (torch.arange(24)<(lens[...,None]*6))).any(-1)
            r.update(selected_discovery_any_step=int(any_step.gather(1,best).sum()),
                     random_discovery_any_step_expected=float(any_step.gather(1,ids).sum()/n),
                     oracle_discovery_any_step=int(any_step.gather(1,ids).any(1).sum()))
        report[str(n)]=r
    return report


def tensors(data,device):
    return {k:torch.as_tensor(v,device=device,dtype=torch.long if k=="first" else torch.float32)
            for k,v in data.items() if k in ("x","first","text","future")}


def write_new(path,data):
    content=json.dumps(data,indent=2,allow_nan=False)+"\n"
    if path.exists() and path.read_text()!=content:
        raise FileExistsError(path)
    path.write_text(content)


def main(project,mode):
    out = project / "jepa_variable_v16/model_training"
    old = project / "jepa_evidence_v15/train"
    sys.path[:0] = [str(old),str(project/"jepa_hidden_v11/models")]
    from train import load
    from evidence_head import EvidenceNet
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    dev="cpu"
    init=old/"direct.pt"
    if mode=="train":
        if (out/"freeze.json").exists():
            raise FileExistsError("already frozen")
        ck,cd=load(project,"calibration",True)
        cal=tensors(cd,dev)
        tr,td=load(project,"train",True)
        tr2,d2=load(project,"extra_train",True)
        tr+=tr2
        train=tensors({k:np.concatenate((v,d2[k])) for k,v in td.items()},dev)
        del td,d2,cd
        train_houses=sorted(set(x[0] for x in tr)); cal_houses=sorted(set(x[0] for x in ck))
        assert not set(train_houses)&set(cal_houses)
        by_house=[[i for i,k in enumerate(tr) if k[0]==h] for h in train_houses]
        opt_record={}
        for arm in ("direct","jepa"):
            torch.manual_seed(SEED)
            rng=np.random.default_rng(SEED)
            net=EvidenceNet(train["text"].shape[-1]).to(dev)
            net.load_state_dict(torch.load(init,map_location=dev,weights_only=True))
            opt=torch.optim.AdamW(net.parameters(),lr=.00023,weight_decay=.02)
            rows=[]; best=(-float("inf"),-float("inf"))
            for step in range(1,CHECKS[-1]+1):
                ix=[rng.choice(by_house[h]) for h in rng.integers(len(by_house),size=64)]
                net.train()
                loss=batch_loss(net,train,arm,ix,rng)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
                if step not in CHECKS:
                    continue
                stats=evaluate(net,ck,cal)
                util=float(np.mean([v["house_macro_discounted_utility"] for v in stats.values()]))
                brier=float(np.mean([v["brier"] for v in stats.values()]))
                rows.append(dict(step=step,house_macro_utility=util,brier=brier))
                if (util,-brier)>best:
                    best=(util,-brier); best_step=step
                    torch.save(net.state_dict(),out/(arm+".pt"))
            opt_record[arm]=dict(selected=best_step,calibration=rows,
                sha256=hashlib.sha256((out/(arm+".pt")).read_bytes()).hexdigest())
        write_new(out/"freeze.json",dict(seed=SEED,step_discount=DISCOUNT,
            initial_shared_v15_direct_sha256=hashlib.sha256(init.read_bytes()).hexdigest(),
            training_queries=len(tr),training_origins=len(set((k[0],k[1]) for k in tr)),
            training_houses=train_houses,calibration_queries=len(ck),calibration_houses=cal_houses,
            selector="mean across K=2,3 of calibration house-macro utility; tie: lower Brier",
            choices=opt_record))
        print("frozen", {k:v["selected"] for k,v in opt_record.items()},flush=True)
        return
    assert mode=="eval"
    freeze=json.loads((out/"freeze.json").read_text())
    keys, features=load(project,"fresh2",False)
    fresh_houses=set(k[0] for k in keys)
    assert fresh_houses.isdisjoint(freeze["training_houses"]+freeze["calibration_houses"])
    dense=[]
    for house,origin,word in keys:
        with np.load(out/"private_fresh"/(origin+".npz")) as sensors:
            assert str(sensors["house"])==house
            dense.append(sensors["visible"][sensors["words"].tolist().index(word)])
    dense=np.stack(dense)
    visible = dense[:,:,5::6]
    assert np.array_equal(visible.any(-1),features["y"]>0)
    features["first"] = np.where(visible.any(-1),visible.argmax(-1),-1)
    data=tensors(features,dev)
    cal=json.loads((project/"jepa_variable_v16/model/calibration.json").read_text())
    scores={}
    for name,path,legacy in (("jepa",out/"jepa.pt",None),("direct",out/"direct.pt",None),
                             ("shared_before_finetune",init,None),
                             ("v15_jepa_same_utility",old/"frozen.pt",None),("production_evidence_rate",old/"frozen.pt",cal)):
        if name in freeze["choices"]:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==freeze["choices"][name]["sha256"]
        net=EvidenceNet(data["text"].shape[-1])
        net.load_state_dict(torch.load(path,map_location=dev,weights_only=True))
        scores[name]=evaluate(net,keys,data,dense,legacy)
    diff={}
    for n in ("2","3"):
        a,b=(scores[k][n]["by_house"] for k in ("jepa","direct"))
        delta=np.asarray([a[h]-b[h] for h in sorted(a)])
        boot=np.random.default_rng(188).choice(delta,(10000,len(delta)),replace=True).mean(1)
        diff[n]=dict(paired_house_macro_delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
    result=dict(split="fresh2",houses=len(fresh_houses),queries=len(keys),origins=len(set(k[1] for k in keys)),
                action_steps_physically_replayed=72*len(set(k[1] for k in keys)),results=scores,
                jepa_minus_direct=diff)
    write_new(out/"fresh2.json",result)
    print(json.dumps({k:[v[n]["selected_discovery_any_step"] for n in ("2","3")] for k,v in scores.items()}))
    print(diff)


if __name__=="__main__":
    main(Path(sys.argv[1]),sys.argv[2])
