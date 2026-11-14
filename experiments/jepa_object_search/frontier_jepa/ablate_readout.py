"""Separate future auxiliary supervision from actually reading the predicted future."""
import argparse
import copy
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from policy import FrontierRanker
from run_pilot import DEFAULT_ROOT, acceptance, evaluate, fit_projection, rankloss, read_states, sha


class ReadoutAblation(FrontierRanker):
    def __init__(self, goals, route):
        nn.Module.__init__(self)
        if route not in ("current","future"):
            raise ValueError("Unknown readout route")
        self.route = route
        self.goal_names = tuple(goals)
        self.register_buffer("center",torch.zeros(768))
        self.register_buffer("basis",torch.zeros(768,32))
        self.register_buffer("scales",torch.ones(32))
        self.trunk = nn.Sequential(nn.Linear(106,96),nn.GELU())
        self.current_head = nn.Linear(96,32)
        self.future_head = copy.deepcopy(self.current_head)
        self.goal = nn.Embedding(len(goals),12)
        self.value = nn.Sequential(nn.Linear(44,32),nn.GELU(),nn.Linear(32,1))

    def forward(self, context, xy, actions, goal_id, teacher_needed=False):
        ctx = self.project(context).reshape(1,32).expand(len(xy),-1)
        cmd = F.one_hot(actions,num_classes=3).float().flatten(1)
        h = self.trunk(torch.cat([ctx,xy/5.,cmd],dim=1))
        future = None
        if self.route == "future" or teacher_needed:
            future = ctx + self.future_head(h)
        readout = future if self.route == "future" else ctx + self.current_head(h)
        target = self.goal(goal_id.reshape(1)).expand(len(xy),-1)
        logits = self.value(torch.cat([readout,target],dim=1)).squeeze(1)
        # During scoring/evaluate only the configured route is executed.
        return logits,future if teacher_needed else readout


def train(model, states, aux, seed, epochs):
    opt = torch.optim.AdamW(model.parameters(),lr=.001)
    order_rng = random.Random(seed)
    seen_grad = set()
    curve = []
    for _ in range(epochs):
        order = list(range(len(states)))
        order_rng.shuffle(order)
        means = []
        for offset in range(0,len(order),8):
            losses=[]
            for idx in order[offset:offset+8]:
                s=states[idx]
                q=torch.tensor(model.goal_names.index(s.goal))
                scores,z=model(s.context,s.xy,s.actions,q,teacher_needed=aux)
                loss=F.binary_cross_entropy_with_logits(scores,s.label)+rankloss(scores,s.label)
                if aux:
                    loss=loss+F.smooth_l1_loss(z,model.project(s.future))
                losses.append(loss)
            avg=torch.stack(losses).mean()
            assert torch.isfinite(avg)
            opt.zero_grad(set_to_none=True)
            avg.backward()
            seen_grad.update(name for name,p in model.named_parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            opt.step()
            means.append(float(avg.detach()))
        curve.append(float(np.mean(means)))
    ngrad=sum(p.numel() for name,p in model.named_parameters() if name in seen_grad)
    return curve,ngrad


@torch.inference_mode()
def isolate_paths(model,s):
    args=(s.context,s.xy,s.actions,s.goal)
    def measure(head):
        calls=[]
        hook=head.register_forward_hook(lambda *unused:calls.append(1))
        try:
            model.score(*args)
        finally:
            hook.remove()
        return len(calls)
    n_current=measure(model.current_head)
    n_future=measure(model.future_head)
    assert (n_current,n_future)==((1,0) if model.route=="current" else (0,1))
    trial=copy.deepcopy(model)
    before=np.array(trial.score(*args))
    trial.future_head.weight.zero_()
    trial.future_head.bias.zero_()
    after=np.array(trial.score(*args))
    delta=float(np.abs(before-after).max())
    assert delta==0 if model.route=="current" else delta>1.e-8
    return dict(current_calls=n_current,future_calls=n_future,score_delta_if_future_head_muted=delta)


def run(data, stage, seeds, epochs):
    split=read_states(data)
    train_set,dev=split["train"],split["dev"]
    goals=sorted({s.goal for s in train_set})
    receipt={"scope":"offline visibility-proxy engineering factorial; no closed-loop evaluation",
             "houses":{"train":len({s.scene for s in train_set}),"dev":len({s.scene for s in dev})},
             "readout":"identically initialized, equal-width parallel heads on a shared current+candidate trunk",
             "aux":"Smooth L1 on predicted future vs the projected true future, train states only",
             "inference_budget":"same shared trunk, exactly one 96->32 head, same goal/value head for every arm",
             "epochs":epochs,"input_hashes":{name:sha(data/name) for name in ("cohort.json","index.jsonl","feature_protocol.json")},
             "implementation_hash":sha(Path(__file__).resolve()),"results":{}}
    for seed in seeds:
        receipt["results"][str(seed)]={}
        noaux_scores={}
        for route,aux in (("current",False),("future",False),("current",True),("future",True)):
            name=f"{route}_{'aux' if aux else 'noaux'}"
            torch.manual_seed(seed)
            model=ReadoutAblation(goals,route)
            fit_projection(model,train_set)
            curve,ngrad=train(model,train_set,aux,seed,epochs)
            fn=stage/f"seed{seed}_{name}.pt"
            torch.save({"format":"hssd-readout-factorial/v1","state":model.state_dict(),
                        "goals":goals,"route":route,"aux":aux},fn)
            loaded=torch.load(fn,map_location="cpu",weights_only=True)
            restored=ReadoutAblation(loaded["goals"],loaded["route"])
            restored.load_state_dict(loaded["state"],strict=True)
            restored.eval()
            metrics,choices=evaluate(restored,dev)
            metrics.pop("state_mean_projected_huber") # current readout has no future interpretation
            checks=acceptance(restored,dev)
            checks.update(isolate_paths(restored,dev[0]))
            pred=[restored.score(s.context,s.xy,s.actions,s.goal) for s in dev]
            if not aux:
                noaux_scores[route]=pred
            rows=[dict(c,scores=p) for c,p in zip(choices,pred)]
            (stage/f"seed{seed}_{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            receipt["results"][str(seed)][name]=dict(online_route=route,future_loss=aux,
                params_total=sum(p.numel() for p in restored.parameters()),params_with_grad=ngrad,
                checkpoint_sha256=sha(fn),development=metrics,checks=checks,objective_by_epoch=curve)
        # The two routes have numerically identical predictions absent future supervision.
        assert all(np.array_equal(a,b) for a,b in zip(noaux_scores["current"],noaux_scores["future"]))
    receipt["control_noaux_routes_numerically_equivalent"]=True
    (stage/"receipt.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps({seed:{name:dict(chosen=item["development"]["chosen_positive"],
                                     called_future=item["checks"]["future_calls"]) for name,item in arms.items()}
                      for seed,arms in receipt["results"].items()}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--seeds",type=int,nargs="+",default=[17,18,19])
    args=p.parse_args()
    if args.output.exists() or args.epochs<=0 or len(set(args.seeds))!=len(args.seeds):
        p.error("Fresh output, positive epochs and unique seeds required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        stage=Path(tmp)/"run"
        stage.mkdir()
        run(args.data,stage,args.seeds,args.epochs)
        stage.rename(args.output)


if __name__=="__main__":
    main()
