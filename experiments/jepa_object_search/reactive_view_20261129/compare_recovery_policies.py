"""Same physical stop and same candidate set: single next-decision ablation, never tuned on these cases."""
from collections import Counter
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/"new_view_20261129"
OUT=HERE/"private_recovery_comparison"
sys.path.insert(0,str(PRIOR))


def simulate(replay=False):
    from sample_views import simulator, camera, inventory, ASSETS, SOURCE_H, ACTION
    records=json.loads((PRIOR/"private_evaluation/blind_choices.json").read_text())
    counts={k:Counter() for k in ["jepa","projected_area","nearest"]}
    pairs={k:Counter() for k in ["projected_area","nearest"]}
    for house in sorted({r["house"] for r in records}):
        folder=PRIOR/"private_render_blind60"/("blind_"+house)
        with np.load(folder/"controls.npz") as f:
            d={k:f[k] for k in ("action","group","kind","budget","reach","start_pixels","scan_pixels")}
        oldprobes=json.loads((HERE/"private_stop_probes"/(house+".json")).read_text())
        eligible=sorted(x["source"] for x in oldprobes if not x["prior_ever_clear"] and x["remaining"]>=2)
        subset=eligible[:10]
        sim=simulator(house)
        try:
            inventory(sim,ASSETS)
            ag=sim.get_agent(0)
            for row in sorted((r for r in records if r["house"]==house and r["source"] in subset),key=lambda r:r["source"]):
                src=row["source"]
                ix=np.flatnonzero(d["group"]==row["group"])
                old=ix[d["kind"][ix]=="old"]
                assert len(old)==7 and not d["reach"][old].any() and max(d["start_pixels"][ix[0]],d["scan_pixels"][ix[0]])<369
                j=row["jepa"]
                plan=[] if j is None else d["action"][j][d["action"][j]>0]
                path=OUT/f"{house}_{src}.npz"
                with np.load(folder/f"{src}_sensor.npz") as s:
                    start,syaw,oid,box=s["position"],float(s["yaw"]),int(s["oid"]),s["box"]
                state=camera(sim,start,syaw,SOURCE_H,0.)
                pos=[];rgb=[];dep=[];bumps=[];peak=0;spent=0
                for a in (0,2,3,*plan):
                    bumped=False
                    if a:
                        before=ag.state.position.copy();hit=ag.act(ACTION[a]);spent+=1
                        bumped=a==1 and (hit or np.linalg.norm((ag.state.position-before)[[0,2]])<.23)
                        state=sim.get_sensor_observations()
                    peak=max(peak,int((state["ids"]==oid).sum()))
                    delta=ag.state.position-start
                    q=ag.state.rotation;angle=2*math.atan2(q.y,q.w)-syaw
                    pos.append([math.cos(syaw)*delta[0]-math.sin(syaw)*delta[2],
                                -math.sin(syaw)*delta[0]-math.cos(syaw)*delta[2],math.atan2(math.sin(angle),math.cos(angle))])
                    rgb.append(state["rgb"][...,:3].copy());dep.append(state["depth"].squeeze().copy());bumps.append(bumped)
                    if bumped:break
                assert peak<369
                remaining=int(d["budget"][ix[0]])-spent
                if not replay:
                    np.savez_compressed(path,source=src,house=house,pose=pos,rgb=rgb,depth=dep,bump=bumps,remaining=remaining,box=box)
                    continue
                with np.load(path) as earlier:assert np.array_equal(earlier["depth"][-1],dep[-1])
                action=json.loads(path.with_suffix(".json").read_text())
                stop=ag.get_state(); outcomes={}
                for arm,p in action.items():
                    assert len(p)<=remaining
                    ag.set_state(stop,reset_sensors=False,infer_sensor_states=False)
                    clear=False;end=0;hit=False;steps=0
                    for a in p:
                        before=ag.state.position.copy();b=ag.act(ACTION[a]);steps+=1
                        end=int((sim.get_sensor_observations()["ids"]==oid).sum())
                        clear|=end>=369
                        hit=a==1 and (b or np.linalg.norm((ag.state.position-before)[[0,2]])<.23)
                        if hit:break
                    outcomes[arm]=clear
                    counts[arm].update(groups=1,actions=steps,ever_clear=int(clear),terminal_clear_and_arrived=int(not hit and bool(p) and end>=369),
                                       contact=int(hit),issued=int(bool(p)))
                for arm in pairs:
                    pairs[arm].update(different=int(action["jepa"]!=action[arm]),jepa_gained=int(outcomes["jepa"] and not outcomes[arm]),
                                      control_gained=int(not outcomes["jepa"] and outcomes[arm]))
        finally:sim.close()
    if replay:
        report=dict(protocol="First ten eligible source indices per house from old-seven-blocked and initial/sweep-unclear requests whose old full path never saw 369 same-instance pixels, with >=2 actions left. Independent one-proposal intervention at the actual previous stop. Same gate, admitted candidates and remaining budget. No verifier or second replan in this local counterfactual. Report is separate from full rollouts.",
                    counts=counts,paired=pairs)
        (HERE/"recovery_policy_comparison.json").write_text(json.dumps(report,indent=2)+"\n")
        print(report)


def predict():
    import torch
    from reactive_route import RoutePlanner
    torch.set_num_threads(2)
    planner=RoutePlanner.load()
    house="";ctx=None
    files=sorted(OUT.glob("*.npz"))
    assert files
    for path in files:
        with np.load(path) as f:
            if house!=str(f["house"]):
                house=str(f["house"])
                with np.load(PRIOR/"private_data"/f"blind_{house}.npz") as z:
                    ctx={int(i):c for i,c in zip(z["source_index"],z["context"])}
            frames=[dict(x=x,z=z,yaw=y,rgb=rgb,depth=depth,bump=bool(b)) for (x,z,y),rgb,depth,b in
                    zip(f["pose"],f["rgb"],f["depth"],f["bump"])]
            result=planner.choose(ctx[int(f["source"])],f["box"],frames[0]["depth"],frames,int(f["remaining"]),comparison=True)
            action=result["actions"]
            choices=dict(jepa=action,projected_area=action,nearest=action)
            choices.update(result.get("control_actions",{}))
            path.with_suffix(".json").write_text(json.dumps(choices))
    print(len(files),"matched proposal sets",flush=True)


if __name__=="__main__":
    os.umask(0o077);OUT.mkdir(exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["collect","infer","replay"]);args=p.parse_args()
    if args.stage=="infer":
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):predict()
    else:simulate(args.stage=="replay")
