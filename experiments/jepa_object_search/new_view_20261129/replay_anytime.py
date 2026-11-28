"""At every executed primitive, render target identity, including shared initial images and bump frames."""
from collections import Counter
import json
import numpy as np
from sample_views import HERE,simulator,camera,SOURCE_H,ACTION,ASSETS,inventory

THRESHOLD=369
CUTOFFS=(6,10,16,22,30)


def trace(sim,start,yaw,oid,action):
    obs=camera(sim,start,yaw,SOURCE_H,0.)
    ag=sim.get_agent(0)
    pixels=[int((obs["ids"]==oid).sum())]
    bump=0
    for step,act in enumerate((2,3)+tuple(action)):
        before=ag.state.position.copy()
        contact=ag.act(ACTION[act])
        pixels.append(int((sim.get_sensor_observations()["ids"]==oid).sum()))
        if act==1 and (contact or np.linalg.norm((ag.state.position-before)[[0,2]])<.23):
            bump=step-1  # 1-based after the two scan primitives
            break
    return pixels,bump


def summarize(records):
    groups=sorted({h for h,*_ in records})
    result={}
    for key in (*CUTOFFS,"full"):
        rows=Counter()
        samples={"old":[],"jepa":[]}
        per_house={}
        for house,budget,old,new in records:
            cap=budget if key=="full" else min(budget,key)
            source_clear=max(old[:3])>=THRESHOLD
            rows["groups"]+=1
            rows["start_and_sweep_unclear"]+=not source_clear
            had=[]
            for arm,px in (("old",old),("jepa",new)):
                first=next((i for i,v in enumerate(px[:cap+1]) if v>=THRESHOLD),None)
                clear=first is not None
                had.append(clear)
                rows[arm+"_seen"]+=clear
                rows[arm+"_new_after_sweep"]+=clear and not source_clear
                rows[arm+"_actions_to_cutoff_or_stop"]+=min(len(px)-1,cap)
                if clear:samples[arm].append(first)
            oldseen,newseen=had
            pair=per_house.setdefault(house,Counter())
            for scope in (rows,pair):
                scope["paired_gain"]+=newseen and not oldseen
                scope["paired_loss"]+=oldseen and not newseen
                scope["paired_gain_initially_unclear"]+=not source_clear and newseen and not oldseen
                scope["paired_loss_initially_unclear"]+=not source_clear and oldseen and not newseen
        out={"counts":rows,"by_house":[per_house[h] for h in groups]}
        for arm,cost in samples.items():
            later=[x for x in cost if x>2]
            def stats(v):
                return dict(n=len(v),mean=float(np.mean(v)) if v else None,median=float(np.median(v)) if v else None)
            out[arm+"_first_clear_actions_including_shared_views"]=stats(cost)
            out[arm+"_first_clear_actions_initially_unclear"]=stats(later)
        result[str(key)]=out
    return result


def main():
    choices=json.loads((HERE/"private_evaluation/blind_choices.json").read_text())
    rec=[];rejected=[]
    totals=Counter()
    for house in sorted({x["house"] for x in choices}):
        sim=simulator(house)
        try:
            inventory(sim,ASSETS)
            folder=HERE/"private_render_blind60"/("blind_"+house)
            with np.load(folder/"controls.npz") as z:
                d={k:z[k] for k in ("group","kind","action","reach","pixels","scan_pixels","start_pixels","first_blocked","budget")}
            for x in (x for x in choices if x["house"]==house):
                ix=np.flatnonzero(d["group"]==x["group"]);groups=d["kind"][ix]
                budget=int(d["budget"][ix[0]])
                with np.load(folder/f"{x['source']}_sensor.npz") as s:
                    start,yaw,oid=s["position"],float(s["yaw"]),int(s["oid"])
                cache={}
                for i in (x["old"],x["jepa"]):
                    if i in cache:continue
                    acts=() if i is None else d["action"][i][d["action"][i]>0].tolist()
                    px,bump=trace(sim,start,yaw,oid,acts)
                    assert len(px)-1<=budget and abs(px[0]-d["start_pixels"][ix[0]])<=1
                    assert abs(px[1]-d["scan_pixels"][ix[0]])<=1 and abs(px[2]-d["start_pixels"][ix[0]])<=1
                    if i is not None:
                        assert bump==d["first_blocked"][i]
                        if d["reach"][i]:assert abs(px[-1]-int(d["pixels"][i]))<=1
                    cache[i]=px
                old,new=cache[x["old"]],cache[x["jepa"]]
                assert old[:3]==new[:3]
                rec.append((house,budget,old,new))
                if not d["reach"][ix[groups=="old"]].any():rejected.append(rec[-1])
                for arm,px in (("old",old),("jepa",new)):
                    totals[arm+"_executed_actions"]+=len(px)-1
                if x["old"] is not None and x["jepa"] is not None and x["old"]==x["jepa"]:assert old==new
        finally:sim.close()
    receipt=json.loads((HERE/"selected_replay.json").read_text())["counts"]
    assert totals["old_executed_actions"]==receipt["old_actions"] and totals["jepa_executed_actions"]==receipt["jepa_actions"]
    results=dict(all=summarize(rec),true_all_blocked=summarize(rejected),physical=totals,
                 metric="Same-instance mask >=369px in any actually rendered RGB-aligned image. Step zero = initial state, step one = left scan, step two = restored orientation. Every subsequent primitive through stop or contact contributes a frame. Final success and correct multi-instance reporting are separate outcomes. Cumulative first seen carries forward after early stop; budgets per group = 22/26/30.")
    (HERE/"anytime_replay.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v["counts"] for k,v in results["all"].items()}))


if __name__=="__main__":main()
