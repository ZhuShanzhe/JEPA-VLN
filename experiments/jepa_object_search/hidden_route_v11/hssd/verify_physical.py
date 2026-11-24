"""Independently re-run published observation and discovery criteria on archived actions."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from collect_branches import ACT, WORDS, simulate, semantics, world_state


def verify(asset,root,split,house):
    path=root/split/house
    rec=json.loads((path/"receipt.json").read_bytes())
    if rec["maximum_trials"]!=450:raise RuntimeError("await completed house")
    entries=json.loads((path/"private_replay.json").read_bytes())
    assert rec["origins"]==len(entries)
    counts=dict(origins=0,panoramic_views=0,routes=0,actions=0,future_images=0,
                evaluated_queries=0,unique_eligible_queries=0)
    if not entries:return counts
    sim=simulate(asset,house);agent=sim.get_agent(0);cls=semantics(sim,asset)
    try:
        for entry in entries:
            position,yaw=entry["position"],entry["yaw"]
            origin=entry["origin"];start=world_state(position,yaw)
            with np.load(path/(origin+".npz")) as data:
                img=data["ctx"];plans=data["actions"];expected=data["y"];future=data["future"];cf=data["collided"]
                names=data["words"].tolist();qi=[WORDS.index(x) for x in names]
                assert np.all(np.isfinite(img)) and len(set(tuple(p) for p in plans))>=2
                for i in range(len(plans)):
                    for j in range(i):
                        if np.array_equal(plans[i],plans[j]):assert np.array_equal(expected[i],expected[j])
                present=set()
                for k in range(12):
                    agent.set_state(world_state(position,yaw+k*math.pi/6))
                    sensors=sim.get_sensor_observations()
                    lut=set(np.unique(sensors["ids"]).tolist())
                    present.update(cls[x] for x in lut if 0<=x<len(cls))
                    idx={11:0,0:1,1:2}.get(k)
                    if idx is not None:assert np.array_equal(img[idx],sensors["rgb"][:,:,:3])
                assert not present.intersection(qi)
                counts["panoramic_views"]+=12
                for i,plan in enumerate(plans):
                    agent.set_state(start);found=[set() for _ in names]
                    for t,action in enumerate(plan,1):
                        bumped=agent.act(ACT[action]);assert bool(bumped and action==0)==bool(cf[i,t-1])
                        if t in (6,12,18,24):
                            sensors=sim.get_sensor_observations()
                            assert np.array_equal(sensors["rgb"][:,:,:3],future[i,t//6-1])
                            depth=sensors["depth"]; ids=sensors["ids"]
                            for ii,nn in zip(*np.unique(ids[(depth>0)&(depth<=2.)],return_counts=True)):
                                if 0<int(ii)<len(cls) and nn>=32 and cls[ii] in qi:
                                    found[qi.index(cls[ii])].add(int(ii))
                    assert [len(q) for q in found]==expected[i].tolist()
                assert np.all((expected.min(0)==0)&(expected.max(0)>0))
                counts["origins"]+=1;counts["routes"]+=len(plans);counts["actions"]+=plans.size
                counts["future_images"]+=len(plans)*4;counts["evaluated_queries"]+=len(qi)
    finally:sim.close()
    counts.pop("unique_eligible_queries",None)
    return counts

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("asset",type=Path);p.add_argument("root",type=Path);p.add_argument("split");p.add_argument("house")
    a=p.parse_args();counts=verify(a.asset,a.root,a.split,a.house)
    dest=a.root/"physical_checks";dest.mkdir(exist_ok=True)
    (dest/(a.split+"-"+a.house+".json")).write_text(json.dumps(counts))
