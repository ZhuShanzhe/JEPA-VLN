"""Regenerate source depth, call the real correction API, then re-execute its selected controls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"pose_horizon_20261128"
sys.path.insert(0,str(OLD))
PRIVATE=HERE/"private_repair"/"runtime"


def physical(replay,batch=False,shortest=False):
    from collect_paths import houses,simulator,ASSETS,inventory,choose_sources,SAMPLES_PER_HOUSE,camera,SOURCE_H
    metrics=Counter()
    for house in sorted(houses()["test"])[:3]:
        sim=simulator(house)
        seed=int(hashlib.sha256(("path_feedback:paired:v1:"+house).encode()).hexdigest()[:8],16)%(2**31-1)
        sim.seed(seed);sim.pathfinder.seed(seed)
        rng=np.random.default_rng(seed)
        try:
            sources=choose_sources(sim,inventory(sim,ASSETS),rng,SAMPLES_PER_HOUSE)
            with np.load(HERE/"private_data"/("test_"+house+".npz")) as z:
                table={k:z[k] for k in ("source_index","plan_id","group","reach","pixels","end_x","end_z")}
            for sid,(_,oid,p,yaw,inputs) in enumerate(sources[:2]):
                raw=camera(sim,p,yaw,SOURCE_H,0)
                state=sim.get_agent(0).get_state()
                stem=PRIVATE/f"{house}_{sid}"
                if not replay:
                    np.savez_compressed(stem.with_suffix(".npz"),depth=raw["depth"].squeeze())
                    continue
                for entry in json.loads(stem.with_suffix(".shortest.json" if shortest else ".batch.json" if batch else ".json").read_text()):
                    i=entry["index"]
                    metrics["tested"]+=1
                    metrics["abstained"]+=int(i is None)
                    if i is None:continue
                    ids=np.flatnonzero((table["group"]==entry["group"])&(table["plan_id"]==i)); idx,=ids
                    ag=sim.get_agent(0)
                    ag.set_state(state,reset_sensors=False,infer_sensor_states=False)
                    metrics["issued"]+=1
                    metrics["issued_alternative"]+=int(i!=0)
                    contact=False
                    for a in entry["actions"]:
                        before=ag.state.position.copy()
                        bumped=ag.act({1:"move_forward",2:"turn_left",3:"turn_right"}[a])
                        delta=np.linalg.norm((ag.state.position-before)[[0,2]])
                        if a==1 and (bumped or delta<.23):contact=True;break
                    distance=.25*int(entry["group"].rsplit("/d",1)[1])
                    offset=distance*np.array([-np.sin(yaw),-np.cos(yaw)])
                    reached=not contact and np.linalg.norm((ag.state.position-p)[[0,2]]-offset)<.14
                    assert reached==bool(table["reach"][idx]),(house,sid,i)
                    assert np.allclose(ag.state.position[[0,2]],[table["end_x"][idx],table["end_z"][idx]],atol=1e-4)
                    if reached:
                        pix=int((sim.get_sensor_observations()["ids"]==oid).sum())
                        assert abs(pix-table["pixels"][idx])<=1
                    metrics["reached"]+=int(reached)
                    metrics["joint"]+=int(reached and pix>=369)
        finally:sim.close()
    if replay:
        receipt="repair_shortest_receipt.json" if shortest else "repair_batch_receipt.json" if batch else "repair_runtime_receipt.json"
        (HERE/receipt).write_text(json.dumps(dict(metrics),indent=2)+"\n")
        print(metrics)


def infer(batch=False,shortest=False):
    import torch
    from checker import PathChecker
    from observation_head import HorizonCameraPredictor
    from replan import RepairScorer
    from local_repair import consult,select_or_repair
    torch.set_num_threads(2)
    checker=PathChecker.load(HERE/"private_checker"/"checker.joblib")
    if shortest:
        from mixed_checker import MixedPathChecker
        checker=MixedPathChecker.load(HERE/"private_checker_mixed"/"checker.joblib")
    jepa=HorizonCameraPredictor.from_checkpoints(OLD/"private_train")
    scorer=RepairScorer(HERE/"private_repair"/"with_hint.pt")
    for file in sorted(PRIVATE.glob("*.npz")):
        house,source=file.stem.rsplit("_",1)
        versions=[]
        with np.load(file) as f:depth=f["depth"]
        with np.load(HERE/"private_data"/("test_"+house+".npz")) as z:
            source_mask=z["source_index"]==int(source)
            for group in sorted(np.unique(z["group"][source_mask])):
                ids=np.flatnonzero(z["group"]==group);ids=ids[np.argsort(z["plan_id"][ids])]
                actions=z["action"][ids]
                options=[tuple(map(int,a[a!=0])) for a in actions]
                poses=dict(zip(options,z["pose"][ids]))
                if batch:
                    op=select_or_repair(options[:3],depth,z["context"][ids[0]],lambda p:poses[tuple(p)],
                                        checker,jepa,scorer,max(map(len,options)),selection="shortest" if shortest else "visible")
                    index=options.index(tuple(op["actions"])) if op["status"]=="execute" else None
                else:
                    op=consult(options[0],depth,z["context"][ids[0]],lambda p:poses[tuple(p)],
                               checker,jepa,scorer,max(map(len,options)))
                    index=op["index"]
                    if index is not None:assert tuple(op["actions"])==options[index]
                versions.append(dict(group=str(group),index=index,actions=op["actions"]))
        file.with_suffix(".shortest.json" if shortest else ".batch.json" if batch else ".json").write_text(json.dumps(versions))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("depth","infer","replay"))
    parser.add_argument("--batch",action="store_true")
    parser.add_argument("--shortest",action="store_true",help="exercise the adopted mixed-checker and shortest admitted policy")
    args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    if args.mode=="infer":infer(args.batch or args.shortest,args.shortest)
    else:physical(args.mode=="replay",args.batch or args.shortest,args.shortest)
