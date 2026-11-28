"""Label pixels occupied by a different rendered surface in the ten native-unclear, isolation-clear requests."""
from collections import Counter
import csv
import json
import re
from occlusion_probe import HERE,REFIT,VIEWS,LOGS,H,FAR,CUT,ASSETS,simulator,inventory,camera,ACTION,hs,mn,np


def main():
    houses=json.loads((VIEWS/"view_design.json").read_text())["cohorts"]["blind"]
    labels={}
    with (ASSETS/"metadata/fpmodels-with-decomposed.csv").open() as stream:
        for row in csv.DictReader(stream):
            labels[row["id"]]=(row["main_wnsynsetkey"] or row["wnsynsetkey"]).split(".")[0]
    counts=Counter();examples=[]
    for h in houses:
        sim=simulator(h)
        try:
            inventory(sim,ASSETS);ag=sim.get_agent(0)
            objects=sim.get_rigid_object_manager().get_objects_by_handle_substring("")
            obj={o.semantic_id:o for o in objects.values()}
            kinds={sid:next((labels[k] for k in re.findall(r"[a-fA-F0-9]{40}",o.handle) if k in labels),"unlabelled_object") for sid,o in obj.items()}
            cases=json.loads((HERE/"private_occlusion"/(h+".json")).read_text())
            paths=sorted((REFIT/"private_inputs"/h).glob("*.npz"))
            for r,p in zip(cases,paths):
                if r["old"]!="never" or max(r["isolated"])<CUT:continue
                t=int(np.argmax(r["isolated"]))
                ev=json.loads((LOGS/"private_roll"/h/(p.stem+".json")).read_text())["events"]
                with np.load(VIEWS/"private_render_blind60"/("blind_"+h)/(p.stem+"_sensor.npz")) as s:
                    start,yaw,sid=s["position"],float(s["yaw"]),int(s["oid"])
                obs=camera(sim,start,yaw,H,0)
                for frame in ev[1:t+1]:
                    ag.act(ACTION[frame["action"]]);obs=sim.get_sensor_observations()
                walls=obs["ids"].copy()
                assert int((walls==sid).sum())==r["native"][t]
                target=obj[sid]; transform=target.transformation;motion_type=target.motion_type
                target.motion_type=hs.physics.MotionType.KINEMATIC
                target.translation=mn.Vector3(*(np.array(transform.translation)+FAR))
                state=ag.get_state()
                theta=2*np.arctan2(state.rotation.y,state.rotation.w)
                iso=camera(sim,state.position+FAR,theta,H,0)["ids"]==sid
                assert abs(int(iso.sum())-r["isolated"][t])<=2
                target.transformation=transform;target.motion_type=motion_type
                blocked=walls[iso&(walls!=sid)]
                by_cat=Counter(kinds.get(int(v),"static_or_unlabelled_scene") or "unlabelled_object" for v in blocked)
                cat,n=by_cat.most_common(1)[0]
                counts[cat]+=1
                examples.append(dict(target=kinds[sid],dominant_blocker=cat,fraction_of_missing_pixels=round(n/len(blocked),3),
                              visible=r["native"][t],isolated=r["isolated"][t]))
        finally:sim.close()
    assert len(examples)==10
    report={"protocol":"One pose per old-never-clear request that became clear in isolation. Choose maximum isolated footprint from all real camera poses; identify categories of frontmost rendered surfaces at missing pixels.",
            "main_blocker_by_request":dict(counts),"category_and_footprint_examples":examples}
    (HERE/"occlusion_blockers.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)


if __name__=="__main__":main()
