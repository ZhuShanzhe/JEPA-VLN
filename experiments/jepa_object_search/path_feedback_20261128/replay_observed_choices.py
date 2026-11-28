"""Replay the executable decisions chronologically in Habitat; grade same instance only after execution."""
from collections import Counter,defaultdict
import json
import numpy as np
from collect_paths import HERE,ASSETS,inventory,SOURCE_H,camera,options,simulator

ACT={1:"move_forward",2:"turn_left",3:"turn_right"}
YAW_LOOK=((),(2,),(2,3,3))


def replay(sim,p,yaw,sid,dist,turns,plan,expected):
    state=camera(sim,p,yaw,SOURCE_H,0.)
    assert np.array_equal(state["depth"].squeeze(),expected[0])
    ag=sim.get_agent(0)
    spent=0;blocked=False
    for t in turns:
        ag.act(ACT[t]);spent+=1
    if turns:
        assert np.array_equal(sim.get_sensor_observations()["depth"].squeeze(),expected[1 if len(turns)==1 else 2])
    for a in plan:
        before=ag.state.position.copy();contact=ag.act(ACT[a]);spent+=1
        if a==1 and (contact or np.linalg.norm((ag.state.position-before)[[0,2]])<.23):
            blocked=True;break
    target=p+[-.25*dist*np.sin(yaw),0.,-.25*dist*np.cos(yaw)]
    arrived=bool(plan) and not blocked and np.linalg.norm((ag.state.position-target)[[0,2]])<.14
    pix=int((sim.get_sensor_observations()["ids"]==sid).sum()) if arrived else 0
    return bool(arrived),blocked,pix,spent


def main():
    path=HERE/"private_observe/holdout_decisions.json"
    cases=json.loads(path.read_text())
    groups=defaultdict(list)
    for row in cases:groups[row["house"]].append(row)
    totals=defaultdict(Counter);by_house=defaultdict(lambda:defaultdict(Counter))
    for house,rows in groups.items():
        sim=simulator(house)
        try:
            inventory(sim,ASSETS)
            with np.load(HERE/"private_looks"/(house+".npz")) as src:
                x,yaw,sid,depth=src["position"],src["yaw"],src["sid"],src["depth"]
            for row in rows:
                i=row["cue"];n=row["distance"];cache={}
                for arm,choice in row["decisions"].items():
                    sent=choice["index"] is not None
                    cost=choice["sensing_actions"]
                    turns={0:(),1:(2,),3:(2,3,3)}[cost]
                    plan=tuple(choice["actions"])
                    key=(turns,plan)
                    assert cost+len(plan)<=n+20
                    if key not in cache:cache[key]=replay(sim,x[i],yaw[i],sid[i],n,turns,plan,depth[i])
                    arrived,blocked,pix,spent=cache[key]
                    tally=dict(groups=1,issued=int(sent),deferred=int(not sent),arrived=int(arrived),
                        arrived_and_clear=int(arrived and pix>=369),physical_contact_or_stall=int(blocked),
                        actions=spent,sensing_actions=cost,extra_views=int(cost>0),checks=choice["checks"])
                    totals[arm].update(tally);by_house[arm][house].update(tally)
        finally:sim.close()
    result=dict(scope="First five available cue sources per test building, requesting the native observation waypoint 1m or 2m straight ahead. Three-view arms may spend one left turn and then two right turns, and undo their orientation in the returned maneuver. All arms have the same per-group cap of straight steps plus 20. Deferred groups have zero reach and are counted separately from contact. Repeated score arms share identical physical replays.",
                houses=len(groups),rows=len(cases),results=totals,paired={})
    rng=np.random.default_rng(1128)
    for start in ("jepa","straight"):
        for other in ("raw","oneshot_3","trained_1"):
            arm=start+"_trained_3";control=start+"_"+other
            hs=sorted(groups)
            delta=[(by_house[arm][h]["arrived_and_clear"]-by_house[control][h]["arrived_and_clear"])/by_house[arm][h]["groups"] for h in hs]
            means=rng.choice(delta,(5000,len(hs)),replace=True).mean(1)
            result["paired"][arm+"_minus_"+other]=dict(mean_house_difference=float(np.mean(delta)),interval95=np.quantile(means,[.025,.975]).tolist())
    (HERE/"observed_replay.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:[v["groups"],v["arrived_and_clear"],v["physical_contact_or_stall"],v["deferred"]] for k,v in totals.items()}))

if __name__=="__main__":main()
