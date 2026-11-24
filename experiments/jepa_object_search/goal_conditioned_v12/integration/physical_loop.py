"""Exercise the installed RGB route scorer on two real Habitat decisions."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def worker(project, folder):
    sys.path.insert(0,str(project/"jepa_route_v12/refinement"))
    from live_residual import BoundedSelector
    model=BoundedSelector(project)
    memory=model.old.remem(project)
    memory.reset()
    xy=np.zeros(2);yaw=0.;clock=0
    for line in sys.stdin:
        request=json.loads(line)
        if request["kind"]=="stop":break
        with np.load(folder/"from_sim.npz") as inp:
            goal=str(inp["goal"]);words=model.encoder.words([goal])
            cues={}
            if len(inp["frames"]):
                _,matches=model.encoder.encode(inp["frames"],words)
                cues=dict(zip(inp["times"].tolist(),matches[:,0].tolist()))
            for i,(a,c) in enumerate(zip(inp["actions"],inp["bumps"]),1):
                if a==0 and not c:
                    xy+=.25*np.array([math.cos(yaw),math.sin(yaw)])
                if a==1:yaw+=math.pi/6
                if a==2:yaw-=math.pi/6
                clock+=1
                memory.observe(xy,clock,[math.cos(yaw),math.sin(yaw)],cues.get(i),bool(c))
            receipt=dict(observed_actions=len(inp["actions"]),observed_frames=len(cues),clock=clock)
            if request["kind"]=="score":
                scores,_=model.scores(inp["views"],goal,inp["plans"],memory,time=clock,
                                      xy=xy,facing=[math.cos(yaw),math.sin(yaw)])
                assert len(scores)==3 and np.isfinite(scores).all()
                receipt.update(choice=int(np.argmax(scores)),scores=scores.tolist())
        print("V12_REPLY "+json.dumps(receipt),flush=True)


def next_plans(sim, start, rng, act, error):
    from collect_branches import fork
    for _ in range(40):
        ends=fork(sim,start,rng)
        if ends is None:continue
        plans=[]
        for dest in ends:
            follower=sim.make_greedy_follower(agent_id=0,goal_radius=.22,stop_key="stop",
                                               forward_key=act[0],left_key=act[1],right_key=act[2])
            try: seq=[act.index(a) for a in follower.find_path(dest) if a in act]
            except error:break
            if len(seq)<8:break
            plans.append((seq+[1,1,2,2,2,2,1,1]*3)[:24])
        if len(plans)==3 and len(set(map(tuple,plans)))==3:
            return np.asarray(plans)
    return None


def simulate(project, asset, model_python, output):
    old=project/"jepa_hidden_v11/hssd"
    # The V12 collector normalizes Magnum's navigable-point vectors.
    sys.path.insert(0,str(project/"jepa_route_v12/expansion/hssd"))
    from collect_branches import ACT, simulate as open_scene, world_state
    import habitat_sim
    cohort=json.loads((old/"cohort.json").read_text())["calibration"]
    src=next(f for h in sorted(cohort) for f in sorted((old/"calibration"/h).glob("*.npz")))
    entries=json.loads((src.parent/"private_replay.json").read_text())
    pose=next(x for x in entries if x["origin"]==src.stem)
    with np.load(src) as raw:goal=str(raw["words"][0]);plans=raw["actions"].copy()
    sim=open_scene(asset,src.parent.name);agent=sim.get_agent(0)
    agent.set_state(world_state(pose["position"],pose["yaw"]));sim.pathfinder.seed(5701)
    rng=np.random.default_rng(5701)
    result=dict(source="first calibration house / origin in filename order; first stored query",
                scorer="BoundedSelector rv_jepa seed=11",decisions=[],selection_uses_ground_truth=False)
    pending=[];bumps=[];frames=[];times=[]
    def step(a,observe=False):
        collision=agent.act(ACT[a])
        pending.append(a);bumps.append(bool(a==0 and collision))
        if observe:
            frames.append(sim.get_sensor_observations()["rgb"][:,:,:3].copy());times.append(len(pending))
    try:
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            proc=subprocess.Popen([str(model_python),"-Bu",__file__,"worker",str(project),str(folder)],
                                  stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            def request(kind,**other):
                np.savez(folder/"from_sim.npz",goal=goal,actions=pending,bumps=bumps,times=times,frames=np.asarray(frames),**other)
                proc.stdin.write(json.dumps(dict(kind=kind))+"\n");proc.stdin.flush()
                for line in proc.stdout:
                    if line.startswith("V12_REPLY "):
                        data=json.loads(line.split(" ",1)[1])
                        pending.clear();bumps.clear();times.clear();frames.clear()
                        return data
                raise RuntimeError("scorer exited before replying")
            try:
                for decision in range(2):
                    if decision:
                        plans=next_plans(sim,np.asarray(agent.get_state().position).copy(),rng,ACT,habitat_sim.errors.GreedyFollowerError)
                        if plans is None:
                            result["termination"]="no three distinct routes at second state"
                            break
                    start=agent.get_state();scan=[]
                    # Four genuine camera turns; yaw is restored for route execution.
                    for i,a in enumerate((2,1,1,2)):
                        step(a,i<3)
                        if i<3:scan.append(frames[-1])
                    at=agent.get_state()
                    rotation_match=abs(float(np.dot(at.rotation.components,start.rotation.components)))
                    assert np.linalg.norm(at.position-start.position)<1e-5 and abs(rotation_match-1)<1e-5
                    choose=request("score",views=np.stack(scan),plans=plans)
                    path=0.
                    for t,a in enumerate(plans[choose["choice"]],1):
                        before=agent.get_state().position.copy()
                        step(int(a),t in (6,12,18,24))
                        path+=float(np.linalg.norm(agent.get_state().position-before))
                    result["decisions"].append(dict(scan_steps=4,search_steps=len(plans[0]),candidate_count=len(plans),
                                                     distinct_plans=len(set(map(tuple,plans))),odometric_path_m=path,
                                                     search_collisions=sum(bumps),search_rgb_samples=len(frames),**choose))
                result["last_observations"]=request("observe")
                result.setdefault("termination","two successive real choices executed")
                proc.stdin.write('{"kind":"stop"}\n');proc.stdin.flush()
                if proc.wait(timeout=30):raise RuntimeError("scorer failed")
            finally:
                if proc.poll() is None:
                    proc.terminate();proc.wait()
    finally:sim.close()
    output.write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    if sys.argv[1]=="worker":worker(Path(sys.argv[2]),Path(sys.argv[3]))
    else:simulate(*(Path(s) for s in sys.argv[1:]))
