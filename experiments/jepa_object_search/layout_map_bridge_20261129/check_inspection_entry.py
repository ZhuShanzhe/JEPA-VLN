"""Exercise the real local-inspection entry in Habitat on the first frozen focus task per house."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT/"new_view_20261129"
REACTIVE=ROOT/"reactive_view_20261129"
sys.path[:0]=list(map(str,[ROOT/"private_sources/MOST-Nav",REACTIVE,ROOT/"refined_confirmation_20261129",
                            ROOT/"clarity_fix_20261129"]))
from agents.navigator import OccupancyGrid
from inspection_session import ReactiveInspection
from strict_rescue import StrictRescueMemory
from verify_inspection_session import receive
from layout_inspection import bind_local_inspection
from layout_memory import LayoutMap, RADIUS, CELL


def run():
    torch.set_num_threads(2)
    rows=[]
    houses=json.loads((BASE/"view_design.json").read_text())["cohorts"]["blind"]
    inspection=ReactiveInspection.load()
    inspection.verifier=StrictRescueMemory
    for house in houses:
        prior=REACTIVE/"private_roll"/house
        selected=next(p for p in sorted(prior.glob("*.json"),key=lambda q:int(q.stem)) if json.loads(p.read_text())["focus"])
        budget=json.loads(selected.read_text())["budget"]
        with np.load(BASE/"private_render_blind60"/("blind_"+house)/(selected.stem+"_sensor.npz")) as f:
            box=f["box"].tolist()
        actuator=bind_local_inspection(inspection)
        recorded=[]
        native=SimpleNamespace(episode_id=house,pose_free=True)
        local_snapshots=0
        # Here the grid is generated from measured depth/odometry of this SAME scene.
        # Genuine RGB-only VGGT-SLAM production is independently checked by check_live_bridge.
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            packet=Path(temp)/"current.npz"
            env=dict(os.environ,HABITAT_SIM_LOG="quiet",MAGNUM_LOG="quiet",OMP_NUM_THREADS="1")
            env.pop("LD_LIBRARY_PATH",None)
            env.pop("PYTHONPATH",None)
            proc=subprocess.Popen([str(ROOT/"simulation/habitat-env/bin/python"),"-u",
                                   str(REACTIVE/"verify_inspection_session.py"),"--house",house,"--packet",str(packet)],
                                  stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
            def pair(command):
                nonlocal local_snapshots
                frame=receive(proc,packet,command)
                t=len(recorded)
                recorded.append(frame)
                layout=LayoutMap.from_depth_history(recorded,episode=house,frame_id=t)
                # Native map axes are +X=forward and +Y=left. Our right/forward
                # array has indices [z,x], so native [y,x] is its transposed x flip.
                native_cells=layout.cells.T[::-1,:]
                origin=np.array([layout.cz,-layout.cx])-(RADIUS+.5)*CELL
                grid=OccupancyGrid(CELL,origin,native_cells==1,native_cells==2,observed=native_cells>0)
                grid.unit_per_m=1.
                native._frontier_grid=grid
                native._frontier_pose=(frame["z"],-frame["x"],frame["yaw"])
                native._frontier_scale=1.
                native._frontier_revision=t+1
                native._last_frontier_step=t
                obs=SimpleNamespace(episode_id=house,step_count=t)
                local_snapshots+=1
                return obs, dict(frame,episode=house,frame_id=t)
            try:
                outcome=actuator.run(native,lambda:pair("source "+selected.stem),lambda a:pair("act "+str(a)),
                                     box,budget-18,budget,Path(temp)/"private_out")
                assert outcome["actions_used"]==len(recorded)-1<=budget
                assert local_snapshots==len(recorded)
            finally:
                proc.stdin.close()
                if proc.wait(timeout=60)!=0:
                    raise RuntimeError("Habitat worker stopped unexpectedly")
        rows.append(dict(actions=outcome["actions_used"],snapshots=local_snapshots,budget=budget,status=outcome["status"]))
    report=dict(scope="Live three-house causal RGB-D grid integration; mapped with measured RGB-D in native OccupancyGrid, not with online VGGT-SLAM. First frozen focus task per house. Single-floor calibrated local observation; no map or target oracle.",houses=len(rows),requests=rows)
    (HERE/"inspection_entry_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)

if __name__=="__main__":
    with threadpool_limits(limits=2):
        run()
