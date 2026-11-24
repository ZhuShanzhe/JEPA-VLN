"""Collect paired causal start images and isolated future label/feature targets in Habitat."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import habitat_sim as hs
from habitat_sim.utils.common import quat_from_angle_axis
import magnum as mn
import numpy as np
from PIL import Image

folder=Path(__file__).resolve()
sys.path[:0]=[str(folder.parents[1]),str(folder.parents[2]/"code")]
from collect_streams import ACTIONS, PLANS, configure


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dst",type=Path);ap.add_argument("objects",type=Path)
    ap.add_argument("house",type=int)
    args=ap.parse_args()
    d,assets,h=args.dst.resolve(),args.objects.resolve(),args.house
    name=f"house_{h:02}"
    room=d/"assets"/(name+".stage_config.json")
    store=d/"raw"/name;store.mkdir(parents=True,exist_ok=True)
    actor_dir=store/"actor";teacher_dir=store/"teacher"
    for f in (actor_dir,teacher_dir):f.mkdir(exist_ok=True)
    manifest=[]
    with hs.Simulator(configure(room)) as sim:
        tm=sim.get_object_template_manager();om=sim.get_rigid_object_manager();agent=sim.get_agent(0)
        loaded={s:tm.load_configs(str(assets/(s+".object_config.json")))[0] for s in
                ("cup","sofa","coffee_station","lamp")}
        def add(s,xyz,sem):
            obj=om.add_object_by_template_id(loaded[s]);obj.translation=mn.Vector3(*xyz)
            obj.motion_type=hs.physics.MotionType.STATIC;obj.semantic_id=sem
            return obj
        def store_rgb(array):
            frame=np.ascontiguousarray(array[:,:,:3])
            digest=hashlib.sha256(frame).hexdigest()
            dest=store/(digest+".png")
            if not dest.exists():Image.fromarray(frame).save(dest,compress_level=1)
            return digest
        for variant in range(3):
            rng=np.random.RandomState(9347+1009*h+variant*17)
            camera_z=float(rng.uniform(3.66,3.80)); cue_x=float(rng.uniform(.72,.92))
            cue_z=float(rng.uniform(2.12,2.39)); target_x=float(rng.uniform(1.95,2.16))
            target_z=float(rng.uniform(-1.78,-1.62))
            for mode in ("aligned","contradictory","no_cues"):
                for left in (0,1):
                    key=f"{variant}_{mode}_{left}"
                    entities=[add(s, [(-1,1)[j^left]*target_x,0.,target_z],101+j) for j,s in enumerate(("cup","sofa"))]
                    if mode!="no_cues":
                        for side in range(2):
                            k=side^left^int(mode=="contradictory")
                            entities.append(add(("coffee_station","lamp")[k],[(-1,1)[side]*cue_x,0.,cue_z],301+k))
                    nav=hs.NavMeshSettings();nav.set_defaults();nav.agent_height,nav.agent_radius,nav.include_static_objects=1.45,.12,True
                    assert sim.recompute_navmesh(sim.pathfinder,nav)
                    start=np.asarray(sim.pathfinder.snap_point(np.array([0,0,camera_z],np.float32)))
                    def initial():
                        st=hs.AgentState();st.position=start;st.rotation=quat_from_angle_axis(0.,np.asarray([0.,1.,0.]));agent.set_state(st)
                    initial()
                    context=[];target_counts=[];cue_counts=[]
                    for _ in range(12):
                        obs=sim.get_sensor_observations();context.append(store_rgb(obs["rgb"]))
                        target_counts.append([int((obs["instance"]==101+j).sum()) for j in (0,1)])
                        cue_counts.append([int((obs["instance"]==301+j).sum()) for j in (0,1)])
                        agent.act("turn_left")
                    future=[];y=[];length=[];end_px=[]
                    for side,plan in enumerate(PLANS):
                        initial();gain=np.zeros(2,bool);distance=0.
                        r=[]
                        for i,a in enumerate(plan,1):
                            before=agent.get_state().position.copy();agent.act(ACTIONS[a])
                            pos=agent.get_state().position.copy()
                            distance+=float(np.linalg.norm(pos-before))
                            obs=sim.get_sensor_observations()
                            pix=np.array([int((obs["instance"]==101+j).sum()) for j in (0,1)])
                            near=np.array([np.linalg.norm((np.array(o.translation)-pos)[[0,2]])<=2. for o in entities[:2]])
                            gain |= (pix>=64)&near
                            if i in (20,24):r.append(store_rgb(obs["rgb"]))
                        future.append(r);y.append(gain.astype(int).tolist());length.append(distance);end_px.append(pix.tolist())
                    truth=dict(future=future,gain=y,travel=length,context_target_max=np.max(target_counts,axis=0).tolist(),
                               context_cue_max=np.max(cue_counts,axis=0).tolist(),end_target_pixels=end_px)
                    actor=dict(context=context,plans=PLANS)
                    (actor_dir/(key+".json")).write_text(json.dumps(actor))
                    (teacher_dir/(key+".json")).write_text(json.dumps(truth))
                    manifest.append(dict(key=key,variant=variant,mode=mode,placement=left))
                    if np.max(target_counts)>0 or not np.allclose(length,5.,atol=.02) or np.array(y).sum(0).tolist()!=[1,1]:
                        raise RuntimeError(str((key,truth)))
                    if mode!="no_cues" and min(truth["context_cue_max"])<64:
                        raise RuntimeError(f"Source cues invisible: {key}")
                    for obj in entities:om.remove_object_by_id(obj.object_id)
        (store/"manifest.json").write_text(json.dumps(manifest))
        print("VERIFIED",name,len(manifest),flush=True)


if __name__=="__main__":main()
