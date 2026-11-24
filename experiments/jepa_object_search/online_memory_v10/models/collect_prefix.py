"""Collect observed physical loops ending at the frozen branch origin. No candidate outcomes are opened."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs

D=Path(__file__).resolve().parent
ROOT=D.parents[1]
SRC=ROOT/'jepa_separation_v8_20261123'
sys.path.insert(0,str(ROOT/'mapping_runtime/pilots/jepa_round2_five_20261121'))
from render_labels import simulation

def run(split,house):
    rows=json.loads((SRC/'primitives'/f'{split}.index.json').read_text())
    starts=(SRC/'natural_holdout/raw/dev' if split=='holdout' else
            ROOT/'a_exploration/paired_branches_v2'/('train' if split=='train' else 'dev'))
    dest=D.parent/'prefixes';dest.mkdir(exist_ok=True)
    sim=simulation(house)
    try:
        for row in (r for r in rows if r['house']==house):
            sid=row['sid'];uid=hashlib.sha256(sid.encode()).hexdigest();output=dest/(uid+'.npz')
            if output.exists():continue
            manifest=starts/house/Path(sid).parent.name/Path(sid).name/'manifest.json'
            pose=np.array(json.loads(manifest.read_text())['frozen_pose_xyz_xyzw'])
            s=habitat_sim.AgentState();s.position=pose[:3];s.rotation=quat_from_coeffs(pose[3:])
            agent=sim.get_agent(0);agent.set_state(s)
            def image():return sim.get_sensor_observations()['rgb'][:,:,:3].copy()
            imgs=[image()];acts=[-1];idx=[0];blocked=[False];yaw=0;xy=np.zeros(2)
            pos=[xy.copy()];head=[[1.,0.]]
            def act(action,keep=False):
                nonlocal xy,yaw
                previous=agent.get_state().position.copy()
                prior=image() if action==0 else None
                agent.act(('move_forward','turn_left','turn_right')[action])
                rgb=image() if keep or action==0 else None
                # RGB-only odometry proxy used by the memory. Simulator pose is used only
                # to drive and validate the closed collection excursion, never by the policy.
                stuck=bool(action==0 and np.abs(rgb.astype(float)-prior).mean() < 1.0)
                if action==0 and not stuck:xy=xy+.25*np.array([math.cos(yaw),math.sin(yaw)])
                if action!=0:yaw+=(1 if action==1 else -1)*math.pi/6
                acts.append(action);pos.append(xy.copy());head.append([math.cos(yaw),math.sin(yaw)]);blocked.append(stuck)
                if keep:
                    imgs.append(rgb);idx.append(len(acts)-1)
                return float(np.linalg.norm(agent.get_state().position-previous))
            # Direction and requested budget are assigned independently of scene and outcomes.
            sign=1 if int(uid[:2],16)%2 else 2
            turns=1+int(uid[2:4],16)%2
            for _ in range(turns):act(sign,True)
            advanced=0
            for _ in range(5):
                distance=act(0,True)
                if distance<.245:break
                advanced+=1
            for j in range(6):act(1,j in (2,5))
            for _ in range(advanced):act(0,True)
            for j in range(6):act(1,j in (2,5))
            for _ in range(turns):act(3-sign,True)
            # A sliding collision or asymmetric return can prevent exact closure. Use the
            # actual captured rotation-only prefix for that origin without discarding it.
            offset=np.linalg.norm(agent.get_state().position-pose[:3])
            ma=float(np.abs(image().astype(float)-imgs[0]).mean())
            closed=bool(offset<=.04 and ma<5.0)
            if not closed:
                agent.set_state(s);imgs=[image()];idx=[0];acts=[-1];pos=[np.zeros(2)];head=[[1.,0.]];blocked=[False]
                yaw=0.;xy=np.zeros(2)
                for j in range(12):act(1,j%2==1)
            np.savez_compressed(output,rgb=imgs,frame_steps=idx,steps=np.arange(len(acts)),actions=acts,
                xy=pos,heading=head,collided=blocked,translated=closed and advanced>0,attempted_closure_error=offset)
    finally:sim.close()

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
