"""Collect one continuous calibration path per natural source, including contacts and alternate approaches."""
import argparse,math,json,os,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'new_view_20261129'
sys.path.insert(0,str(PRIOR))
from sample_views import camera,simulator,inventory,ASSETS,ACTION,SOURCE_H

def run(i):
    os.umask(0o077)
    for house in json.loads((PRIOR/'view_design.json').read_text())['cohorts']['val'][i::3]:
        directory=HERE/'private_memory_validation'/house;directory.mkdir(exist_ok=True,parents=True)
        sim=simulator(house)
        try:
            inventory(sim,ASSETS);ag=sim.get_agent(0)
            for f in sorted((PRIOR/'private_render'/('val_'+house)).glob('*_sensor.npz')):
                num=int(f.name.split('_')[0]);dest=directory/(str(num)+'.npz')
                if dest.exists():continue
                with np.load(f) as s:start=s['position'];yaw=float(s['yaw']);box=s['box'];oid=int(s['oid'])
                camera(sim,start,yaw,SOURCE_H,0)
                sign=2 if num%2 else 3;other=5-sign
                plan=[2,3]+([1]*7+[sign,1,1,other,1,1,other,1,1,sign]*2 if num%3==0 else [sign,1,1,1,other,1,1,other,1,1,sign]*3)
                frames=[]
                for t in range(23):
                    pos=ag.state.position-start;q=ag.state.rotation;a=2*math.atan2(q.y,q.w)-yaw
                    pose=(math.cos(yaw)*pos[0]-math.sin(yaw)*pos[2],-math.sin(yaw)*pos[0]-math.cos(yaw)*pos[2],math.atan2(math.sin(a),math.cos(a)))
                    obs=sim.get_sensor_observations()
                    frames.append((obs['rgb'][...,:3].copy(),obs['depth'].squeeze().astype(np.float16),np.packbits(obs['ids']==oid),pose))
                    if t==22:break
                    act=plan.pop(0);prev=ag.state.position.copy();hit=ag.act(ACTION[act])
                    if act==1 and (hit or np.linalg.norm((ag.state.position-prev)[[0,2]])<.23):
                        plan=[sign,1,1,other,1,1]+plan;sign,other=other,sign
                rgb,depth,mask,pose=map(np.asarray,zip(*frames))
                temp=dest.with_suffix('.partial.npz');np.savez_compressed(temp,rgb=rgb,depth=depth,pose=pose,mask=mask,box=box);temp.replace(dest)
        finally:sim.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('i',type=int);run(ap.parse_args().i)
