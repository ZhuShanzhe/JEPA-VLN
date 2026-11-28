"""Visibility of the lower-body volume in real onboard depth before each grid-recovery forward step."""
from collections import Counter
import json,math,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'new_view_20261129'
sys.path.insert(0,str(BASE))
from sample_views import ASSETS,ACTION,SOURCE_H,camera,simulator,inventory
def support(history,xy,yaw):
    cs,sn=math.cos(yaw),math.sin(yaw)
    # Sample hypothetical lower body at the next 25 cm forward action, including robot width.
    samples=np.array([(xy[0]+x*cs-z*sn,xy[1]+x*sn+z*cs,h) for z in (.125,.25,.35) for x in (-.15,0,.15) for h in (.15,.4)])
    seen=np.zeros(len(samples),bool);blocked=np.zeros(len(samples),bool)
    for p,a,depth in history:
        dx=samples[:,0]-p[0];dz=samples[:,1]-p[1];z=-dx*math.sin(a)+dz*math.cos(a)
        right=dx*math.cos(a)+dz*math.sin(a)
        u=np.floor(96+96*right/np.maximum(.001,z)).astype(int)
        v=np.floor(96+96*(SOURCE_H-samples[:,2])/np.maximum(.001,z)).astype(int)
        inside=(z>.05)&(u>=0)&(v>=0)&(u<192)&(v<192)
        if not np.any(inside):continue
        ids=np.flatnonzero(inside);d=depth[v[ids],u[ids]]
        good=np.isfinite(d)&(d>.05)
        seen[ids[good]]=True
        # A discontinuity in front of a body sample is an occluder, not a confirmed collision.
        blocked[ids[good]]|=d[good]<z[ids[good]]-.10
    return float(seen.mean()),float(blocked.mean())
def run(house):
    path=HERE/'private_motion'/house;src=BASE/'private_render_blind60'/('blind_'+house)
    sim=simulator(house);ag=sim.get_agent(0);rows=[]
    try:
        inventory(sim,ASSETS)
        for file in sorted(path.glob('*.json')):
            r=json.loads(file.read_text())
            with np.load(src/(file.stem+'_sensor.npz')) as d:
                p,angle=d['position'],float(d['yaw'])
            before=camera(sim,p,angle,SOURCE_H,0.);history=[]
            for j,e in enumerate(r['events']):
                if j:
                    old=r['events'][j-1]
                    if e['action']==1 and j>r['baseline_length']:
                        seen,occ=support(history,np.array(old['pose'][:2]),old['pose'][2])
                        rows.append((bool(e['bump']),seen,occ))
                    ag.act(ACTION[e['action']]);before=sim.get_sensor_observations()
                history.append((e['pose'][:2],e['pose'][2],before['depth'].squeeze().copy()))
    finally:sim.close()
    return rows
if __name__=='__main__':
    rooms=json.loads((BASE/'view_design.json').read_text())['cohorts']['blind']
    arr=np.array([e for h in rooms for e in run(h)])
    out={}
    for hit,k in ((1,'contact'),(0,'successful_forward')):
        d=arr[arr[:,0]==hit]
        out[k]=dict(actions=len(d),none_in_any_prior_camera=int((d[:,1]==0).sum()),at_least_half_in_prior_fov=int((d[:,1]>=.5).sum()),
                    median_lower_body_samples_in_prior_fov=float(np.median(d[:,1])),some_occluder_before_sample=int((d[:,2]>0).sum()))
    out['interpretation']='18 sample points at 0.15 m and 0.4 m height, across the swept body. Count a point as within view if an actual previous camera ray covers it and has finite depth. An intervening surface can occlude a point. No inference of causal collision geometry.'
    (HERE/'low_body_support.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
