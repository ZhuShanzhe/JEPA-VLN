"""Post-decision audit of changed viewpoints using physically re-rendered source and destination frames."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import os
import numpy as np

HERE=Path(__file__).resolve().parent


def capture():
    from sample_views import simulator, camera, inventory, ASSETS, SOURCE_H, ACTION
    rows=json.loads((HERE/"private_evaluation/blind_choices.json").read_text())
    report=defaultdict(Counter)
    for house in sorted({r["house"] for r in rows}):
        folder=HERE/"private_render_blind60"/("blind_"+house)
        out=HERE/"private_live_inputs"/house
        out.mkdir(parents=True,exist_ok=True)
        with np.load(folder/"controls.npz") as f:
            data={k:f[k] for k in ("action","group","kind","pixels","pose","start_pixels","scan_pixels","first_blocked","budget")}
        sim=simulator(house)
        try:
            inventory(sim,ASSETS)
            agent=sim.get_agent(0)
            for item in (x for x in rows if x["house"]==house):
                i=item["source"]
                with np.load(folder/f"{i}_sensor.npz") as f:
                    pos,yaw,oid,box=f["position"],float(f["yaw"]),int(f["oid"]),f["box"]
                camera(sim,pos,yaw,SOURCE_H,0.)
                agent.act(ACTION[2])
                left=sim.get_sensor_observations()["depth"].squeeze().copy()
                agent.act(ACTION[3])
                obs=sim.get_sensor_observations()
                depth=obs["depth"].squeeze().copy()
                np.savez_compressed(out/f"{i}.npz",rgb=obs["rgb"][...,:3],depth=depth,left_depth=left,box=box)
                j=item["jepa"]
                if j is None or data["kind"][j]!="alt":
                    continue
                category="blocked" if data["first_blocked"][j] else "clear" if data["pixels"][j]>=369 else "reached_unclear"
                count=report[category];count["groups"]+=1
                l,t,r,b=map(int,box); u,v=(l+r)/2,(t+b)/2
                ids=obs["ids"]==oid
                count["source_box_center_outside_instance"]+=int(not ids[int(v),int(u)])
                patch=depth[max(0,int(v)-2):int(v)+3,max(0,int(u)-2):int(u)+3]
                reference=depth[ids & np.isfinite(depth) & (depth>.05)]
                if len(reference) and np.any(np.isfinite(patch)&(patch>.05)):
                    z=float(np.median(patch[np.isfinite(patch)&(patch>.05)]))
                    count["source_anchor_differs_from_target_median_over_half_m"]+=int(abs(z-float(np.median(reference)))>.5)
                if category=="blocked":continue
                # Reproject visible source-instance surface samples into the actually reached camera.
                yy,xx=np.where(ids & np.isfinite(depth) & (depth>.05))
                jj=np.linspace(0,len(xx)-1,min(len(xx),128),dtype=int);yy,xx=yy[jj],xx[jj]
                zz=depth[yy,xx]; sx=(xx+.5-96)*zz/96; sy=(96-yy-.5)*zz/96
                for a in data["action"][j]:
                    if not a:break
                    agent.act(ACTION[int(a)])
                end=sim.get_sensor_observations()
                assert abs(int((end["ids"]==oid).sum())-data["pixels"][j])<=1
                x,z=data["pose"][j,:2]
                sine,cosine=data["pose"][j,4:6]
                dx,dz=sx-x,zz-z
                forward=-dx*sine+dz*cosine
                uu=np.rint(96+96*(dx*cosine+dz*sine)/np.maximum(.001,forward)-.5).astype(int)
                vv=np.rint(96-96*sy/np.maximum(.001,forward)-.5).astype(int)
                inside=(forward>.05)&(uu>=0)&(uu<192)&(vv>=0)&(vv<192)
                count["source_surface_majority_outside_end_frame"]+=int(inside.mean()<.5)
                if inside.any():
                    du=uu[inside];dv=vv[inside];ed=end["depth"].squeeze()[dv,du]
                    hit=(ed>.05)&np.isfinite(ed)&(ed<forward[inside]-.15)&(end["ids"][dv,du]!=oid)
                    count["source_surface_majority_blocked_by_another_surface"]+=int(inside.mean()>=.5 and hit.mean()>.5)
                count["final_250_to_368_pixels"]+=int(250<=data["pixels"][j]<369)
        finally:
            sim.close()
    (HERE/"visual_geometry_diagnosis.json").write_text(json.dumps(dict(
        scope="Post-hoc only, alternative views actually selected. Checks may overlap. Source-box depth is compared with visible pixels of the true instance. Occlusion compares sampled source-instance surface depths with actual destination depth and identity.",
        outcomes=report),indent=2)+"\n")

if __name__=="__main__":
    os.umask(0o077)
    capture()
