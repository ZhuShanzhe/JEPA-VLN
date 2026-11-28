"""Stress test frozen verifier at physical views of other instances of the same category."""
from collections import Counter
import json
import math
import sys
from pathlib import Path
import numpy as np
from frame_confirm import FrameConfirmer,HERE
OLD=HERE.parent/"new_view_20261129"
sys.path.insert(0,str(OLD))
from sample_views import camera,simulator,inventory,ASSETS,SOURCE_H

def main():
    houses=json.loads((OLD/"view_design.json").read_text())["cohorts"]["blind"]
    summary=Counter()
    for house in houses:
        path=OLD/"private_render_blind60"/("blind_"+house)
        with np.load(path/"controls.npz") as f:
            source,category=f["source_index"],f["category"]
        sim=simulator(house)
        try:
            inventory(sim,ASSETS)
            bank=[]
            for i in np.unique(source):
                with np.load(path/(str(i)+"_sensor.npz")) as d:
                    pos,theta,oid,box=d["position"],float(d["yaw"]),int(d["oid"]),d["box"]
                obs=camera(sim,pos,theta,SOURCE_H,0.)
                bank.append(dict(pos=pos,theta=theta,oid=oid,box=box,
                                 cat=str(category[np.flatnonzero(source==i)[0]]),
                                 rgb=np.copy(obs["rgb"][...,:3]),d=obs["depth"].squeeze().copy(),
                                 ids=obs["ids"].copy()))
            counts=Counter()
            for a in bank:
                track=FrameConfirmer(a["rgb"],a["d"],a["box"])
                for b in bank:
                    if a["oid"]==b["oid"] or a["cat"]!=b["cat"]:continue
                    delta=b["pos"]-a["pos"]
                    distance=np.linalg.norm(delta[[0,2]])
                    if abs(delta[1])>.12 or distance>8.:continue
                    other=b["ids"]==b["oid"];same=b["ids"]==a["oid"]
                    if not (other.sum()>=369 and same.sum()<369):continue
                    counts["frames"]+=1
                    counts["within_three_m"]+=int(distance<=3)
                    counts["true_source_instance_absent"]+=int(not same.any())
                    sn,cs=np.sin(a["theta"]),np.cos(a["theta"])
                    angle=b["theta"]-a["theta"]
                    pose=(delta[0]*cs-delta[2]*sn,-delta[0]*sn-delta[2]*cs,math.atan2(math.sin(angle),math.cos(angle)))
                    result=track.observe(b["rgb"],b["d"],pose)
                    counts["predicted_box"]+=int(result["candidate"])
                    counts["false_confirmations"]+=int(result["accept"])
                    if result["accept"]:
                        l,t,r,bot=result["box"]
                        counts["false_box_over_distractor"]+=int(other[t:bot,l:r].sum()>.5*other.sum())
            summary.update(counts)
        finally:sim.close()
    report=dict(houses=len(houses),counts=dict(summary),
                protocol="All ordered pairs of distinct same-category source instances among sampled held-out cameras, within 8m and same floor. Destination sees >=369 pixels of distractor and <369 pixels of original target. Camera geometry and appearance only at inference. Separate camera-pair stress test; not part of navigation episodes.")
    (HERE/"distractor_frames.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)

if __name__=="__main__":main()
