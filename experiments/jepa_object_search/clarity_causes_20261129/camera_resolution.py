"""Repeat the exact frozen poses at 384x384; score angular coverage in both rasterizations."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT/"new_view_20261129"
PREV=ROOT/"reactive_view_20261129"
FIT=ROOT/"refined_confirmation_20261129"
sys.path.insert(0,str(BASE))


def house_job(house):
    from sample_views import ACTION,ASSETS,SOURCE_H,camera,inventory,pose_data,simulator
    pose_data.S=384
    sim=simulator(house)
    records=[]
    try:
        inventory(sim,ASSETS)
        agent=sim.get_agent(0)
        spec=agent.agent_config.sensor_specifications[0]
        sensor={k:str(getattr(spec,k,None)) for k in ("sensor_subtype","noise_model","hfov","near","far")}
        path=BASE/"private_render_blind60"/("blind_"+house)
        for p in sorted((PREV/"private_roll"/house).glob("*.json")):
            prior=json.loads(p.read_text())
            if not prior["focus"]:continue
            chosen=json.loads((FIT/"private_paired/full"/house/p.stem/"decision.json").read_text())
            trace=chosen["trace"]
            with np.load(path/(p.stem+"_sensor.npz")) as f:
                start,yaw,oid=f["position"],float(f["yaw"]),int(f["oid"])
            with np.load(FIT/"private_scoring"/house/(p.stem+".npz")) as f:
                low=np.unpackbits(f["packed"],axis=1).reshape(-1,192,192)
            obs=camera(sim,start,yaw,SOURCE_H,0.)
            low_px=[];high_px=[];ious=[]
            for t,event in enumerate(prior["events"]):
                if t:
                    before=agent.state.position.copy()
                    a=event["action"]
                    hit=agent.act(ACTION[a])
                    move=np.linalg.norm((agent.state.position-before)[[0,2]])
                    assert event["bump"]==bool(a==1 and (hit or move<.23))
                    obs=sim.get_sensor_observations()
                diff=agent.state.position-start
                rel=[np.cos(yaw)*diff[0]-np.sin(yaw)*diff[2],-np.sin(yaw)*diff[0]-np.cos(yaw)*diff[2]]
                assert np.allclose(rel,event["pose"][:2],atol=1e-5)
                mask=obs["ids"]==oid
                assert mask.shape==(384,384) and int(low[t].sum())==event["px"]
                low_px.append(event["px"]);high_px.append(int(mask.sum()))
                # Area-weighted intersection/union at the original pixel grid, for pure rendering alignment.
                tile=mask.reshape(192,2,192,2).sum(axis=(1,3))/4.
                intersect=np.minimum(tile,low[t]).sum()
                union=np.maximum(tile,low[t]).sum()
                ious.append(intersect/union if union>0 else None)
            old=prior["new_confirmation"];prev_frames=prior["events"]
            old_never=old is None and max(low_px)<369
            fail14=False;best_good=None
            if old is None and max(low_px)>=369 and chosen["step"] is None:
                matches=[]
                for i,state in enumerate(trace):
                    if not state["box"] or low_px[i]<369:continue
                    l,t,r,b=state["box"];mask=low[i]
                    u,v=np.where(mask)
                    ov=max(0,min(r,v.max()+1)-max(l,v.min()))*max(0,min(b,u.max()+1)-max(t,u.min()))
                    denom=(r-l)*(b-t)+(v.max()+1-v.min())*(u.max()+1-u.min())-ov
                    if ov/denom>=.3 and mask[t:b,l:r].sum()>=mask.sum()*.5:
                        matches.append(i)
                if matches:
                    best_good=max(matches,key=lambda i:trace[i]["p_clear"]*trace[i]["p_same"])
                    g=trace[best_good];fail14=g["p_clear"]<.835 and g["p_same"]>=.835
            records.append(dict(lo=low_px,hi=high_px,iou=ious,old_never=old_never,until=len(trace),
                                fail14=bool(fail14),best_good=best_good))
    finally:sim.close()
    out=HERE/"private_camera";out.mkdir(mode=0o700,exist_ok=True)
    (out/(house+".json")).write_text(json.dumps(dict(sensor=sensor,records=records)))


def summarize(r,mode):
    if mode=="peak_actual": ix=[(i,int(np.argmax(g["lo"][:g["until"]]))) for i,g in enumerate(r)]
    if mode=="peak_all": ix=[(i,int(np.argmax(g["lo"]))) for i,g in enumerate(r)]
    if mode=="best_good":ix=[(i,g["best_good"]) for i,g in enumerate(r)]
    if mode=="actual":ix=[(i,j) for i,g in enumerate(r) for j in range(g["until"])]
    lo=np.asarray([r[i]["lo"][j] for i,j in ix]);hi=np.asarray([r[i]["hi"][j] for i,j in ix])
    iou=[r[i]["iou"][j] for i,j in ix if r[i]["iou"][j] is not None and r[i]["lo"][j]>=30]
    gt=lo>0
    return dict(observations=len(lo),native_clear=int(sum(lo>=369)),
        double_resolution_clear_same_fraction=int(sum(hi>=1476)),
        double_resolution_clear_one_percent=int(sum(hi>=1475)),
        double_resolution_clear_using_old_raw_count=int(sum(hi>=369)),
        crossed_up=int(sum((hi>=1476)&(lo<369))),crossed_down=int(sum((hi<1476)&(lo>=369))),
        median_native_px=float(np.median(lo)),median_double_resolution_px=float(np.median(hi)),
        median_area_ratio_when_visible=float(np.median(hi[gt]/lo[gt])),
        median_area_weighted_iou_when_at_least_30px=float(np.median(iou)))


def main():
    p=argparse.ArgumentParser();p.add_argument("--house-index",type=int);args=p.parse_args()
    os.umask(0o077)
    houses=json.loads((BASE/"view_design.json").read_text())["cohorts"]["blind"]
    if args.house_index is not None:
        house_job(houses[args.house_index]);return
    docs=[json.loads((HERE/"private_camera"/(h+".json")).read_text()) for h in houses]
    groups=[r for d in docs for r in d["records"]];never=[r for r in groups if r["old_never"]];low=[r for r in groups if r["fail14"]]
    assert (len(groups),len(never),len(low))==(94,33,14)
    assert all(d["sensor"]==docs[0]["sensor"] for d in docs)
    result=dict(protocol="Exact frozen motions replayed at 384x384 with no change to horizontal FOV, pitch, camera height or scene. Native segmentation previously recorded in the same poses; actual frames stop at the new first submission or natural termination. Normalized threshold 1476=4*369. Offline, no policy inference on larger frames.",
        sensor=docs[0]["sensor"],native_dimensions=[192,192],double_dimensions=[384,384],height_m=1.31,vertical_pitch_deg=0.,
        all_94_peak_actual_frame=summarize(groups,"peak_actual"),all_deployed_frames=summarize(groups,"actual"),
        previously_never_clear_33_peak_frame_full_original_path=summarize(never,"peak_all"),
        clarity_score_limited_14_best_valid_frame=summarize(low,"best_good"),
        previous_33_ever_crossed_equal_fraction=sum(max(g["hi"])>=1476 for g in never),
        previous_33_ever_crossed_unchanged_raw_cutoff=sum(max(g["hi"])>=369 for g in never))
    (HERE/"camera_resolution.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":main()
