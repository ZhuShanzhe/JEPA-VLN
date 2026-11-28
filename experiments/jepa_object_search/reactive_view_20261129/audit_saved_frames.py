"""Independently replay selected motions, verify saved images and grade the first submitted evidence."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/"new_view_20261129"
sys.path.insert(0,str(BASE))
from sample_views import ACTION,ASSETS,SOURCE_H,camera,inventory,simulator


def audit(house):
    folder=BASE/"private_render_blind60"/("blind_"+house)
    records=HERE/"private_roll"/house
    counts=Counter()
    errors=Counter()
    error_focus=Counter()
    probe=json.loads((HERE/"private_stop_probes"/(house+".json")).read_text())
    probe_by_source={r["source"]:r for r in probe}
    cases=[]
    sim=simulator(house)
    try:
        inventory(sim,ASSETS)
        agent=sim.get_agent(0)
        for source in range(60):
            d=json.loads((records/f"{source}.json").read_text())
            with np.load(folder/f"{source}_sensor.npz") as s:
                start,yaw,oid=s["position"],float(s["yaw"]),int(s["oid"])
            obs=camera(sim,start,yaw,SOURCE_H,0.)
            events=d["events"]; base=d["baseline_length"]
            assert len(events)-1<=d["budget"] and 2<=base<len(events)
            first=next((j for j,e in enumerate(events) if e["accept"]),None)
            assert first==d["new_confirmation"]
            assert (first if first is not None and first<=base else None)==d["old_confirmation"]
            validated=False;error=None
            for t,e in enumerate(events):
                assert t==e["t"]
                if t:
                    a=e["action"];prior=agent.state.position.copy()
                    contact=agent.act(ACTION[a])
                    delta=np.linalg.norm((agent.state.position-prior)[[0,2]])
                    stall=a==1 and (contact or delta<.23)
                    assert bool(stall)==e["bump"]
                    obs=sim.get_sensor_observations()
                else:
                    assert e["action"]==0 and not e["bump"]
                actual=agent.state.position-start
                coords=[np.cos(yaw)*actual[0]-np.sin(yaw)*actual[2],
                        -np.sin(yaw)*actual[0]-np.cos(yaw)*actual[2]]
                assert np.allclose(coords,e["pose"][:2],atol=1e-4)
                mask=obs["ids"]==oid
                pixels=int(mask.sum())
                assert pixels==e["px"]
                if t!=first:continue
                rgb=obs["rgb"][...,:3]; l,b,r,f=map(int,e["box"])
                assert 0<=l<r<=rgb.shape[1] and 0<=b<f<=rgb.shape[0]
                with Image.open(records/f"{source}_step{t}.png") as im: assert np.array_equal(np.asarray(im),rgb)
                with Image.open(records/f"{source}_step{t}_crop.png") as im: assert np.array_equal(np.asarray(im),rgb[b:f,l:r])
                inter=int(mask[b:f,l:r].sum())
                yy,xx=np.where(mask)
                iou=0.
                if len(xx):
                    xlo,xhi=int(xx.min()),int(xx.max()+1); ylo,yhi=int(yy.min()),int(yy.max()+1)
                    overlap=max(0,min(r,xhi)-max(l,xlo))*max(0,min(f,yhi)-max(b,ylo))
                    union=(r-l)*(f-b)+(xhi-xlo)*(yhi-ylo)-overlap
                    iou=overlap/union
                assert e["intersection"]==inter and abs(iou-e["gt_box_iou"])<1e-6
                validated=bool(pixels>=369 and inter>=.5*pixels and iou>=.30)
                assert validated==e["correct"]
                counts["full_and_crop_pairs_verified"]+=1
                if not validated:
                    error=("target_absent_inside_submitted_box" if inter==0 else
                           "target_present_but_below_clarity" if pixels<369 else
                           "target_clear_but_insufficient_localisation")
                    errors[error]+=1
                    if d["focus"]:error_focus[error]+=1
            counts["groups"]+=1
            counts["post_source_physical_actions_verified"]+=len(events)-1
            if d["focus"]:
                prior_seen=any(e["px"]>=369 for e in events[:base+1])
                full_seen=any(e["px"]>=369 for e in events)
                q=probe_by_source[source]
                assert q["prior_ever_clear"]==prior_seen
                if first is None:
                    failure="clear_seen_but_no_frame_submitted" if full_seen else "no_clear_frame_on_executed_path"
                elif validated:failure="valid_saved_confirmation"
                else:failure=error
                cases.append(dict(prior_seen=prior_seen,full_seen=full_seen,success=validated,
                                  alternative_probe=q["probe_opportunity"],failure=failure))
    finally:
        sim.close()
    return counts,errors,error_focus,cases


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--house-index",type=int);args=ap.parse_args()
    houses=json.loads((BASE/"view_design.json").read_text())["cohorts"]["blind"]
    tmp=HERE/"private_verification";tmp.mkdir(mode=0o700,exist_ok=True)
    if args.house_index is not None:
        h=houses[args.house_index]; result=audit(h)
        (tmp/(h+".json")).write_text(json.dumps(result))
        return
    counts=Counter(); errors=Counter(); error_focus=Counter(); cases=[]
    for h in houses:
        c,e,f,r=json.loads((tmp/(h+".json")).read_text())
        counts.update(c); errors.update(e); error_focus.update(f); cases.extend(r)
    assert counts["groups"]==180 and len(cases)==94
    available=[r for r in cases if r["alternative_probe"]]
    result=dict(checks=counts,first_submit_errors=errors,focus_submit_errors=error_focus,
                focus=Counter(r["failure"] for r in cases),offline_probe_subset=dict(
                prior_failed_and_probe_found_an_opportunity=len(available),
                new_route_saw_clear=sum(r["full_seen"] for r in available),
                new_correctly_confirmed=sum(r["success"] for r in available)),
                notes="First submission only. Original and submitted-crop PNGs exactly match the independently re-rendered target step. Geometry-based box recall >=50%, bounding rectangle IoU >=0.30 and at least 369 target pixels. Private images never leave the experiment.")
    (HERE/"saved_frame_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__=="__main__":main()
