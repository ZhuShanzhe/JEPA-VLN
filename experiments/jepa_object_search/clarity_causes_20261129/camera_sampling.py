"""Isolate RGB raster sampling at the exact 14 score-limited frames; keep depth and verifier fixed."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT/"new_view_20261129"
OLD=ROOT/"reactive_view_20261129"
FIT=ROOT/"refined_confirmation_20261129"
sys.path[:0]=[str(BASE),str(OLD),str(FIT)]


def render(h):
    from sample_views import ACTION,ASSETS,SOURCE_H,camera,inventory,pose_data,simulator
    pose_data.S=384
    sim=simulator(h)
    try:
        inventory(sim,ASSETS)
        agg=json.loads((HERE/"private_camera"/(h+".json")).read_text())["records"]
        files=[p for p in sorted((OLD/"private_roll"/h).glob("*.json")) if json.loads(p.read_text())["focus"]]
        assert len(agg)==len(files)
        dest=HERE/"private_sampling"/h;dest.mkdir(mode=0o700,exist_ok=True,parents=True)
        for d,p in zip(agg,files):
            if not d["fail14"]:continue
            t=d["best_good"]
            with np.load(BASE/"private_render_blind60"/("blind_"+h)/(p.stem+"_sensor.npz")) as f:
                obs=camera(sim,f["position"],float(f["yaw"]),SOURCE_H,0.)
                oid=int(f["oid"])
            row=json.loads(p.read_text())
            for step in row["events"][1:t+1]:
                sim.get_agent(0).act(ACTION[step["action"]])
            if t:obs=sim.get_sensor_observations()
            assert int((obs["ids"]==oid).sum())==d["hi"][t]
            color=obs["rgb"][...,:3]
            down=np.rint(color.reshape(192,2,192,2,3).mean(axis=(1,3))).astype(np.uint8)
            np.savez_compressed(dest/(p.stem+".npz"),t=t,rgb=down)
    finally:sim.close()


def score():
    from frame_confirm import FrameConfirmer,thresholds
    from clarity_refit import ClarityEstimator
    from memory_refit import relative
    model=ClarityEstimator.load()
    cutoff=thresholds()["threshold"]
    old=[];new=[];joint=[];pixels=[]
    for f in sorted((HERE/"private_sampling").glob("*/*.npz")):
        h=f.parent.name;k=f.stem
        with np.load(f) as q:t=int(q["t"]);rgb_new=q["rgb"]
        with np.load(FIT/"private_inputs"/h/f.name) as z:
            ver=FrameConfirmer(z["rgb"][0],z["depth"][0],z["cue"])
            depth=z["depth"][t];rgb=z["rgb"][t];pose=relative(z["odom"][t],(0,0,0))
        logged=json.loads((FIT/"private_paired/full"/h/k/"decision.json").read_text())["trace"][t]
        assert logged["state"]=="trusted_initial"
        static=ver.observe(rgb,depth,pose);box=static["box"];same=static["p_same"]
        assert list(box)==logged["box"] and abs(same-logged["p_same"])<1e-6
        c=model.predict(rgb,depth,box,ver,pose,static)
        assert abs(c-logged["p_clear"])<1e-6
        nc=model.predict(rgb_new,depth,box,ver,pose,static)
        old.append(c);new.append(nc);joint.append(nc*same)
        pixels.append(float(np.abs(rgb.astype(np.float32)-rgb_new).mean()))
    assert len(old)==14
    delta=np.asarray(new)-old
    report=dict(groups=14,
        protocol="Diagnostic intervention on current RGB only at the previously selected best frame in each of fourteen requests. Native 384x384 rendered RGB area-averaged to 192x192. Hold native source memory, box, depth, associations, pose, gate and nominally previous projected-surface features fixed. No 384-native classifier or full-policy retest.",
        cutoff=cutoff,median_p_clear_baseline=float(np.median(old)),median_p_clear_supersampled=float(np.median(new)),
        median_paired_effect=float(np.median(delta)),min_paired_effect=float(delta.min()),max_paired_effect=float(delta.max()),
        median_RGB_absolute_change_8bit=float(np.median(pixels)),admitted_native=0,
        admitted_with_supersampled_current_rgb=int(sum(np.asarray(joint)>=cutoff)))
    (HERE/"camera_sampling.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)


if __name__=="__main__":
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument("--house-index",type=int);a=p.parse_args()
    if a.house_index is None:score()
    else:render(json.loads((BASE/"view_design.json").read_text())["cohorts"]["blind"][a.house_index])
