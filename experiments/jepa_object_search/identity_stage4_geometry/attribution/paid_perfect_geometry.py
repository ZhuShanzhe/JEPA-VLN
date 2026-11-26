"""Oracle ablation with true per-camera depth/pose. Semantic scoring runs only after predictions are written."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image


P=Path(os.environ["NAV_ROOT"])
BASE=P/"identity_stage4_geometry/attribution"; PRIVATE=BASE/"private"
ARMS=("keyframes_only","anchored_pnp")


def load(path):
    return json.loads(path.read_bytes())


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def capture():
    import quaternion
    from evaluation.main import run_eval
    from evaluation.main.evaluator import _env_action,_reset_environment
    from habitat.config import read_write
    from habitat.config.default_structured_configs import HabitatSimDepthSensorConfig
    from omegaconf import OmegaConf
    from unittest.mock import patch
    from verify_habitat import scene,state
    from paid_pair_labels import episode
    for arm in ARMS:
        doc,location=episode(arm)
        ep,=doc["episodes"]
        trace=[json.loads(l) for l in (location/"agent/action_trace.jsonl").read_bytes().splitlines()]
        src_t={r["mapping_frame_id"]:r["step"] for r in trace if (r.get("mapping") or {}).get("is_keyframe")}
        folder=PRIVATE/"oracle_geometry"/arm; folder.mkdir(parents=True,exist_ok=True,mode=0o700)
        pairs=sorted((location/"private_stage4_pairs").glob("*/complete.json"))
        requests=[]
        for f in pairs:
            c=load(f)
            requests.append(dict(pair=str(f.parent),step=c["step"],source=c["source"],source_step=src_t[c["source"]]))
        selected={r[k] for r in requests for k in ("step","source_step")}
        make=run_eval.Env
        calibration={}
        def augment(cfg):
            sim=cfg.habitat.simulator
            rgb=sim.agents.main_agent.sim_sensors.rgb
            d=OmegaConf.structured(HabitatSimDepthSensorConfig)
            for field in ("hfov","position","orientation","width","height"): d[field]=rgb[field]
            d.normalize_depth=False; d.min_depth=0.; d.max_depth=100.
            focal=float(rgb.width)/(2*np.tan(np.deg2rad(float(rgb.hfov))/2.))
            calibration["K"]=np.array([[focal,0,float(rgb.width)/2],[0,focal,float(rgb.height)/2],[0,0,1.]])
            with read_write(sim): sim.agents.main_agent.sim_sensors["depth"]=d
            return make(cfg)
        with patch.object(run_eval,"Env",augment): env,item,data=scene(doc)
        try:
            frame=_reset_environment(env,item,data)
            for t,row in enumerate(ep["paper"]["steps"]):
                if t in selected:
                    assert np.linalg.norm(state(env)[0]-row["position_before"])<.002
                    s=env.sim.get_agent_state().sensor_states["rgb"]
                    pose=np.eye(4); pose[:3,3]=s.position
                    pose[:3,:3]=quaternion.as_rotation_matrix(s.rotation)@np.diag([1.,-1.,-1.])
                    d=np.asarray(frame["depth"]).squeeze().copy()
                    assert d.shape==(1280,1280) and np.isfinite(d).all() and d.max()>1.
                    d[(d<=.01)|(d>=99.9)]=np.nan
                    np.savez_compressed(folder/f"geometry_{t}.npz",depth=d,K=calibration["K"],pose=pose)
                if t==max(selected): break
                if row["action_role"]=="navigate":frame=env.step(_env_action(row["action"]))
        finally:
            env.close()
        write(folder/"requests.json",requests)
        print(arm,"coplanar depth captures",len(selected),flush=True)


def predict():
    import cv2
    sys.path.insert(0,str(P/"identity_stage3_model/train"))
    from identity import IdentityVerifier
    model=IdentityVerifier(P,"jepa","cpu")
    result=[]
    for arm in ARMS:
        folder=PRIVATE/"oracle_geometry"/arm
        requests=load(folder/"requests.json")
        source_paths={}
        for r in requests:
            src=Path(r["pair"])/f"anchor_{r['source']}.png"
            if src.exists():
                if r["source"] in source_paths:
                    assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(source_paths[r["source"]].read_bytes()).digest()
                source_paths[r["source"]]=src
        memo={}
        def encode(rgbfile,t,boxes):
            im=np.asarray(Image.open(rgbfile).convert("RGB"))
            h,w=im.shape[:2]
            norm=np.asarray(boxes).reshape(-1,4)
            key=(t,h,w,tuple(norm.ravel()))
            if key not in memo:
                with np.load(folder/f"geometry_{t}.npz") as g:
                    d=g["depth"]; K=g["K"].copy(); pose=g["pose"]
                if (h,w)!=d.shape:
                    # Mapper source RGBs use a square downsample of the very same sensor exposure.
                    assert d.shape[0]==d.shape[1] and h==w and h==518
                    K[:2]*=h/d.shape[0]
                    d=cv2.resize(d,(w,h),interpolation=cv2.INTER_NEAREST_EXACT)
                memo[key]=model.encode(im,norm*[w,h,w,h],d,K,pose)
            return memo[key]
        for r in requests:
            base=Path(r["pair"]); c=load(base/"complete.json")
            live_boxes=np.asarray(c["live_boxes_work_width"])/c["work_width"]
            source_rgb=source_paths[c["source"]]
            cur=encode(base/"live.png",c["step"],live_boxes)
            src=encode(source_rgb,r["source_step"],[c["source_normalized_roi"]])
            score=model.score(src,cur); decision,=model.decide(score)
            valid=bool(score.size and np.isfinite(score).all())
            result.append(dict(arm=arm,step=c["step"],source=c["source"],result=decision["state"] if valid else "withheld",
                               scored=valid,selected_box=decision["index"]))
    write(PRIVATE/"oracle_geometry/frozen_outputs.json",result)


def grade():
    from grade_paid_pairs import summarize,row_key
    predictions=load(PRIVATE/"oracle_geometry/frozen_outputs.json")
    summary={}
    for arm in ARMS:
        labels=load(PRIVATE/"evaluator_only"/arm/"request_labels.json")
        rows=[r for r in predictions if r["arm"]==arm]
        v=summarize(labels,rows)
        v["finite_frozen_scores"]=sum(r["scored"] for r in rows)
        index={row_key(r):r for r in labels}
        decisions=[]
        for r in rows:
            if r["result"]!="same": continue
            z=index[row_key(r)]
            b=z["live"][r["selected_box"]]
            src=z["click_ids"]
            known=len(set(src))==1 and src[0] in z["targets"]
            if known:
                same_pixels=b["task_pixels"].get(str(src[0]),0); fraction=same_pixels/b["area"]
                visible=np.load(PRIVATE/"evaluator_only"/arm/f"step_{r['step']}.npz")["semantic"]
                coverage=same_pixels/max(1,int(np.count_nonzero(visible==src[0])))
            else:
                fraction=coverage=None
            decisions.append(dict(step=r["step"],selected_box=r["selected_box"],reference_click_is_task=known,
                selected_box_same_instance_fraction=round(fraction,4) if known else None,
                fraction_of_visible_source_instance_in_selected=round(coverage,4) if known else None,
                registered_task_in_range=bool(known and src[0] in z["official_near"])))
        v["selected_region_overlap"]=decisions
        summary[arm]=v
    write(BASE/"paid_55_oracle_geometry.json",dict(scope="Scorer-only true depth and pose at every requested source/current camera, original RGBs, boxes, ordering and frozen JEPA.",arms=summary))
    for a,v in summary.items():print(a,v["finite_frozen_scores"],dict(v["counts"]),v["selected_region_overlap"],flush=True)


if __name__=="__main__":
    {"capture":capture,"predict":predict,"grade":grade}[sys.argv[1]]()
