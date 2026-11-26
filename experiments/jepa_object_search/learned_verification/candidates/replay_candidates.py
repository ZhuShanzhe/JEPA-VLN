"""Regenerate archived mapped views, evaluate local RGB proposals after inference."""
import hashlib
import json
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np


P=Path(os.environ["NAV_ROOT"])
sys.path[:0]=[str(P/"jepa_reid_v21/detector"),str(P/"jepa_exec_v14"),str(P/"jepa_completion_v17/physical")]
from verify_habitat import state, deny_network
from check_routes import semantic_scene
from evaluation.main.evaluator import _reset_environment,_env_action
from proposal_worker import ProposalWorker
from source_quality import coarse_category
sys.addaudithook(deny_network)


def load(p):
    return json.loads(p.read_bytes())


def capture(index,worker):
    pilot=P/"mapping_runtime/pilots/jepa_reid_v24_first2"
    row=load(pilot/"panel.json")["rows"][index]
    ticket=load(pilot/"comparison/accepted/second"/(row["episode_id"]+".json"))
    path=Path(ticket["result"])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==ticket["result_sha256"]
    doc=load(path)
    ep,=doc["episodes"]
    source={r["step"]:r["mapping_frame_id"] for r in
            map(json.loads,(path.parent/"agent/action_trace.jsonl").read_text().splitlines())
            if r.get("mapping_frame_id") and (r.get("mapping") or {}).get("is_keyframe")}
    decisions=list(map(json.loads,(path.parent/"agent/decision_trace.jsonl").read_text().splitlines()))
    reviews=[(r["step"],t["request"].get("frame_id")) for r in decisions for t in r.get("tools",[])
             if t["request"].get("name")=="propose_candidates"]
    cutoffs=[(r["step"],r["world_state"].get("current_frame_id")) for r in decisions if "world_state" in r]
    env,item,dat=semantic_scene(doc)
    output=[]
    caption=coarse_category(ep["goal_text"])
    try:
        frame=_reset_environment(env,item,dat)
        for i,act in enumerate(ep["paper"]["steps"]):
            if i in source:
                assert np.linalg.norm(state(env)[0]-act["position_before"])<.002
                rgb=np.asarray(Image.fromarray(np.asarray(frame["rgb"])[...,:3]).resize((518,518),Image.Resampling.BICUBIC))
                proposals=worker.detect(rgb,caption)
                # The scorer accesses the semantic observations only after the detector call.
                mask=np.asarray(frame["semantic"]).squeeze()
                h,w=mask.shape; truth=[]
                for tid in item.target_object_ids:
                    yy,xx=np.nonzero(mask==int(tid))
                    if len(xx)>=103:
                        truth.append([float(xx.min()/w*518),float(yy.min()/h*518),
                                      float((xx.max()+1)/w*518),float((yy.max()+1)/h*518)])
                output.append(dict(step=i,fid=source[i],boxes=proposals,truth=truth))
            if act["action_role"]=="navigate":
                frame=env.step(_env_action(act["action"]))
    finally:
        env.close()
    return dict(frames=output,requests=reviews,decisions=cutoffs)


if __name__=="__main__":
    dst=Path(__file__).resolve().parent/"private_replay"
    dst.mkdir(mode=0o700,exist_ok=True)
    detector=ProposalWorker(P/"b_verification/env/bin/python",P,"cuda")
    try:
        for i,name in enumerate(("clock","painting")):
            f=dst/(name+".json")
            if not f.exists():
                data=capture(i,detector)
                f.write_text(json.dumps(data,allow_nan=False))
                print(name,len(data["frames"]),flush=True)
    finally:
        detector.close()
