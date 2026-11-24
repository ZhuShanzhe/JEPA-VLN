"""Frozen model inference from RGB, causal odometry and executed actions; no truth files are read."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project",type=Path)
    ap.add_argument("folder",type=Path)
    ap.add_argument("--split",choices=("train","test"),default="test")
    ap.add_argument("--expected",type=int)
    args = ap.parse_args()
    root, d = args.project.resolve(), args.folder.resolve()
    base = root/"jepa_five_20261123"
    mem = root/"jepa_memory_v10_20261124/models"
    sys.path[:0] = list(map(str,(mem,base/"report_v8",base/"report_v7",base/"clip_v5",
                                root/"mapping_runtime/pilots/jepa_round2_five_20261121")))
    from cache_observed import assets
    from fit_v8 import restore_heads
    from route_heads import load_heads
    from grounded_route import evidence_fields,jepa_fields
    from report_head import forecast_inputs
    from v10_live import RouteMemory

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    enc,_ = assets()
    ground,nets,_ = restore_heads(base/"report_v8/v8.pt")
    route = load_heads(base/"clip_v5/head_v5.pt",base/"sparse_v6/sparse_head.pt")
    queries = ["cup","sofa"]
    texts = enc.encoder.texts(queries)
    text = torch.stack([torch.from_numpy(texts[q]) for q in queries])
    actor_paths = sorted((d/"streams"/args.split).glob("*.actor.json"))
    if len(actor_paths) != (args.expected if args.expected is not None else (24 if args.split=="test" else 48)):
        raise ValueError(f"Incomplete split: {len(actor_paths)}")
    sessions = {p.name.split(".actor")[0]: json.loads(p.read_bytes()) for p in actor_paths}
    # Key-frame cadence is fixed by action step, independent of trials, query, score or outcome.
    selected,decision_rgbs = set(),set()
    for actor in sessions.values():
        check = {e["t"] for e in actor["checkpoints"]}
        for t,act,xy,heading,hit,key in actor["history"]:
            if t%4==0 or t in check:
                selected.add(key)
            if t in check:
                decision_rgbs.add(key)
    read_rgb = lambda k: np.array(Image.open(d/"streams/rgb"/(k+".png")).convert("RGB"))
    visual,candidate = {},{}
    route_actions = sessions[next(iter(sessions))]["actions"]
    acts = torch.tensor(route_actions,dtype=torch.long)
    with torch.inference_mode():
        ordered = sorted(selected)
        for start in range(0,len(ordered),12):
            group = ordered[start:start+12]
            local=enc.local([read_rgb(k) for k in group])
            code,anc,sims = enc.compressed(local,text)
            v,n=ground(code.half().float(),anc.half().float(),sims.half().float(),text)
            for i,key in enumerate(group):
                visual[key]=v[i].amax(0).numpy().tolist()
                if key not in decision_rgbs:
                    continue
                ctx=F.normalize(torch.tensor(enc.encoder.images([read_rgb(key)])).float(),dim=-1)
                hd=torch.zeros(1)
                p,near,_=forecast_inputs(route,"sparse",ctx,hd,acts)
                dense,geom=evidence_fields(v[i:i+1],n[i:i+1],hd,acts)
                latent=jepa_fields(p,near,text,enc.anchors)
                candidate[key]=[nets[k](dense,latent,geom)[0].sigmoid().numpy().T.tolist()
                                for k in ("joint","observed","geometry")]
            if start%96==0:
                print("encoded",min(start+12,len(ordered)),len(ordered),flush=True)

    (d/(args.split+".camera.json")).write_text(json.dumps(visual,allow_nan=False))
    results={}
    for sid,actor in sessions.items():
        assert actor["actions"]==route_actions
        checkpoints=[x["t"] for x in actor["checkpoints"]]
        rows=[[] for _ in queries]
        for qi in range(2):
            runtime=RouteMemory(mem/"v10.json")
            peak={}
            for t,act,xy,heading,hit,key in actor["history"]:
                score=visual[key][qi] if t%4==0 or t in checkpoints else None
                runtime.observe(xy,t,heading,score,hit)
                if score is not None:
                    kk=runtime._key(np.array(xy),np.array(heading))
                    if kk not in peak or peak[kk][3]<score:
                        peak[kk]=copy.deepcopy(runtime.history[kk])
                if t not in checkpoints:
                    continue
                inputs=[v[qi] for v in candidate[key]]
                by_arm = {arm:runtime.rank(route_actions,xy,heading,t,*inputs,arm=arm)["scores"]
                          for arm in ("fixed","geometry","observed","full","v8")}
                features=runtime.features(route_actions,xy,heading,t)
                by_arm["exploration_only"]=features[:,0].tolist()
                # Preserve other components. The target trace alone remembers the maximum
                # ever observed per cell/bearing, forever, suppressing replacement and decay.
                persistent=RouteMemory(mem/"v10.json")
                persistent.history=copy.deepcopy(peak)
                for record in persistent.history.values():
                    record[5]=t
                older=persistent.features(route_actions,xy,heading,t)[:,2]
                by_arm["persistent_trace"]=(np.array(by_arm["full"])+runtime.weights["target_trace"]*(older-features[:,2])).tolist()
                for _ in range(checkpoints.count(t)):
                    rows[qi].append(dict(step=t,scores=by_arm))
        results[sid]=rows
    out=d/(args.split+".predictions.json")
    blob=json.dumps(dict(weights=sha(mem/"v10.json"),v8=sha(base/"report_v8/v8.pt"),
                         memory_source=sha(mem/"v10_live.py"),stride=4,visual_keys=len(selected),
                         episodes=results),allow_nan=False)
    if out.exists() and out.read_text()!=blob:
        raise RuntimeError("Output already exists with different values")
    out.write_text(blob)
    print("scored",len(results),len(ordered),flush=True)


if __name__=="__main__":
    main()
