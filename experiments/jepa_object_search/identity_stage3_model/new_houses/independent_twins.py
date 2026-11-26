"""Apply the previously published A/B object-removal protocol to the independent homes."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def collect(project,assets,folder):
    natural=rows(folder/"private/frames.jsonl")
    src=folder/"private/twin_sources"
    for module in ("model","collect"):
        link=src/module
        link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():
            link.symlink_to(project/"jepa_reid_v20"/module,target_is_directory=True)
    for name,cat in (("data","wall_art"),("clocks","clock")):
        (src/name).mkdir(exist_ok=True)
        seeds=[]
        for r in natural:
            if r["category"]!=cat:
                continue
            T=np.array(r["T"])
            dx,dy,dz=T[:3,2]
            seeds.append(dict(house=r["house"],category=cat,split="test",instance=r["uid"],image=r["name"],
                              camera_xyz=T[:3,3].tolist(),camera_heading_deg=math.degrees(math.atan2(dz,dx)),
                              pitch_deg=math.degrees(math.atan2(dy,math.hypot(dx,dz)))))
        (src/name/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in seeds))
    sys.path.insert(0,str(project/"jepa_reid_v21/data"))
    from controlled_twins import main
    main(assets,src,folder/"private/twins")


def score(project,folder,stage):
    sys.path[:0]=[str(project/n) for n in ("identity_stage3_model/eval","identity_stage3_model/train",
                                           "jepa_reid_v21/detector")]
    twin=folder/"private/twins"
    frames=rows(twin/"frames.jsonl")
    predfile=twin/"dino.jsonl"
    if stage=="detect":
        from proposal_worker import ProposalWorker
        worker=ProposalWorker(project/"b_verification/env/bin/python",project,device="cuda")
        try:
            with predfile.with_suffix(".tmp").open("w") as f:
                for fr in frames:
                    with Image.open(twin/fr["rgb"]) as im:
                        rgb=np.array(im.convert("RGB"))
                    b=worker.detect(rgb,fr["category"])
                    f.write(json.dumps(dict(name=fr["frame_id"],boxes=[x["bbox"] for x in b]))+"\n")
        finally:
            worker.close()
        predfile.with_suffix(".tmp").replace(predfile)
        return
    from compare import evaluate,summarize
    from identity import IdentityVerifier
    bylayout=defaultdict(list)
    pred={r["name"]:r["boxes"] for r in rows(predfile)}
    images={}
    for fr in frames:
        k=fr["frame_id"]
        bylayout[fr["layout"]].append(fr)
        images[k]=dict(rgb=twin/fr["rgb"],depth=twin/fr["depth"],K=fr["K"],c2w=fr["T_world_from_camera"],
                       boxes=pred[k],truth=fr["boxes"])
    cases=[]
    for group,seq in sorted(bylayout.items()):
        for src in seq:
            if src["stratum"]!="double" or len(src["boxes"])!=2: continue
            for live in seq:
                if live["stratum"]!="single" or len(live["boxes"])!=1: continue
                a=live["boxes"][0]["instance"]
                ids={v["instance"] for v in src["boxes"]}
                if "synthetic" in a or a not in ids: continue
                b,=ids-{a}
                assert "synthetic" in b
                cases.append(dict(panel="twins",source=src["frame_id"],live=live["frame_id"],uids=[a,b],visible=True,
                                  unit=group,house=src["house"],category=src["category"],
                                  view="same_camera" if src["view"]==live["view"] else "changed_camera"))
    checkdirs=[("126111",project/"identity_stage3_model/train/private")]
    for seed in (126112,126113):
        d=checkdirs[0][1]/("seed"+str(seed))
        if (d/"training_results.json").exists() or (d/"jepa.pt").exists():
            checkdirs.append((str(seed),d))
    cache={}
    method_rows={}
    class Position:
        def __init__(self,encoder): self.encode=encoder.encode
        def score(self,left,right):
            return np.array([[np.linalg.norm(np.array(a)-b) if a is not None and b is not None else np.inf
                              for b in right["points"]] for a in left["points"]]).reshape(len(left["points"]),len(right["points"]))
        def decide(self,matrix):
            return [dict(state="same",index=int(np.argmin(d))) if len(d) and np.min(d)<=.347302806383428
                    else dict(state="uncertain",index=None) for d in matrix]
    def stats(mode,verifier=None,seed=None):
        result=evaluate(project,images,cases,mode,"cuda",verifier,cache)
        method_rows[seed,mode]=result
        return {k:v for k,v in summarize(result).items() if k.startswith("twins")}
    report=dict(cases=len(cases),frames=len(frames),layouts=len(bylayout),
                panel_sha256=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
                legacy=stats("legacy"),paid_calls=0,learned={})
    for seed,ck in checkdirs:
        opts={} if seed==checkdirs[0][0] else {"checkpoint_dir":ck}
        dm=IdentityVerifier(project,"direct",**opts)
        jm=IdentityVerifier(project,"jepa",**opts)
        if seed==checkdirs[0][0]:
            report["position"]=stats("position",Position(dm))
        report["learned"][seed]={m:stats(m,v,seed) for m,v in (("direct",dm),("jepa",jm))}
    def event(row,key):
        a,b=row["pred"]
        if key=="joint":
            return int(a["outcome"]=="original" and b["state"]!="same")
        return int(b["outcome"]=="original")
    interval={}
    for key in ("joint","false_accept_copy"):
        by_house=defaultdict(list)
        for i,case in enumerate(cases):
            diff=[event(method_rows[s,"jepa"][i],key)-event(method_rows[s,"direct"][i],key) for s,_ in checkdirs]
            by_house[case["house"]].append(np.mean(diff))
        v=[np.mean(h) for h in by_house.values()]
        boot=np.random.default_rng(7129).choice(v,size=(10000,len(v))).mean(1)
        interval[key]=dict(mean=float(np.mean(v)),ci95=np.quantile(boot,[.025,.975]).tolist(),houses=len(v))
    report["seed_averaged_paired_house_delta_vs_direct"]=interval
    (folder/"independent_twins.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("twin probes",len(cases),"seeds",list(report["learned"]))


if __name__=="__main__":
    project,folder=map(Path,sys.argv[1:3])
    stage=sys.argv[3]
    if stage=="collect":
        collect(project,Path(sys.argv[4]),folder)
    else:
        assert stage in ("detect","score")
        score(project,folder,stage)
