"""Paired action-conditioned future-image teachers from physically continuous HSSD trees."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/"jepa_env_v13/simulator"
HORIZONS=(12,24,52,80,108)
SCAN=(3,2,2,3)
POSE_NAMES=("forward_m","left_m","sin_yaw","cos_yaw","camera_height_m","sin_pitch","cos_pitch")


class FrozenImageEncoder:
    def __init__(self,project=ROOT):
        sys.path.insert(0,str(project/"jepa_five_20261123/clip_v5"))
        from clip_views import Clip, EXPECTED
        c=Clip()
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.net=c.net.visual.to(self.device).eval()
        self.prep=c.prep
        self.weights_sha=EXPECTED

    @torch.inference_mode()
    def encode(self,images):
        features=[]
        for s in range(0,len(images),64):
            batch=torch.stack([self.prep(Image.fromarray(frame)) for frame in images[s:s+64]])
            z=self.net(batch.to(self.device))
            features.append(F.normalize(z.float(),dim=-1).cpu().numpy().astype(np.float16))
        return np.concatenate(features)


def plan_pose(actions):
    x=y=yaw=0.
    for a in actions:
        if a==1:
            x+=.25*math.cos(yaw)
            y+=.25*math.sin(yaw)
        else:
            yaw+=math.pi/6 if a==2 else -math.pi/6
    return [x,y,math.sin(yaw),math.cos(yaw),1.25,0.,1.]


def fixed_child(origin,start,branch,depth):
    value=f"{origin}/{start}/{branch}/{depth}".encode()
    return int.from_bytes(sha256(value).digest()[:4],"big")%3


def one_tree(path,encoder):
    nodes={}
    for entry in sorted(path.glob("node_*.npz")):
        key=entry.stem[5:]
        with np.load(entry,allow_pickle=False) as z:
            nodes[key]=(z["ctx"][1],z["plans"]+1,z["walk_rgb"],z["collided"])
    assert len(nodes)==40
    images={}
    def image(frame):
        key=sha256(frame.tobytes()).digest()
        images.setdefault(key,frame)
        return key
    rows=[]
    for start in sorted(k for k in nodes if len(k)<=3):
        src=image(nodes[start][0])
        for first in range(3):
            key=start
            actions=[]
            contacts=[]
            for depth in range(1,5):
                if key not in nodes:
                    break
                _,plans,frames,bumped=nodes[key]
                branch=first if depth==1 else fixed_child(path.name,start,first,depth)
                b=(bumped[branch]).astype(bool).tolist()
                p=plans[branch].astype(int).tolist()
                assert len(b)==len(p)==24 and set(p)<=set((1,2,3))
                if depth>1:
                    # Intermediate three-view observations consumed four actual turns.
                    # Neither intermediate images nor collision events are policy inputs.
                    actions.extend(SCAN)
                    contacts.extend([False]*4)
                for tap in ((12,24) if depth==1 else (24,)):
                    prefix=actions+p[:tap]
                    horizon=len(prefix)
                    assert horizon in HORIZONS
                    dest=image(frames[branch,tap//6-1])
                    rows.append((src,dest,prefix,plan_pose(prefix),f"{path.name}:{start}:h{horizon}",
                                 int(any(contacts+b[:tap])),path.name,start))
                actions.extend(p)
                contacts.extend(b)
                key+=str(branch)
    names=list(images)
    encoded=encoder.encode([images[k] for k in names])
    vectors=dict(zip(names,encoded))
    return [(vectors[s],vectors[f],a,p,g,c,o,n) for s,f,a,p,g,c,o,n in rows]


def build(house,source_split,split,encoder,directory):
    folder=SOURCE/"trees"/source_split/house
    receipt=json.loads((folder/"receipt.json").read_text())
    origins=sorted(p for p in folder.iterdir() if p.is_dir())
    assert len(origins)==receipt["origins"]
    rows=[sample for p in origins for sample in one_tree(p,encoder)]
    ctx,fut,plans,pose,groups,contact,orig,node=zip(*rows)
    padded=np.zeros((len(rows),108),np.uint8)
    for i,p in enumerate(plans):
        padded[i,:len(p)]=p
    dest=directory/f"{split}-{house}.npz"
    temp=dest.with_suffix(".building.npz")
    np.savez_compressed(temp,context=np.stack(ctx),future=np.stack(fut),pose=np.float32(pose),action=padded,
                        group=np.asarray(groups),split=np.full(len(rows),split),house=np.full(len(rows),house),
                        contact=np.asarray(contact,dtype=bool),origin=np.asarray(orig),node=np.asarray(node))
    temp.replace(dest)
    return dict(origins=len(origins),rows=len(rows),houses=1)


def cohorts():
    old=json.loads((SOURCE/"cohort.json").read_text())
    training=json.loads((SOURCE/"training_cohort.json").read_text())
    splits=[("train","train",training["train"]),("calibration","val",old["calibration"]),
            ("fresh","test",old["fresh"])]
    houses=[h.split("_")[0] for _,_,all_houses in splits for h in all_houses]
    assert len(set(houses))==len(houses)==30
    return splits


def audit(directory):
    summary={}
    for _,split,houses in cohorts():
        lengths=Counter()
        groups=total_origins=contacts=0
        for house in houses:
            with np.load(directory/f"{split}-{house}.npz") as z:
                assert set(z["house"])=={house} and set(z["split"])=={split}
                assert z["context"].shape==z["future"].shape==(len(z["pose"]),1024)
                assert np.isfinite(z["pose"]).all() and z["pose"].shape[1]==len(POSE_NAMES)
                assert np.allclose(np.linalg.norm(z["future"].astype("float32"),axis=1),1,atol=.001)
                assert np.allclose(np.linalg.norm(z["context"].astype("float32"),axis=1),1,atol=.001)
                assert (z["action"]<=3).all()
                a=z["action"]
                assert not ((a[:,:-1]==0)&(a[:,1:]!=0)).any()
                h=(a>0).sum(1)
                lengths.update(map(int,h))
                total_origins+=len(set(z["origin"]))
                for group in set(z["group"]):
                    ids=np.flatnonzero(z["group"]==group)
                    assert len(ids)==3 and np.array_equal(z["context"][ids[0]],z["context"][ids[1]])
                groups+=len(set(z["group"]))
                contacts+=int(z["contact"].sum())
        assert set(lengths)==set(HORIZONS)
        summary[split]=dict(houses=len(houses),trajectory_roots=total_origins,
                            distinct_source_horizon_groups=groups,row_count=sum(lengths.values()),
                            count_by_physical_actions=dict(sorted(lengths.items())),paths_with_reported_contact=contacts)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--audit-only",action="store_true")
    args=p.parse_args()
    directory=HERE/"private_long"
    directory.mkdir(exist_ok=True,mode=0o700)
    os.umask(0o077)
    if not args.audit_only:
        encoder=FrozenImageEncoder()
        for old,new,houses in cohorts():
            for house in sorted(houses):
                dest=directory/f"{new}-{house}.npz"
                if not dest.exists():
                    info=build(house,old,new,encoder,directory)
                    print(new,house,info,flush=True)
    report=dict(splits=audit(directory),action_encoding={"padding":0,"forward":1,"left":2,"right":3},
                split_origin="Disjoint groups within existing HSSD official training houses; calibration and fresh are held out at building-family level.",
                sensor=dict(hfov_deg=90,pitch_deg=0,height_m=1.25,resolution=[128,128]),
                pose_fields=POSE_NAMES,image_encoder="Frozen OpenAI CLIP RN50 whole frame, unit-norm 1024-D.",
                horizon_protocol="Source central RGB from the same three-view scan. Each leg is 24 physically executed moves. Each intermediate scan consumes R,L,L,R. Steps start immediately after the source scan. Context excludes all intermediate images. Actor pose is the requested endpoint under ideal odometry, never simulator truth.",
                provenance={s:sha256((SOURCE/s).read_bytes()).hexdigest() for s in
                            ("tree.py","training_cohort.json","cohort.json","replay_summary.json")})
    (HERE/"long_receipt.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
