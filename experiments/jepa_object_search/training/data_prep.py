"""Build immutable, house-split branch examples from audited simulator products."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def within(base, relative):
    p = (base / relative).resolve()
    if not p.is_relative_to(base.resolve()):
        raise ValueError(f"Path outside data root: {relative}")
    return p


def require(test, message):
    if not test:
        raise ValueError(message)


def quality(counts, positions):
    """Best two consecutive distinct views per instance, soft 8..128 pixels at 256²."""
    require(counts.ndim == 2 and len(counts) == len(positions), "pixel/pose mismatch")
    valid = np.any(np.abs(np.diff(positions, axis=0)) > 1.e-4, axis=1)
    evidence = np.minimum(counts[1:], counts[:-1])[valid]
    peak = evidence.max(axis=0, initial=0)
    return np.clip((peak.astype(np.float32) - 8) / 120., 0., 1.)


class Builder:
    def __init__(self, root, output, device):
        self.root = root
        self.output = output
        self.device = device
        self.rows = []
        self.ids = set()
        self.image_cache = {}
        self.encoder = None
        self.freeze = read(root / "simulation/train_dev_v1.json")
        require(not set(self.freeze["train"]) & set(self.freeze["dev"]), "house leakage")

    def encode(self, frames):
        frames = np.asarray(frames)
        require(frames.ndim == 4 and len(frames) and frames.shape[-1] == 3
                and frames.dtype == np.uint8, "invalid real RGB clip")
        indices = np.linspace(0, len(frames)-1, 4, dtype=np.int64)
        key = hashlib.sha256(frames[indices].tobytes()).digest()
        if key not in self.image_cache:
            if self.encoder is None:
                import torch
                torch.set_num_threads(2)
                sys.path.insert(0, str(self.root / "frontier_jepa"))
                from vision import ContextEncoder
                self.encoder = ContextEncoder(self.root, self.device)
            self.image_cache[key] = self.encoder.encode(frames)
        return self.image_cache[key]

    def add(self, meta, context, future, xy, actions, gain, step_m, turn_deg,
            memory=None, memory_xy=None, region_presence=None):
        source, house, split, uid = (meta[k] for k in ("source","house","split","id"))
        require(source in ("hssd","blender") and split in ("train","dev") and uid not in self.ids,
                f"invalid or duplicate example {uid}")
        if source == "hssd":
            require(house in self.freeze[split] and house not in self.freeze["test_excluded"], "unfrozen house")
        self.ids.add(uid)
        xy = np.asarray(xy, dtype=np.float32)
        n = len(xy)
        require(n >= 2 and xy.shape == (n,2) and np.shape(future)==(n,768) and np.shape(context)==(768,),
                f"feature size {uid}")
        actions = np.asarray(actions)
        require(actions.ndim==2 and actions.shape[0]==n and actions.shape[1]>0 and actions.dtype.kind in "ui"
                and np.isin(actions,[0,1,2]).all(), "invalid primitive plan")
        require(np.shape(gain)==(n,) and np.min(gain)>=0 and step_m>0 and turn_deg>0, "invalid target or physics")
        mem = memory is not None
        require(mem == (memory_xy is not None), "incomplete observed memory")
        presence = np.full(n,-1) if region_presence is None else np.asarray(region_presence)
        require(presence.shape==(n,) and np.isin(presence,[-1,0,1]).all(), "invalid presence")
        arrays = dict(context=context, future=future, xy=xy, actions=actions.astype(np.int64),
                      gain=np.asarray(gain,np.float32), step_m=np.float32(step_m), turn_deg=np.float32(turn_deg),
                      memory=np.zeros(768,np.float32) if not mem else np.asarray(memory,np.float32),
                      memory_xy=np.zeros(2,np.float32) if not mem else np.asarray(memory_xy,np.float32),
                      has_memory=np.bool_(mem), region_presence=presence.astype(np.int8))
        require(arrays["memory"].shape==(768,) and arrays["memory_xy"].shape==(2,) and
                all(np.isfinite(x).all() for x in arrays.values()), "nonfinite or invalid example")
        path = Path("states") / f"{len(self.rows):06d}.npz"
        np.savez(self.output/path, **arrays)
        self.rows.append({**meta,"path":path.as_posix(),"sha256":sha(self.output/path)})

    def hssd(self):
        base=self.root/"a_exploration/paired_branches_v2"
        audit=read(base/"audit.json")
        require(audit["complete"] and sha(base/"cohort.json")==audit["cohort_sha256"],"HSSD cohort audit")
        house_hash=sha(self.root/"simulation/train_dev_v1.json")
        for row in records(base/"index.jsonl"):
            p=within(base,row["id"])
            m=read(p/"manifest.json")
            require(m["split_sha256"]==house_hash and m["source_scene"]==row["scene"] and m["split"]==row["split"],
                    "HSSD split or manifest mismatch")
            with np.load(p/"features_vjepa21_b384.npz") as f:
                ctx,future=f["context_embedding"],f["future_targets"]
            with np.load(p/"context.npz") as f:
                xy,actions=f["frontiers_xy"],f["planned_actions"]
            teacher=read(p/"teacher/labels.json")
            require(teacher["synset"]==row["target"] and m["action_budget"]==24,"query/physics mismatch")
            ids=sorted(map(int,teacher["instance_map"]))
            prefix=np.asarray(teacher["prefix_pixel_counts"])
            require(len(prefix)==12 and len(ids)==prefix.shape[1],"HSSD prefix")
            # Prefix is a full turn with twelve distinct yaw values. Include the seam, since
            # 330° and the observed 0° frame are adjacent viewing directions.
            prefix_quality=quality(np.vstack([prefix,prefix[:1]]), np.arange(13)[:,None])
            gain=[]
            for i in range(len(xy)):
                with np.load(p/"teacher"/f"branch{i:03d}.npz") as f:
                    require(f["instance_ids"].tolist()==ids,"instance column order")
                    counts=f["target_pixel_counts"]
                with np.load(p/f"branch{i:03d}.npz") as f:
                    pose=f["relative_pose_xyyaw"]
                score=quality(counts,pose)
                gain.append(float(np.maximum(score-prefix_quality,0).sum()))
            self.add(dict(id="hssd/"+row["id"],source="hssd",house=row["scene"],split=row["split"],
                          query=row["target"],label_kind="hssd_visible_evidence_delta"),
                     ctx,future,xy,actions,gain,.25,30.)
        require(len(self.rows)==audit["rows"],"incomplete HSSD index")

    def report_pairs(self):
        base=self.root/"a_exploration/paired_branches_v2"
        for i in (0,2,4):
            p=self.root/"frontier_jepa/runs"/f"hssd_report_train_pair{i}"
            r=read(p/"receipt.json")
            require(r["status"]=="pass" and r["train_only"],"report pair audit")
            for name,digest in r["decision_input_sha256"].items():
                require(sha(within(p/"decision",name))==digest,"report history changed")
            with np.load(within(base,r["source"])/"features_vjepa21_b384.npz") as f:
                future=f["future_targets"]
            with np.load(within(base,r["source"])/"context.npz") as f:
                plans=f["planned_actions"]
                xy=f["frontiers_xy"]
            for h in ("h0","h1"):
                d=read(p/"decision"/(h+".json"))
                require(np.array_equal(plans,[c["planned_actions"] for c in d["candidates"]]) and
                        np.allclose(xy,[c["relative_forward_left_m"] for c in d["candidates"]]),
                        "return-state plan mismatch")
                mem=d["reports"][0]
                history=np.stack([rgb(within(p/"decision",name)) for name in d["observed_rgb"]])
                memory=self.encode(rgb(within(p/"decision",mem["observed_rgb"]))[None])
                labels=read(p/"judge/witnesses_and_labels.json")[h]["utility"]
                self.add(dict(id=f"hssd/report-pair{i}/{h}",source="hssd",house=r["scene"],split="train",
                              query=r["query"],label_kind="controlled_prior_report"),
                         self.encode(history),future,xy,plans,labels,.25,30.,memory,
                         mem["odometry_forward_left_yaw"][:2])

    def blender(self):
        base=self.root/"frontier_scenes"
        p=base/"acceptance"
        scene=read(p/"scene_smoke.json")
        sides=["left","right"]
        # The fixture camera resets to identity rotation, so forward is -z, left is -x.
        def localxz(xz):
            return -np.asarray(xz)[...,::-1]
        xy=localxz([scene["candidates"][s]["target_relative_xz_m"] for s in sides])
        vocab=["move_forward","turn_left","turn_right"]
        plans=np.array([[vocab.index(a) for a in scene["candidates"][s]["planned_actions"]] for s in sides])
        futures=[]
        start=rgb(p/"start.png")
        for side in sides:
            clip=p/f"{side}_trajectory.npz"
            require(sha(clip)==scene["trajectory_sha256"][side],"Blender scene changed")
            with np.load(clip) as f:
                frames=f["rgb"]
            require(np.array_equal(start,frames[0]),"different Blender prefix")
            futures.append(self.encode(frames[1:]))
        for side in sides:
            hist=scene["report_histories"][side]
            pos=(np.asarray(hist["report_observed_pose"])-np.asarray(scene["start"]))[[0,2]]
            self.add(dict(id="blender/report/"+side,source="blender",house="two-wing-fixture",split="train",
                          query="chair.n.01",label_kind="controlled_prior_report"),
                     self.encode(start[None]),np.stack(futures),xy,plans,[hist["candidate_labels"][s] for s in sides],
                     .25,15.,self.encode(rgb(p/hist["report_rgb"])[None]),localxz(pos))
        p=base/"attribute_fixture"
        report=read(p/"reports/counterbalance.json")
        for row in report["rows"]:
            layout=row["layout"]
            start=rgb(p/"clips"/(layout+"_start.png"))
            futures=[]
            for k,side in enumerate(sides):
                with np.load(p/"clips"/f"{layout}_{side}.npz") as f:
                    frames=f["rgb"]
                    require(np.array_equal(start,frames[0]) and
                            [vocab.index(a) for a in f["planned_actions"]]==plans[k].tolist(),
                            "Blender layout prefix or planned steps changed")
                    futures.append(self.encode(frames[1:]))
            self.add(dict(id=f"blender/{layout}/{row['query']}",source="blender",house="two-wing-fixture",
                          split="train",query=row["query"]+" chair",label_kind="controlled_color_query"),
                     self.encode(start[None]),np.stack(futures),xy,plans,[row["supervision"][s] for s in sides],.25,15.)

    def extra(self, directory):
        for row in records(directory/"index.jsonl"):
            with np.load(within(directory,row["path"]),allow_pickle=False) as f:
                frames=f["future_rgb"]
                require(len(frames)==len(f["xy"]) and frames.shape[1]==f["actions"].shape[1],
                        "future clips must contain exactly the post-action RGB frames")
                has="report_rgb" in f
                require(has==("memory_xy" in f),"extra observed report missing position or RGB")
                self.add(dict(id="extra/"+row["id"],source="blender",house=row["house"],split=row["split"],
                              query=row["query"],label_kind=row["label_kind"]),
                         self.encode(f["context_rgb"]),np.stack([self.encode(c) for c in frames]),f["xy"],
                         f["actions"],f["gain"],float(f["step_m"]),float(f["turn_deg"]),
                         self.encode(f["report_rgb"]) if has else None,f["memory_xy"] if has else None,
                         f["region_presence"] if "region_presence" in f else None)

    def import_base(self, directory):
        receipt=read(directory/"receipt.json")
        require(receipt["schema"]=="most-branch-training/v1" and
                sha(directory/"index.jsonl")==receipt["index_sha256"],"bad base index")
        for row in records(directory/"index.jsonl"):
            inp=within(directory,row["path"])
            require(row["id"] not in self.ids and sha(inp)==row["sha256"],"bad base data")
            self.ids.add(row["id"])
            path=Path("states")/f"{len(self.rows):06d}.npz"
            shutil.copyfile(inp,self.output/path)
            self.rows.append({**row,"path":path.as_posix()})

    def finish(self):
        ownership={}
        groups=defaultdict(lambda:dict(states=0,branches=0,discriminating_states=0,with_memory=0,
                                        region_labels=0,houses=set()))
        for row in self.rows:
            key=(row["source"],row["house"])
            require(ownership.setdefault(key,row["split"])==row["split"], "layout or house in multiple splits")
            stats=groups["/".join([row["source"],row["split"],row["label_kind"]])]
            with np.load(self.output/row["path"]) as f:
                y=f["gain"]
                stats["branches"]+=len(y)
                stats["discriminating_states"]+=int(np.ptp(y)>1.e-6)
                stats["with_memory"]+=int(f["has_memory"])
                stats["region_labels"]+=int((f["region_presence"]>=0).sum())
            stats["states"]+=1
            stats["houses"].add(row["house"])
        for stats in groups.values():
            stats["houses"]=len(stats["houses"])
        index=self.output/"index.jsonl"
        index.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in self.rows))
        result=dict(schema="most-branch-training/v1",states=len(self.rows),index_sha256=sha(index),
                    groups=dict(groups),house_split_sha256=sha(self.root/"simulation/train_dev_v1.json"),
                    feature="V-JEPA2.1 ViT-B/16 384 EMA, mean pooled, four uniform RGB frames",
                    target="Positive increment of sustained per-instance evidence on HSSD, controlled conditional utility on labelled fixtures. Not official navigation rewards.",
                    memory="Observed report frame + decision-relative camera XY for constructed reports; absent elsewhere.",
                    geometry="Candidate and memory XY in forward,left metres; actions 0 forward, 1 left, 2 right.")
        (self.output/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))


def rgb(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--base-pack",type=Path)
    parser.add_argument("--extra-raw",type=Path)
    parser.add_argument("--encoder-device",choices=["cpu","cuda"],default="cpu")
    args=parser.parse_args()
    require(not args.output.exists(),"Output must be a new directory")
    require(args.base_pack is None or args.extra_raw is not None,"Use --base-pack with --extra-raw")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as t:
        pending=Path(t)/"pack"
        (pending/"states").mkdir(parents=True)
        b=Builder(args.root.resolve(),pending,args.encoder_device)
        if args.base_pack:
            b.import_base(args.base_pack)
        else:
            b.hssd()
            b.report_pairs()
            b.blender()
        if args.extra_raw:
            b.extra(args.extra_raw)
        b.finish()
        pending.rename(args.output)


if __name__=="__main__":
    main()
