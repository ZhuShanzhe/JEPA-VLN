"""Rescore every prespecified seed on the unchanged panels, with one shared RGB-D encoding."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from compare import build, evaluate, paired, rows, sha, summarize


def run(project,out):
    train=project/"identity_stage3_model/train"
    sys.path.insert(0,str(train))
    from identity import IdentityVerifier
    frames,tasks,protocol=build(project)
    assert protocol==json.loads((out/"protocol.json").read_text())
    encoded={}
    records={}
    checksums={}
    protocol_metrics={}
    for seed in ("default","seed126112","seed126113"):
        directory=train/"private" if seed=="default" else train/"private"/seed
        protocol_metrics[seed]={}
        records[seed]={}
        for mode in ("direct","jepa"):
            verifier=IdentityVerifier(project,mode=mode,device="cuda",checkpoint_dir=directory)
            result=evaluate(project,frames,tasks,mode,"cuda",verifier=verifier,encoded_frames=encoded)
            rid=seed+"_"+mode
            checksums[rid]=sha(directory/(mode+".pt"))
            if seed=="default":
                assert result==rows(out/"private"/(mode+".jsonl")), "default run differs under cached frozen inputs"
            else:
                (out/"private"/(rid+".jsonl")).write_text("".join(json.dumps(r)+"\n" for r in result))
            records[seed][mode]=result
            protocol_metrics[seed][mode]=summarize(result)
            print(seed,mode,protocol_metrics[seed][mode]["visible"]["original"],
                  protocol_metrics[seed][mode]["twins"]["joint_withheld"],protocol_metrics[seed][mode]["twins"]["copy"],flush=True)
    paired_by_seed={k:paired(v["direct"],v["jepa"]) for k,v in records.items()}
    # Seeds share the same images and houses. Bootstrap houses, averaging the paired seed deltas within each case.
    def bootstrap(panel):
        diffs={}
        for i,task in enumerate(tasks):
            if panel=="visible_correct" and (task["panel"]!="revisit" or not task["visible"]):
                continue
            if panel!="visible_correct" and task["panel"]!="twins":
                continue
            def event(row):
                orig=row["pred"][0]["outcome"]=="original"
                if panel=="visible_correct": return orig
                if panel=="twins_joint": return orig and row["pred"][1]["state"]!="same"
                return row["pred"][1]["outcome"]=="original"
            d=[int(event(z["jepa"][i]))-int(event(z["direct"][i])) for z in records.values()]
            diffs.setdefault(task["house"],[]).append(np.mean(d))
        byhouse=np.asarray([np.mean(r) for r in diffs.values()])
        sampled=np.random.default_rng(260127).choice(byhouse,size=(10000,len(byhouse))).mean(axis=1)
        return dict(houses=len(byhouse),delta=float(byhouse.mean()),ci95=np.quantile(sampled,[.025,.975]).tolist())
    pooled={p:bootstrap(p) for p in ("visible_correct","twins_joint","twins_copy_false_accept")}
    (out/"stability.json").write_text(json.dumps(dict(paid_calls=0,default_reproduced_exactly=True,sources_sha256=checksums,
        protocol_sha256=sha(out/"protocol.json"),per_seed=protocol_metrics,deltas_per_seed=paired_by_seed,
        seed_averaged_paired_house_deltas=pooled),sort_keys=True,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("project",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    run(args.project,args.output)
