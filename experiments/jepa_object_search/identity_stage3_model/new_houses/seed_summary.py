"""Aggregate the three predetermined seeds without selecting on test outcomes."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import sys
import numpy as np


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def summarize(out):
    tasks=read(out/"private/cases.jsonl")
    data={}
    scorelines={}
    for seed in (126111,126112,126113):
        folder=out if seed==126111 else out/"seed_checks"/("seed"+str(seed))
        metrics=json.loads((folder/"results.json").read_text())
        record={}
        for m in ("direct","jepa","position"):
            lines=read(folder/"private"/(m+".jsonl"))
            assert len(lines)==len(tasks)
            scorelines[seed,m]=lines
            record[m]={kind:{tag:dict(Counter(lines[i][tag] for i,t in enumerate(tasks) if t["kind"]==kind))
                             for tag in ("clean","pose_stress")} for kind in ("same","absent_with_same_class")}
        data[seed]=dict(methods=record,checkpoints=metrics["checkpoints"])
    deltas={}
    for tag in ("clean","pose_stress"):
        for m in ("direct","position"):
            byhouse=defaultdict(list)
            for i,t in enumerate(tasks):
                if t["kind"]=="same":
                    byhouse[t["house"]].append(np.mean([int(scorelines[s,"jepa"][i][tag]=="correct")-
                                    int(scorelines[s,m][i][tag]=="correct") for s in data]))
            values=[np.mean(v) for v in byhouse.values()]
            draws=np.random.default_rng(7129).choice(values,size=(10000,len(values))).mean(1)
            deltas[tag+"/"+m]=dict(n_houses=len(values),mean=float(np.mean(values)),ci95=np.quantile(draws,[.025,.975]).tolist())
    (out/"stability.json").write_text(json.dumps(dict(seeds=data,paired_house_macro_delta=deltas,paid_calls=0),
                                                 indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    summarize(Path(sys.argv[1]))
