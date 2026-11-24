"""Evaluate visible-target recognition and query-sensitive route choices after inference."""
import json
from pathlib import Path
import sys
import numpy as np


def load(p):
    return json.loads(p.read_bytes())


def main(root):
    d=root/"streams/test";cam=load(root/"test.camera.json")
    predicts=load(root/"test.predictions.json")["episodes"];rows={};choice={}
    for f in sorted(d.glob("*.actor.json")):
        k=f.name.split(".actor")[0];hist=load(f)["history"]
        counts=load(d/(k+".visibility.json"))
        for obs,pix in zip(hist,counts):
            if obs[0]%4!=0:continue
            ident=obs[-1];y=[float(p>=64) for p in pix]
            if ident in rows and rows[ident][0]!=y:
                raise RuntimeError("Equal camera image with contradictory labeling")
            rows[ident]=(y,cam[ident])
        truth=load(d/(k+".truth.json"));result=predicts[k]
        for j,stage in enumerate(truth["labels"]):
            mode="moved" if truth["relocated"] else "stationary"
            by_arm=choice.setdefault(mode+"/"+stage["phase"],{})
            for arm in result[0][j]["scores"]:
                # Compare both target words in exactly the same world and agent history.
                diffs=[r[j]["scores"][arm][0]-r[j]["scores"][arm][1] for r in result]
                by_arm.setdefault(arm,[]).append(bool(diffs[0]*diffs[1]<0 and min(map(abs,diffs))>1e-9))
    labels,scores=[np.array(z) for z in zip(*rows.values())]
    quality={}
    for i,query in enumerate(("cup","sofa")):
        present=scores[labels[:,i]==1,i];absent=scores[labels[:,i]==0,i]
        margin=present[:,None]-absent
        quality[query]=dict(present=len(present),absent=len(absent),
            AUC=float(np.mean((margin>0)+.5*(margin==0))),median_when_visible=float(np.median(present)),
            median_when_absent=float(np.median(absent)))
    result=dict(unique_sampled_rgb=len(rows),per_query_visible_target_grounding=quality,
        different_branch_for_cup_vs_sofa={k:{a:[sum(vals),len(vals)] for a,vals in v.items()} for k,v in choice.items()})
    (root/"perception_diagnostic.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(quality),"postmove",result["different_branch_for_cup_vs_sofa"]["moved/revisited_both"])


if __name__=="__main__":main(Path(sys.argv[1]))
