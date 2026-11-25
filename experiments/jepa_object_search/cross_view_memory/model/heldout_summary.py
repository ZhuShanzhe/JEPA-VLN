"""Aggregate existing predictions without refitting or changing a threshold."""
import argparse
import json
from pathlib import Path

import numpy as np


def counts_by_mask(y, scores, threshold, subset):
    pred=scores[subset]>=threshold
    pos=y[subset].astype(bool)
    return np.array([np.sum(pred&pos),pos.sum(),np.sum(pred&~pos),(~pos).sum()])


def summarize(counts):
    tp,p,fp,n=map(int,counts)
    return dict(positives=p,negatives=n,correct_matches=tp,false_matches=fp,
                recall=tp/p if p else None,false_merge=fp/n if n else None)


def main(root, manifest):
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    scores=np.load(root/"pairs_test.npz")
    meta=json.loads((root/"metrics.json").read_text())
    i,j,y=(scores[k] for k in ("i","j","y"))
    houses=np.array([rows[k]["house"] for k in i])
    categories=np.array([rows[k]["category"] for k in i])
    a,b=(np.array([rows[k]["asset"] for k in ids]) for ids in (i,j))
    train_assets={r["asset"] for r in rows if r["split"]=="train"}
    unseen=np.array([asset not in train_assets for asset in a])
    unseen_other=np.array([asset not in train_assets for asset in b])
    house_ids=np.unique(houses)
    rng=np.random.default_rng(3161)
    repeats=rng.multinomial(len(house_ids),np.repeat(1/len(house_ids),len(house_ids)),size=5000)
    summary={}
    rep_rates={}
    for mode in meta["thresholds"]:
        s=scores["s_"+mode]
        t=meta["thresholds"][mode]
        counts=np.stack([counts_by_mask(y,s,t,houses==h) for h in house_ids])
        aggregate=summarize(counts.sum(axis=0))
        boot=repeats@counts
        recall=boot[:,0]/boot[:,1]
        fpr=boot[:,2]/boot[:,3]
        rep_rates[mode]=np.stack([recall,fpr],axis=1)
        by_category={c:summarize(counts_by_mask(y,s,t,categories==c)) for c in sorted(set(categories))}
        subsets={"impostor_same_asset":(a==b)&(y==0),"impostor_different_asset":(a!=b)&(y==0),
                 "both_assets_unseen_in_training":unseen&unseen_other,"same_instance_unseen_asset":unseen&(y==1)}
        summary[mode]=dict(all_pairs=aggregate,per_category=by_category,
                           recall_and_false_merge_CI95=np.percentile(rep_rates[mode],[2.5,97.5],axis=0).T.tolist(),
                           by_asset={k:summarize(counts_by_mask(y,s,t,mask)) for k,mask in subsets.items()})
    diff={}
    for other in ("direct","frozen","jepa_object"):
        diff[other]=np.percentile(rep_rates["jepa"]-rep_rates[other],[2.5,97.5],axis=0).T.tolist()
    result=dict(protocol="5000 paired draws of held-out houses; evaluation pair thresholds fixed by validation. Columns are recall and false-merge rate. Rows per interval are [low, high].",
                number_test_houses=len(house_ids),summaries=summary,jepa_minus_comparator_CI95=diff)
    (root/"heldout_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("model",type=Path)
    parser.add_argument("manifest",type=Path)
    args=parser.parse_args()
    main(args.model,args.manifest)
