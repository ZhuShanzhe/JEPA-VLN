"""Score fixed, previously generated HSSD proposals. Writes aggregate counts only."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


def table(path):
    records = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    out = {r["frame_id"]: r for r in records}
    assert len(out) == len(records)
    return out


def coords(records):
    return [r.get("bbox", r.get("xyxy")) for r in records]


def matched(matrix, threshold):
    if not matrix.size:
        return set()
    ok = matrix >= threshold
    pred, truth = linear_sum_assignment(-(ok*(min(matrix.shape)+1)+matrix))
    return {int(j) for i,j in zip(pred, truth) if ok[i,j]}


def merge(dino, sam):
    # Read only scores available before evaluation; break exact ties in favor of DINO.
    scored = [(r,source) for source,pool in enumerate([dino,sam]) for r in pool]
    return [r for r,source in sorted(scored, key=lambda item:(-item[0]["review_margin"], item[1]))]


def main(project, output):
    prev = project/"jepa_box_v26"
    v21 = project/"jepa_reid_v21"
    v25 = project/"jepa_reid_v25/candidates"
    sys.path[:0] = [str(prev/"compare"),str(v25)]
    from score_boxes import overlaps
    from source_quality import best
    spec = json.loads((v25/"selector.json").read_bytes())
    manifest_path = v21/"data/frames.jsonl"
    frames = {fid:r for fid,r in table(manifest_path).items() if r["split"]=="test"}
    paths = {arm:prev/f"compare/private/hssd_{arm}.jsonl" for arm in ("dino","sam")}
    ranks = {arm:table(path) for arm,path in paths.items()}
    native = table(v21/"detector/predictions.jsonl")
    dims = table(prev/"sam/private/hssd/rows.jsonl")
    assert all(set(rows)==set(frames) for rows in ranks.values())
    assert len(frames)==288 and len({r["house"] for r in frames.values()})==16
    summary = defaultdict(lambda: defaultdict(Counter))
    size = defaultdict(Counter)
    paired = Counter()
    households = defaultdict(Counter)
    sequence_evidence=defaultdict(list)
    for fid, frame in frames.items():
        gt = coords(frame["boxes"])
        w,h = dims[fid]["width"],dims[fid]["height"]
        small = {j for j,(x1,y1,x2,y2) in enumerate(gt) if (x2-x1)*(y2-y1)<=.02*w*h}
        score, idx = best(spec, native[fid]["boxes"],w,h)
        source = [native[fid]["boxes"][idx]] if idx is not None and score>=spec["threshold"] else []
        d,s = [ranks[arm][fid]["boxes"] for arm in ("dino","sam")]
        pools = {
            "dino_common_top1": d[:1], "dino_common_top3": d[:3], "dino_all":d,
            "dino_common_gated":d[:1] if d and d[0]["review_margin"]>0 else [],
            "dino_source_category_gate":source,
            "sam_common_top3":s[:3],"sam_all":s,
            "merged_common_top3":merge(d,s)[:3],"dino_top3_plus_sam_top3":d[:3]+s[:3],
        }
        matrices={name:overlaps(coords(pool),gt) for name,pool in pools.items()}
        groups=["all", "class_"+frame["category"]]
        if len(gt)>0:
            groups.append("same_class_multiple" if len(gt)>=2 else "single")
        for cutoff in (.30,.50,.75):
            hits={name:matched(mat,cutoff) for name,mat in matrices.items()}
            for name,pool in pools.items():
                target_hits=hits[name]
                nh=len(target_hits)
                for g in groups if cutoff==.5 else ["all"]:
                    summary[f"{cutoff:.2f}"][g+"/"+name].update(
                        frames=1, positives=int(bool(gt)), negatives=int(not gt), targets=len(gt),
                        proposals=len(pool), offered_frames=int(bool(pool)),
                        positive_frames_hit=int(nh>0), at_least_two=int(nh>=2),
                        negative_frames_offered=int(not gt and bool(pool)),
                        wrong_only_positive_frames=int(bool(gt) and bool(pool) and not nh),
                        no_offer_positive_frames=int(bool(gt) and not pool), one_to_one_hits=nh)
                if cutoff==.5:
                    size[name].update(small=len(small),big=len(gt)-len(small),
                        small_hits=len(target_hits&small),big_hits=len(target_hits-small))
                if cutoff==.5 and name in ("dino_common_top3","dino_all","dino_source_category_gate"):
                    c=households[frame["house"]]
                    c[name+"_hit_frames"]+=int(nh>0)
                    c[name+"_hit_targets"]+=nh
                    c[name+"_offered"]+=int(bool(pool))
                    c[name+"_targets"]+=len(gt)
                    c[name+"_positives"]+=int(bool(gt))
            if cutoff==.5 and gt:
                allhit=bool(hits["dino_all"])
                tophit=bool(hits["dino_common_top1"])
                threehit=bool(hits["dino_common_top3"])
                paired.update(frames_with_target=1, no_dino_proposals=int(not d),
                    proposals_exist_none_localize=int(bool(d) and not allhit),
                    any_localization_but_none_top3=int(allhit and not threehit),
                    top3_localizes_but_top1_wrong=int(threehit and not tophit),
                    top1_localizes=int(tophit))
                for n in ("sam_common_top3","sam_all","dino_top3_plus_sam_top3","merged_common_top3"):
                    paired[n+"_new_positive_frames_vs_dino3"]+=int(bool(hits[n]) and not threehit)
                    paired[n+"_lost_positive_frames_vs_dino3"]+=int(not hits[n] and threehit)
            if cutoff==.5:
                ids=[r["instance"] for r in frame["boxes"]]
                assert len(ids)==len(set(ids))
                sequence_evidence[frame["sequence"]].append((set(ids),{
                    arm:{ids[i] for i in hits[arm]} for arm in ("dino_all","dino_common_top3","dino_source_category_gate")
                }))
    assert len(households)==16 and len(sequence_evidence)==96
    cross=defaultdict(Counter)
    for views in sequence_evidence.values():
        assert len(views)==3
        for (v1,h1),(v2,h2) in combinations(views,2):
            both=v1&v2
            for arm in h1:
                cross[arm].update(visible_both=len(both),matched_both=len(both&h1[arm]&h2[arm]),
                    exactly_one=len(both&(h1[arm]^h2[arm])),neither=len(both-h1[arm]-h2[arm]))
    # Preserve spatial correlation by resampling whole houses and recomputing pooled ratios.
    house_ci={}
    rng=np.random.default_rng(2611)
    sampled=rng.integers(len(households),size=(10000,len(households)))
    for label,arm,num,den in [
        ("top3_any_frame","dino_common_top3","hit_frames","positives"),
        ("all_target_recall","dino_all","hit_targets","targets"),
        ("source_gate_precision","dino_source_category_gate","hit_frames","offered")]:
        pairs=np.array([[c[arm+"_"+num],c[arm+"_"+den]] for c in households.values()])
        sums=pairs[sampled].sum(axis=1)
        props=sums[:,0]/sums[:,1]
        house_ci[label]=dict(counts=pairs.sum(axis=0).tolist(),ci95=np.quantile(props,[.025,.975]).tolist())
    assert sum(paired[k] for k in ("no_dino_proposals","proposals_exist_none_localize","any_localization_but_none_top3","top3_localizes_but_top1_wrong","top1_localizes"))==244
    result=dict(houses=16,paid_calls=0,manifest_digest=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        frozen_scorer_digests=sorted({r["model_hash"] for arm in ranks.values() for r in arm.values()}),
        source_gate_model=spec["mode"],size_threshold_fraction=.02,counts=summary,size=size,
        positive_frame_partition=paired,house_bootstrap=house_ci,paired_views=cross)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(partition=paired,small=size["dino_all"],ci=house_ci),indent=2))


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("project",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    main(args.project,args.output)
