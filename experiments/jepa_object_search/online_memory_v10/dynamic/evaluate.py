"""Join independently computed scores and Habitat instance/route labels at evaluation time."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def accuracy(pair,correct):
    difference=pair[correct]-pair[1-correct]
    if abs(difference)<=1e-9:
        return .5
    return float(difference>0.)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder",type=Path)
    args=ap.parse_args()
    d=args.folder
    pred=read(d/"test.predictions.json")
    records=pred.pop("episodes")
    if len(records)!=24:
        raise RuntimeError("Missing held-out sequences")
    acc=defaultdict(list); tie=defaultdict(list); false_changes=defaultdict(list); recovered=defaultdict(list)
    detailed=defaultdict(list); cue_groups=defaultdict(list)
    freeze=hashlib.sha256(); distances=[]; rgb_diff=[]; max_pred_delta=0.
    for sid,queries in sorted(records.items()):
        path=d/"streams/test"/(sid+".truth.json");truth=read(path)
        actor_path=d/"streams/test"/(sid+".actor.json");actor=read(actor_path)
        freeze.update(sid.encode()+bytes.fromhex(hashlib.sha256(path.read_bytes()).hexdigest())+
                      bytes.fromhex(hashlib.sha256(actor_path.read_bytes()).hexdigest()))
        house=truth["house"]; layout="moved" if truth["relocated"] else "stationary"
        labels=truth["labels"]; arms=None
        assert len(queries)==2 and len(actor["history"])==301 and truth["same_origin_rgb_at_event"]
        assert all(abs(x-y)<.02 for l in labels for x,y in [l["travel_m"]])
        distances += [v for l in labels for v in l["travel_m"]]
        if not truth["relocated"]:
            paired=sid[:-1]+"1"
            other=read(d/"streams/test"/(paired+".actor.json"))
            changes=[r[0] for r,s in zip(actor["history"],other["history"]) if r[-1]!=s[-1]]
            if not changes or min(changes)<=labels[4]["t"]:
                raise RuntimeError(f"Unblinded event in {sid}: {changes[:1]}")
            rgb_diff.append(min(changes))
            for q,rows in enumerate(queries):
                for i in range(5):
                    for arm,u in rows[i]["scores"].items():
                        v=records[paired][q][i]["scores"][arm]
                        max_pred_delta=max(max_pred_delta,float(np.max(np.abs(np.array(u)-v))))
        for q,rows in enumerate(queries):
            target=truth["queries"][q]
            assert [r["step"] for r in rows]==[l["t"] for l in labels]
            stages=defaultdict(dict)
            for row,l in zip(rows,labels):
                good=l["query_routes"][q]
                assert sum(good)==1
                phase=l["phase"];arms=row["scores"].keys()
                for arm,pair in row["scores"].items():
                    hit=accuracy(pair,good.index(1))
                    stages[arm][phase]=hit
                    acc[layout,phase,arm].append(hit)
                    tie[layout,phase,arm].append(int(hit==.5))
                    detailed[house,layout,phase,arm].append(hit)
                    cue_groups[truth["cue"],layout,phase,arm].append(hit)
            for arm in arms:
                before=stages[arm]["both_observed"]
                for phase in ("old_evidence","revisited_left","revisited_both"):
                    now=stages[arm][phase]
                    if before==1 and layout=="stationary":
                        false_changes[phase,arm].append(float(now==0))
                    if before==1 and layout=="moved":
                        recovered[phase,arm].append(float(now==1))
    assert max_pred_delta<1e-8
    train=list((d/"streams/train").glob("*.truth.json"))
    summary=dict(train_rooms=sorted({read(x)["house"] for x in train}),train_sequences=len(train),
        training_visible_event_count=sum(int(not read(x)["same_origin_rgb_at_event"]) for x in train),
        test_rooms=sorted({k[0] for k in detailed}),test_sequences=len(records),goal_categories=2,
        true_object_relocations=12,stationary_pairs=12,steps_per_history=300,observations_per_history=301,
        checkpoints_per_query=7,queries_per_sequence=2,pairs_with_equal_prefix=len(rgb_diff),
        first_distinguishing_frame_minmax=[min(rgb_diff),max(rgb_diff)],prefix_prediction_difference=max_pred_delta,
        target_label="independent instance mask >=64 pixels; <=2m horizontal distance; actual 24-action legal trajectory",
        route_length_m_minmax=[min(distances),max(distances)],split_sha256=freeze.hexdigest(),
        checkpoint_sha256=pred["weights"],base_sha256=pred["v8"],memory_source_sha256=pred["memory_source"],
        sample_unit="paired same-house paths, point estimates, ties 0.5; no independent-house significance claim",
        samples_per_layout_phase=24,acc_by_layout_phase_arm={},tie_count={},acc_by_house={},acc_by_cue={},
        maintained_initially_correct={},recovered_after_change={})
    for mapping,values in [(summary["acc_by_layout_phase_arm"],acc),(summary["acc_by_house"],detailed),
                           (summary["acc_by_cue"],cue_groups)]:
        for key,items in values.items():
            *index,arm=key
            mapping.setdefault("/".join(index),{})[arm]=float(np.mean(items))
    for key,items in tie.items():
        *index,arm=key
        summary["tie_count"].setdefault("/".join(index),{})[arm]=sum(items)
    for key,items in false_changes.items():
        summary["maintained_initially_correct"]["/".join(key)]={"initially_correct":len(items),"became_wrong":int(sum(items))}
    for key,items in recovered.items():
        summary["recovered_after_change"]["/".join(key)]={"initially_correct_unmoved":len(items),"correct_new_position":int(sum(items))}
    (d/"dynamic_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k in ("train_sequences","test_sequences","prefix_prediction_difference",
        "first_distinguishing_frame_minmax","acc_by_layout_phase_arm","training_visible_event_count")},indent=2))


if __name__=="__main__":
    main()
