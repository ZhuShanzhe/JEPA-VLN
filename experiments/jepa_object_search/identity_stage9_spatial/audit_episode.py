"""Audit one accepted spatial-policy clock episode with the previous two real runs."""
from collections import Counter
from pathlib import Path
import json
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'identity_stage8_relook'))
from audit_clock import accepted, check_pair, digest, lines, load, score


def main():
    study=ROOT/'mapping_runtime/pilots/identity_stage9_spatial'
    parent=study.parent/'identity_stage8_weak_relook'
    ref,=load(study/'panel.json')['rows']
    assert ref==load(parent/'panel.json')['rows'][0]
    name=ref['episode_id']+'.json'
    p,doc,ep=accepted(study/'comparison/accepted/second'/name)
    _,ndoc,native=accepted(ref['ref'])
    _,prevdoc,previous=accepted(parent/'comparison/accepted/second'/name)
    for other in (ndoc,prevdoc):check_pair(doc,other)
    events=lines(p.parent/'agent/confirm_events.jsonl')
    counts=Counter(r['event'] for r in events)
    ranked=[r for r in events if r['event']=='spatial_view_selection']
    judges=[r for r in events if r['event']=='mapped_identity_decision']
    def by(event,key):return dict(Counter(str(r.get(key)) for r in events if r['event']==event))
    # Subtract the starting set: all new distinct IDs must come from official report receipts.
    def credited(e):
        return {x for row in e['report_events'] for x in row['new_targets']}
    out={'protocol':'one paired autonomous development episode; same official reporting geometry, upper model, start and 300-action budget',
         'frozen_source_sha256':digest(p),'native':score(native),'previous':score(previous),'spatial':score(ep),
         'shared_targets_native':len(credited(ep)&credited(native)),'additional_vs_native':len(credited(ep)-credited(native)),
         'shared_targets_previous':len(credited(ep)&credited(previous)),
         'evidence':{'new_view_rankings':len(ranked),'nontrivial_view_comparisons':sum(r['options']>1 for r in ranked),
                     'capture_count':counts['weak_relook_capture'],'relook_actions':counts['weak_relook_action'],
                     'spatial_verdicts':len(judges),'spatial_status':by('mapped_identity_decision','status'),
                     'spatial_reason':by('mapped_identity_decision','reason'),'delayed':by('relook_delayed_result','reason'),
                     'immediate':by('weak_relook_evidence','reason'),'departures':counts['weak_region_departure'],
                     'confirmed_at_capture':counts['weak_duplicate_confirmed']},
         'closed':(study/'closed.json').is_file()}
    assert out['closed']
    target=HERE/'episode_results.json'
    text=json.dumps(out,indent=2,allow_nan=False)+'\n'
    if target.exists():assert target.read_text()==text
    else:target.write_text(text)
    print(json.dumps(out['spatial']),flush=True)


if __name__=='__main__':main()
