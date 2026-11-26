"""Score the frozen identity head with tracked vs exact geometry on identical HSSD candidates."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

from transfer import lines, source_view


def run(project):
    root=project/'identity_stage4_geometry/selection'
    split=project/'identity_stage3_model/new_houses/private'
    sys.path[:0]=[str(project/p) for p in ('identity_stage3_model/train','identity_stage4_geometry/provider',
                    'jepa_reid_v21/association','jepa_report_v19','jepa_reid_v23/report')]
    from identity import IdentityVerifier
    from evaluate import label_boxes
    from appearance import WORK_WIDTH
    from report_evidence import eligible_boxes
    from anchored_tracking import estimate_from_anchor

    cases=lines(split/'cases.jsonl')
    frames={r['name']:r for r in lines(split/'frames.jsonl')}
    dino={r['name']:r['boxes'] for r in lines(split/'dino.jsonl')}
    assert Counter(r['kind'] for r in cases)=={'same':92,'absent_with_same_class':94}
    K=np.asarray(next(iter(frames.values()))['K'])
    assert all(np.allclose(K, rec['K']) for rec in frames.values())
    model=IdentityVerifier(project,'jepa',device='cpu')
    cv2.setNumThreads(2)
    cv2.setRNGSeed(173)
    torch.set_num_threads(8)

    @lru_cache(maxsize=15)
    def rgb(name):
        with Image.open(split/'observations'/(name+'.png')) as f:
            return np.asarray(f.convert('RGB'))

    @lru_cache(maxsize=None)
    def query_boxes(name):
        im=rgb(name)
        eligible=eligible_boxes([{'bbox':b} for b in dino[name]],im.shape)
        return [np.asarray(b)*im.shape[1]/WORK_WIDTH for b in eligible]

    @lru_cache(maxsize=None)
    def truth_frame(name,is_registered_source):
        # Reference depth is allowed, query depth/pose in the separate upper-bound arm only.
        boxes=dino[name] if is_registered_source else query_boxes(name)
        with np.load(split/'observations'/(name+'.npz'),allow_pickle=False) as data:
            return model.encode(rgb(name),boxes,data['depth'],K,np.asarray(frames[name]['T']))

    @lru_cache(maxsize=None)
    def tracked(name):
        path=root/'private_anchors'/(name+'.npz')
        if not path.is_file():
            return 'historical_anchor_missing',None
        live=rgb(name)
        with np.load(path,allow_pickle=False) as hist:
            view=source_view(hist['depth'],K,hist['T'])
            geom,_=estimate_from_anchor(hist['rgb'],view,live)
        if geom is None:
            return 'tracking_deferred',None
        boxes=query_boxes(name)
        if not boxes:
            return 'no_eligible_candidates',None
        depth,cam,pose=geom
        seq=[np.isfinite(depth[max(0,int(b)):int(np.ceil(d)),max(0,int(a)):int(np.ceil(c))]).sum()
             for a,b,c,d in boxes]
        if min(seq)<9:
            return 'insufficient_candidate_depth',None
        return 'geometry_pair_ready',model.encode(live,boxes,depth,cam,pose)

    def choose(src,index,live):
        decision,=model.decide(model.score(src,live)[index:index+1])
        return decision['state'],decision['index']

    def grade(name, decision, uid):
        # The query instance labels are used only after both arms have predicted.
        state,index=decision
        if index is None:
            return state
        boxes=query_boxes(name)
        lab=label_boxes(boxes,frames[name]['boxes'])[index]
        return 'correct' if lab and lab['instance']==uid else 'wrong' if lab else 'unverified'

    records=[]
    for t in cases:
        s,q,uid=t['source'],t['query'],t['uid']
        # Fixed pre-registered source (IoU >= .5 to its known instance). Never supply target truth for the query.
        src_ids=[a['instance'] if a else None for a in label_boxes(dino[s],frames[s]['boxes'])]
        state,live=tracked(q)
        paired=state!='historical_anchor_missing'
        out=dict(kind=t['kind'],house=t['house'],paired=paired,geo=state,registered=uid in src_ids)
        if uid in src_ids:
            source=truth_frame(s,True)
            i=src_ids.index(uid)
            pred_sparse=choose(source,i,live) if live is not None else (state,None)
            if query_boxes(q):
                pred_truth=choose(source,i,truth_frame(q,False))
            else:
                pred_truth=('no_eligible_candidates',None)
            out.update(sparse=grade(q,pred_sparse,uid),dense=grade(q,pred_truth,uid))
        else:
            out.update(sparse='missing_source_registration',dense='missing_source_registration')
        records.append(out)

    report={}
    for kind in ('same','absent_with_same_class'):
        subset=[r for r in records if r['kind']==kind]
        paired=[r for r in subset if r['paired']]
        def counts(group):
            return dict(cases=len(group),houses=len({r['house'] for r in group}),
                        registered_source=sum(r['registered'] for r in group),
                        tracker=dict(Counter(r['geo'] for r in group)),
                        tracked_head=dict(Counter(r['sparse'] for r in group)),
                        dense_head=dict(Counter(r['dense'] for r in group)),
                        dense_to_tracked=dict(Counter(r['dense']+' -> '+r['sparse'] for r in group)))
        report[kind]=dict(total_cases=len(subset),without_near_anchor=len(subset)-len(paired),
                          all_dense_oracle_outcomes=dict(Counter(r['dense'] for r in subset)),
                          paired=counts(paired),both_registered_and_tracking_gate_passed=counts([
                              r for r in paired if r['registered'] and r['geo']=='geometry_pair_ready']))
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    doc=dict(paid_calls=0,checkpoint_sha256=sha(project/'identity_stage3_model/train/private/jepa.pt'),
             panel_sha256=sha(split/'cases.jsonl'),dino_sha256=sha(split/'dino.jsonl'),
             scope='12 held-out HSSD houses; pre-registered DINO reference selected by source-only annotation; same detector, original eligible_boxes and frozen JEPA API in both arms. Every eligible live box needs 9 finite transferred pixels in the tracked arm. Controlled nearby historical camera with exact past map depth and pose. Current image and fixed calibration only for tracking. Dense upper bound uses simulator query geometry. Missing near anchors excluded from paired denominators. Same threshold and rivals; correct/wrong/unverified are selected boxes, different/uncertain abstain. No navigation or reporting score.',
             note='Tracking counts here are for all eligible (size-gated and rounded) navigation rectangles. The tracking-only transfer.py table used all raw DINO rectangles.',
             results=report)
    path=root/'identity_transfer.json'
    with path.open('x') as f:
        json.dump(doc,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    run(Path(sys.argv[1]).resolve())
