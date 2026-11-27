"""Offline observation-pair evaluation; never send a network request or navigation action."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGE8 = ROOT / 'identity_stage8_relook'
CACHE = HERE / 'private_spatial_replay'


def load(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line]


def clock_files():
    study = ROOT / 'mapping_runtime/pilots/identity_stage8_weak_relook'
    ticket, = (study/'comparison/accepted/second').glob('*.json')
    result = Path(load(ticket)['result'])
    assert hashlib.sha256(result.read_bytes()).hexdigest() == load(ticket)['result_sha256']
    mapper = study.parents[1]/'runs'/load(study/'mapper_process.json')['run']
    return study, result, mapper


def restore_maps():
    """Recompute the original native graph through each capture's first available batch."""
    import torch
    sys.path.insert(0, str(ROOT/'private_sources/MOST-Nav'))
    from mapping.server import MappingServer
    native_hub = torch.hub.load
    def local_hub(repo, model, *args, **kwargs):
        if repo in ('facebookresearch/dinov2', 'facebookresearch/dinov2:main'):
            repo = str(Path(os.environ['TORCH_HOME'])/'hub/facebookresearch_dinov2_main')
            kwargs['source'] = 'local'
        return native_hub(repo, model, *args, **kwargs)
    torch.hub.load = local_hub
    study, result, mapper = clock_files()
    cfg = load(study/'comparison/lock.json')
    orig = ROOT/'private_sources/MOST-Nav/mapping/server.py'
    assert hashlib.sha256(orig.read_bytes()).hexdigest() == cfg['files'][str(orig)]
    keys = {int(p.stem[6:]): p for p in (mapper/'mapping/keyframes').glob('frame_*.png')}
    actions = rows(result.parent/'agent/action_trace.jsonl')
    log = (mapper/'server.log').read_text()
    first, = re.findall(r'同步提交尾部子图 #1, (\d+) 帧', log)
    windows = {1: sorted(keys)[:int(first)]}
    for time, number, size in re.findall(r'\[server\] (\d\d:\d\d:\d\d) 启动子图 #(\d+), (\d+) 帧', log):
        if int(number)>9: break
        event, = [a for a in actions if a['t'][-8:]==time and a['mapping']['busy']
                  and a['mapping']['queued_keyframes']==1 and a['mapping']['is_keyframe']]
        stop = event['mapping_frame_id']
        assert keys[stop].is_file()
        windows[int(number)] = [i for i in sorted(keys) if i<=stop][-int(size):]
    assert list(windows)==list(range(1,10)) and 150 in windows[8] and 170 in windows[9]
    CACHE.mkdir(mode=0o700, exist_ok=True)
    args = SimpleNamespace(submap_size=16,overlapping_window_size=1,max_loops=1,
        min_disparity=40,max_keyframe_interval=3,conf_threshold=25.,lc_thres=.95,
        keyframe_dir=str(CACHE/'unused_keys'),diag_dir=str(CACHE/'unused_diag'),no_semantic=True,vis=False)
    server = MappingServer(args)
    assert server.use_loop_closure
    for rev, ids in windows.items():
        server._process_submap([str(keys[i]) for i in ids],release_solver_lock=False)
        # get_latest_submap() may return a 2-frame loop-closure edge. Check
        # the latest ordinary submap rather than that service status counter.
        ordinary=[m for m in server.solver.map.ordered_submaps_by_key() if not m.get_lc_status()]
        assert list(ordinary[-1].get_frame_ids())==ids
        if rev==1:
            np.save(CACHE/'first_poses.npy',ordinary[0].get_all_poses_world(server.solver.graph))
            if all((CACHE/f'map_{rev}_{k}.npz').exists() for rev,k in [(8,150),(9,170)]):
                return
        if rev not in (8,9):continue
        meta, blob=server.get_frame_points(8)
        wanted = {13, 140, 150 if rev==8 else 170}
        saved=set()
        offset=0
        for rec in meta['frames']:
            n=rec['h']*rec['w']; k=rec['frame_id']
            end=offset+12*n
            if k in wanted:
                xyz=np.frombuffer(blob[offset:end],np.float32).reshape(n,3)
                colors=np.frombuffer(blob[end:end+3*n],np.uint8).reshape(n,3)
                np.savez_compressed(CACHE/f'map_{rev}_{k}.npz',frame_id=k,points=xyz,
                    pose=rec['pose'],colors=colors,rows=np.repeat(np.arange(rec['h'])*8,rec['w']))
                saved.add(k)
            offset=end+3*n
        assert offset==len(blob) and wanted==saved
    print('rebuilt both clock map revisions')


def delta_summary(q,r):
    if q is None or r is None:
        return dict(current_anchor=q is not None,reference_anchor=r is not None)
    delta=np.asarray(q.center)-r.center
    result=dict(separation_m=round(float(np.linalg.norm(delta)),3),
        camera_translation_m=round(float(np.linalg.norm(np.asarray(q.camera)-r.camera)),3),
        source_uncertainty_m=round(float(r.sigma),3),current_uncertainty_m=round(float(q.sigma),3),
        combined_uncertainty_m=round(float(np.hypot(q.sigma,r.sigma)),3),
        source_surface=bool(r.surface),current_surface=bool(q.surface))
    if q.surface and r.surface:
        normal=np.array(q.surface.normal); other=np.array(r.surface.normal)
        result['surface_normal_disagreement_deg']=round(float(np.degrees(np.arccos(np.clip(normal@other,-1,1)))),2)
    return result


def audited_anchor(view,rgb,box,**kw):
    """Inspect frozen extractor intermediates without altering its output or thresholds."""
    from spatial_identity import make_anchor
    collected={}
    def trace(frame,event,arg):
        if frame.f_code is not make_anchor.__code__:return None
        if event=='return':
            v=frame.f_locals
            if 'core' in v:collected['finite_core_samples']=int(v['core'].sum())
            if 'near' in v:collected['near_depth_samples']=int(v['near'].sum())
            if arg is not None:
                collected['sigma_terms_m']=dict(base=.035,depth_mad=round(float(1.48*v['mad']),4),
                    grid=round(float(.40*v['stride']*v['pixel_m']),4),
                    center_or_clipping=round(float((.12 if v['clipped'] else .045)*max(v['extent'])*v['pixel_m']),4))
        return trace
    old_trace=sys.gettrace()
    try:
        sys.settrace(trace)
        value=make_anchor(view,rgb,box,**kw)
    finally:sys.settrace(old_trace)
    return value,collected


def clock(recovery=False):
    sys.path.insert(0,str(STAGE8))
    from agents import navigator
    from appearance import _rgb, build_reference
    from matcher_after_delay import archived_frame_reader
    from proposal_worker import ProposalWorker
    from spatial_identity import associate,make_anchor
    from visual_witness import compare_regions,describe,strong_object
    from optics import skew_view
    from map_views import rgb_aligned,point_in_box
    from weak_route_agent import grounded_region
    from trusted_reference import assess_reference
    from local_match import match_prepared,prepare_reference
    from verify_rank import deny_ip_sockets
    if recovery:from spatial_recovery import ReferenceView,OwnerView,recognize_seen
    deny_ip_sockets()
    study,result,mapper=clock_files()
    ep,=load(result)['episodes']
    log=rows(result.parent/'agent/confirm_events.jsonl')
    dec=rows(result.parent/'agent/decision_trace.jsonl')
    obs,=[o for o in ep['paper']['agent_trace']['observations'] if o['observation_id']==1]
    assert obs['frame_id']==13
    files={int(p.stem[6:]):p for p in (mapper/'mapping/keyframes').glob('frame_*.png')}
    mapped=archived_frame_reader(load(study/'comparison/lock.json'),files)
    old=mapped(13)
    ref=build_reference(old,obs['bbox'],point_1000=obs['pixel'])
    oldbox=np.array(ref.roi)*np.tile(old.shape[1::-1],2)
    model=ProposalWorker(ROOT/'b_verification/env/bin/python',ROOT,'cuda')
    align=navigator.gravity_alignment(np.load(CACHE/'first_poses.npy'),cam_up=navigator.mount_compensated_cam_up())
    report=[]
    try:
        for c in load(STAGE8/'relook_capture_audit.json')['captures']:
            k,step,rev=c['frame_id'],c['capture_step'],c['later_successful_submap']
            ctx=[d for d in dec if d['step']<=step][-1]['agent_snapshot']
            scale=ctx['metric_transform']['scale']
            node,=[n for n in ctx['instances'] if n['instance_id']==3]
            snaps={f:dict(np.load(CACHE/f'map_{rev}_{f}.npz')) for f in [13,k]}
            native=np.array(Image.open(files[k]).convert('RGB')); small=mapped(k)
            registered=rgb_aligned(snaps[k],small,8)
            native_ok=rgb_aligned(snaps[k],native,8)
            assert registered and rgb_aligned(snaps[13],old,8)
            rgb=native if native_ok else small
            boxes=model.detect(rgb,'distinct clock objects')
            boxes=[d for d in boxes if d['score']>=.70]
            view,sv=[skew_view(snaps[f],scale) for f in [k,13]]
            bound=grounded_region(view,rgb,boxes,np.array(node['point']),align,scale) if view else None
            # Separate the identity subgate from the registered-node bounding gate.
            reference,source_quality=audited_anchor(sv,old,oldbox,frame_id=13)
            rr=[]
            for box in boxes:
                b=np.array(box['bbox'])
                q,query_quality=audited_anchor(view,rgb,b,frame_id=k)
                strong=strong_object(compare_regions(describe(old,oldbox),describe(rgb,b)))
                d=associate(q,{1:[reference]},appearances={1:'strong' if strong else 'unknown'})
                raw=match_prepared(prepare_reference(old,oldbox),rgb,[b])==0
                offtrack={}
                if recovery:
                    owners={n['instance_id']:OwnerView(tuple(align.T@(np.array(n['point'])*scale)),n['reported'],
                              (ReferenceView(reference,bool(strong),raw),) if n['instance_id']==1 and reference is not None else ())
                              for n in ctx['instances'] if n['instance_id']!=node['instance_id']}
                    rec=recognize_seen(q,align.T@(np.array(node['point'])*scale),owners)
                    offtrack={'offtrack_sighting':rec.status,'offtrack_reason':rec.reason}
                quality=assess_reference(rgb,b,(b[:2]+b[2:])/2,boxes)
                world=point_in_box(view,b*np.tile(np.array(view['grid'].shape[1::-1])*view['stride']/rgb.shape[1::-1],2)) if view else None
                offsets={}
                if world is not None:
                    for record in ctx['instances']:
                        key='chased_clue' if record['instance_id']==node['instance_id'] else 'confirmed_owner' if record['reported'] else 'other_clue'
                        offsets[key]=round(float(np.linalg.norm(align@world-np.array(record['point'])*scale)),3)
                rr.append(dict(**delta_summary(q,reference),registered_offsets_m=offsets,
                    current_localization=query_quality,reference_localization=source_quality,strong_object=bool(strong),
                    visual_only_match=raw,**offtrack,
                    single_owner_subgate=d.status,reason=d.reason,reference_eligible=quality.eligible()))
            report.append(dict(step=step,recomputed_at=c['processed_by_step'],native_color_alignment=native_ok,
                       box_count=len(boxes),original_registration_binding=bool(bound is not None),
                       bound_residual_m=None if bound is None else round(bound[0],3),evidence=rr))
    finally:model.close()
    return dict(scope='Fixed trajectory; original RGB keyframes run through frozen VGGT-SLAM, original chronological submap boundaries, loops and pinned original reference. No new navigation. No oracle inputs.',results=report)


def hssd(recovery=False):
    """One pair per independently detected source/query box in each frozen view-pair."""
    from collections import Counter,defaultdict
    from functools import lru_cache
    sys.path[:0]=[str(ROOT/n) for n in ('jepa_reid_v21/association','identity_stage6_approach','identity_stage7_transfer')]
    from evaluate import label_boxes
    from visual_witness import describe,compare_regions,strong_object
    from local_match import prepare_reference,match_prepared
    from spatial_identity import make_anchor,associate
    if recovery:from spatial_recovery import ReferenceView,OwnerView,recognize_seen
    base=ROOT/'identity_stage3_model/new_houses/private'
    frames={r['name']:r for r in rows(base/'frames.jsonl')}
    boxes={r['name']:r['boxes'] for r in rows(base/'dino.jsonl')}
    cohort=sorted({(r['source'],r['query']) for r in rows(base/'cases.jsonl')})
    @lru_cache(maxsize=155)
    def features(k):
        f=frames[k];T=np.array(f['T']);K=np.array(f['K'])
        im=np.array(Image.open(base/'observations'/(k+'.png')).convert('RGB'))
        with np.load(base/'observations'/(k+'.npz')) as archive: depth=archive['depth']
        # Simulated registered RGB-D sensor & known CV camera transform: useful
        # as a place-matching control, never represented as monocular SLAM.
        assert np.issubdtype(depth.dtype,np.floating) and depth.shape==im.shape[:2]
        y,x=np.mgrid[0:depth.shape[0]:4,0:depth.shape[1]:4]
        grid=((np.stack([x+.5,y+.5,np.ones_like(x)],axis=-1)@np.linalg.inv(K).T)
              *depth[::4,::4,None])@T[:3,:3].T+T[:3,3]
        view=dict(grid=grid,K=K,c2w=T,stride=4)
        return im,[(make_anchor(view,im,b,frame_id=int(k[:14],16)),prepare_reference(im,b)) for b in boxes[k]]
    scores=defaultdict(Counter)
    wrong_houses=set()
    for s,q in cohort:
        src,sa=features(s); query,qa=features(q)
        local=[]
        track=None
        if recovery:
            # Construct a deterministic third-view pursuit from actual sensor detections.
            # There is no recorded autonomous controller in this HSSD fixture.
            candidates=sorted((fid for fid,f in frames.items() if fid not in (s,q)
                and f['house']==frames[s]['house'] and f['category']==frames[s]['category']),
                key=lambda f:hashlib.sha256(('third-view-v1'+s+q+f).encode()).digest())
            for fid in candidates:
                cands=[anchor.center for anchor,_ in features(fid)[1] if anchor is not None]
                if cands:
                    track=cands[0];break
        for i,(source,template) in enumerate(sa):
            for j,(current,other) in enumerate(qa):
                raw=match_prepared(template,query,[boxes[q][j]]) is not None
                strong=strong_object(compare_regions(template[1],other[1]))
                d=associate(current,{1:[source]},appearances={1:'strong' if strong else 'unknown'})
                inc=None
                if recovery:
                    owners={}
                    for k,(anc,memory) in enumerate(sa):
                        references=() if anc is None else (ReferenceView(anc,bool(strong),raw),) if k==i else ()
                        owners[k]=OwnerView(anc.center if anc else (np.nan,)*3,k==i,references)
                    inc=recognize_seen(current,track,owners).status
                local.append((i,j,raw,d.status,inc))
        # Only the evaluator opens instance annotations, after every decision.
        ls=label_boxes(boxes[s],frames[s]['boxes']);lq=label_boxes(boxes[q],frames[q]['boxes'])
        for i,j,raw,pred,inc in local:
            a,b=ls[i],lq[j]
            key=('unlabeled' if not a or not b else
                 'same' if a['instance']==b['instance'] else 'different')
            c=scores[key];c['pairs']+=1;c['visual_match']+=int(raw);c[pred]+=1
            c['visual_and_spatial_match']+=int(raw and pred=='same')
            c['visual_or_spatial_match']+=int(raw or pred=='same')
            if recovery:c['offtrack_'+inc]+=1
            if key=='different' and pred=='same':wrong_houses.add(frames[s]['house'])
            if key=='different' and a.get('asset')==b.get('asset'):
                t=scores['different_copy_of_same_asset'];t['pairs']+=1;t['visual_match']+=int(raw);t[pred]+=1
                if recovery:t['offtrack_'+inc]+=1
                m,n=sa[i][0],qa[j][0]
                if m and n and m.surface and n.surface:
                    # HSSD uses +Y gravity. Identify same-wall-like observations
                    # from sensed planes only; no architectural wall IDs available.
                    na=np.array(m.surface.normal);nb=np.array(n.surface.normal)
                    dv=np.array(m.center)-n.center
                    if (max(abs(na[1]),abs(nb[1]))<.25 and na@nb>np.cos(np.radians(15)) and
                            abs(dv@na)<.15 and abs(dv@nb)<.15 and np.linalg.norm(dv)<2.0):
                        t=scores['different_copy_nearby_parallel_vertical_support']
                        t['pairs']+=1;t['visual_match']+=int(raw);t[pred]+=1
                        if recovery:t['offtrack_'+inc]+=1
    return dict(scope='Observed proposals in frozen same-category view pairs in 12 HSSD houses; sensor depth and supplied camera poses. All source and query proposals considered; no labels or target poses used by matcher. Pairwise diagnosis, not a navigation success rate.',unique_view_pairs=len(cohort),
        houses=len({frames[s]['house'] for s,q in cohort}),false_merge_houses=len(wrong_houses),counts={k:dict(v) for k,v in scores.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode')
    args=parser.parse_args()
    if args.mode=='maps':restore_maps()
    elif args.mode in ('clock','hssd','clock_recovery','hssd_recovery'):
        rec=args.mode.endswith('_recovery')
        result=(clock if args.mode.startswith('clock') else hssd)(rec)
        if rec:
            import spatial_recovery
            result['recovery_sha256']=hashlib.sha256(Path(spatial_recovery.__file__).read_bytes()).hexdigest()
            import inspect
            result['recovery_logic_sha256']=hashlib.sha256((inspect.getsource(spatial_recovery._joint_place)+inspect.getsource(spatial_recovery.recognize_seen)).encode()).hexdigest()
            if args.mode.startswith('hssd'):
                result['offtrack_protocol']='Incidental sighting subgate. Each source proposal in turn is a reported owner, other localized source detections compete. A pursuit point is the first available detector anchor in a fixed hash-ordered third same-house/category view. No true object IDs select a viewpoint, detection or point. Different from autonomous operation; frozen HSSD box archive does not include confidence.'
        result['matcher_sha256']=hashlib.sha256((HERE/'spatial_identity.py').read_bytes()).hexdigest()
        (HERE/('replay_'+args.mode+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(result,flush=True)
