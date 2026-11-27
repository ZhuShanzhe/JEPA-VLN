"""Join two expired reobservations to the mapper's FIFO and completed SLAM batches."""
from pathlib import Path
import json
import hashlib
import re


def load(p):
    return json.loads(p.read_bytes())


def rows(p):
    return [json.loads(line) for line in p.read_bytes().splitlines() if line]


def digest(b):
    return hashlib.sha256(b).hexdigest()


def summary(project, study):
    mapper=project/'mapping_runtime/runs'/load(study/'mapper_process.json')['run']
    started,=(study/'comparison/attempts/second').glob('*/*/started.json')
    run=project/'mapping_runtime/runs'/load(started)['run_id']
    agent=run/'agent'
    manifest,=(mapper/'mapping/diagnostics').glob('*_frame_captions.jsonl')
    m={r['frame_id']:r for r in rows(manifest) if r['event']=='frame_saved'}
    decisions=rows(agent/'confirm_events.jsonl')
    actions={r['step']:r for r in rows(agent/'action_trace.jsonl')}
    detections=rows(agent/'multiview_events.jsonl')
    serverlog=(mapper/'server.log').read_text()
    launches=list(re.finditer(r'^\[server\] (\d\d:\d\d:\d\d) 启动子图 #(\d+), (\d+) 帧, 待处理 (\d+) 帧',serverlog,re.M))
    evidence=[]
    for row in [r for r in decisions if r['event']=='weak_relook_capture']:
        start=row['step']
        fid=row['frame']
        end=next(r['step'] for r in decisions if r['step']>=start and r['instance']==row['instance']
                 and r['event']=='weak_relook_end')
        assert all(not r['event']=='weak_relook_evidence' for r in decisions if start<=r['step']<=end)
        assert actions[start]['mapping_frame_id']==fid and actions[start]['mapping']['is_keyframe']
        assert m[fid]['keyframe_reason']=='flow'
        assert all(not actions[t]['mapping']['busy'] for t in range(start,end+1))
        launch=next(t for t in range(end+1,max(actions)+1) if
            actions[t]['mapping']['queued_keyframes']<actions[t-1]['mapping']['queued_keyframes'])
        matching=[(i,l) for i,l in enumerate(launches) if l.group(1)==actions[launch]['t'][-8:]]
        (idx,l),=matching
        size=int(l.group(3))
        assert size==actions[launch-1]['mapping']['queued_keyframes']+1
        last=actions[launch]['mapping_frame_id']
        members=sorted(i for i,r in m.items() if i<=last and r['is_keyframe'])[-size:]
        assert fid in members and actions[start]['mapping']['queued_keyframes']==members.index(fid)+1
        block=serverlog[l.end():launches[idx+1].start() if idx+1<len(launches) else None]
        assert '子图完成' in block and '子图处理失败' not in block
        idle=next(t for t in range(launch,max(actions)+1) if not actions[t]['mapping']['busy'])
        assert idle>launch
        checks=[r for r in detections if r['event']=='boxes' and r['step']==start and r['source']=='live']
        assert checks and all(not r['worker_failed'] for r in checks)
        pg=mapper/'mapping/keyframes'/('frame_%06d.png'%fid)
        source_bytes=json.dumps([m[fid],row,*[actions[t] for t in range(start,idle+1)]],
                                separators=(',',':'),sort_keys=True).encode()
        evidence.append(dict(capture_step=start,frame_id=fid,episode_source='native_flow_keyframe',
            detector_box_count=checks[-1]['count'],detector_failure=False,
            capture_queue_depth=actions[start]['mapping']['queued_keyframes'],end_step=end,
            end_queue_depth=actions[end]['mapping']['queued_keyframes'],fifo_batch_size=size,
            launch_step=launch,processed_by_step=idle,steps_after_relook_closed=idle-end,
            later_successful_submap=int(l.group(2)),exact_keyframe_in_fifo_window=True,
            keyframe_rgb_saved=pg.exists(),keyframe_rgb_sha256=digest(pg.read_bytes()) if pg.exists() else None,
            selected_log_records_sha256=digest(source_bytes)))
    assert len(evidence)==2
    return dict(scope='Retrospective verification of the two observed weak-route captures at steps 149 and 169 using immutable prefixes of the live logs. No live mapper queries.',
        captures=evidence,observed='Both actual detector lookups returned proposals. Both flow-selected RGB keyframes belonged to regular 17-frame FIFO batches that completed after the relook controller discarded its pending frame IDs.',
        geometry='Regular completed submaps supply a per-frame point grid and pose under get_frame_points. No direct per-pixel depth snapshot or cross-view identity verdict was archived for these two crops. Finite depth in the relevant object, correct identity, and the effect of waiting longer remain unverified.',
        mechanism='The snapshot cache is per agent step. The controller requested snapshots across several steps, but the batches had not launched before its local timeout. Neither capture used the optional force-capture path.')


if __name__=='__main__':
    here=Path(__file__).resolve().parent
    project=here.parent
    data=summary(project,project/'mapping_runtime/pilots/identity_stage8_weak_relook')
    path=here/'relook_capture_audit.json'
    if path.exists():
        assert load(path)==data
    else:
        path.write_text(json.dumps(data,indent=2)+'\n')
    print(data['captures'])
