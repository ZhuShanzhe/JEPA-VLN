"""Paired MOST ManyON execution, durable acceptance, and official complete-case scoring."""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl, hashlib, json, os, re, subprocess, sys, tempfile, uuid
from pathlib import Path

RT=Path(__file__).resolve().parent.parent
BASE=RT.parent
SCENES=BASE/'frontier_scenes'
BENCH=SCENES/'most_release'
REPO=BASE/'private_sources/MOST-Nav'
LAUNCH=RT/'harness/run_pipeline.sh'
MANIFEST=SCENES/'acceptance/most_many_category_attribute_frozen.jsonl'
PANEL_HASH='48f0f21020b3038121efd48e4f6dc880e47d3d1420c3a54745b3b55301d5cd3d'
MODEL='gpt-4o-2024-11-20'
ARMS=('native','second')
sys.path.insert(0,str(BENCH))
from evaluation.main.metrics import aggregate_metrics

def require(ok,msg):
    if not ok:raise ValueError(msg)
def data(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def digest(p):
    with Path(p).open('rb') as fp:return hashlib.file_digest(fp,'sha256').hexdigest()
def encode(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
def write(p,x):
    # Link is atomic and cannot replace a previously committed result.
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=p.parent,delete=False) as tmp:
        tmp.write(encode(x));tmp.flush();os.fsync(tmp.fileno())
    try:os.link(tmp.name,p)
    finally:os.unlink(tmp.name)
def scene_inventory():
    hashes=data(SCENES/'acceptance/hm3d_val_file_hashes.json')
    root=SCENES/'hm3d_v02'
    paths=sorted(hashes)
    for name in paths:
        fp=root/name; require(fp.is_file(),'missing scene '+name)
    assets={}
    for name in paths:
        stat=(root/name).stat()
        assets[name]=[stat.st_ino,stat.st_size,stat.st_mtime_ns]
    return assets
def config(path):
    raw=data(path)
    needed=('agent','source','adapter_root','checkpoints','config','environment')
    require(set(needed)<=set(raw)<=set(needed)|{'module_roots'},'second-arm spec fields must equal '+str(needed))
    require(re.fullmatch(r'[a-zA-Z_]\w*(?:\.\w+)*:[a-zA-Z_]\w*',raw['agent']),'invalid module:Class')
    require(raw['agent']!='agents.nav_agent:NavAgent','second arm must be distinct')
    root=Path(raw['adapter_root']).resolve(strict=True)
    module=raw['agent'].split(':')[0]
    require(Path(raw['source']).resolve(strict=True)==root.joinpath(*module.split('.')).with_suffix('.py'),'adapter root/source mismatch')
    require(raw['checkpoints'] and raw['config'],'pin checkpoint(s) and treatment configuration')
    for group in ('config','checkpoints'):
        require(isinstance(raw[group],list),group+' must be a path list')
        for p in raw[group]:require(Path(p).is_absolute() and Path(p).is_file(),'missing '+str(p))
    require(all(Path(p).is_absolute() and Path(p).is_dir() for p in raw.get('module_roots',[])), 'invalid scorer module root')
    env=raw['environment']
    for k,v in env.items():
        require((k=='MOST_EXEC_SCORER' or re.fullmatch(r'(MOST_FRONTIER_|JEPA_)[A-Z0-9_]+',k)) and not re.search(r'KEY|TOKEN|SECRET|PASSWORD',k) and isinstance(v,str),'only non-secret scorer settings allowed')
    if raw['agent']=='execute_frontier:FrontierSelector': require(env.get('MOST_EXEC_SCORER'),'explicit execution scorer is required')
    scorer_sources(raw)
    return raw

def scorer_sources(spec):
    roots=[RT/'probes',RT/'harness',REPO,BENCH/'evaluation/main',BENCH,Path(spec['adapter_root']),*(Path(x) for x in spec.get('module_roots',[]))]
    found=[]
    for key in ('MOST_EXEC_SCORER','MOST_FRONTIER_SCORER'):
        callback=spec['environment'].get(key)
        if callback is None: continue
        require(re.fullmatch(r'[a-zA-Z_]\w*(?:\.\w+)*:[a-zA-Z_]\w*',callback),'invalid callback import path')
        module=Path(*callback.split(':')[0].split('.'))
        matches=[]
        for root in roots:
            for candidate in (root/module/'__init__.py',(root/module).with_suffix('.py')):
                if candidate.is_file() and candidate.resolve() not in matches: matches.append(candidate.resolve())
        require(len(matches)==1,'missing or ambiguous scoring module: '+callback)
        found.extend(matches)
    return found
def panel():
    require(digest(MANIFEST)==PANEL_HASH,'frozen manifest checksum mismatch')
    rows=lines(MANIFEST)
    require(len({r['episode_id'] for r in rows})==300 and Counter(r['condition'] for r in rows)=={'many_category':150,'many_attribute':150},'invalid frozen panel')
    return rows
def pinned_files(spec):
    paths=[LAUNCH,Path(__file__),RT/'weights.env',RT/'weights/verified_weights.json',RT/'harness/start_mapping.py',
           BASE/'ops/hm3d_val_env.sh',BASE/'private_runtime/most_vlm/activate.sh',MANIFEST,SCENES/'acceptance/hm3d_val_file_hashes.json',
           BENCH/'evaluation/main/hm3d_config.yaml',RT/'env_build/acceptance.json']
    paths.extend([RT/'harness/execute_frontier.py',RT/'probes/most_frontier_probe.py',*scorer_sources(spec)])
    paths+=list(BENCH.joinpath('evaluation').rglob('*.py'))
    paths+=[RT/'env_build/wheels.sha256',RT/'env_build/activate_mapping.sh',RT/'env/lib/python3.12/site-packages/zz_mapping_paths.pth',RT/'env/lib/python3.12/site-packages/00_mapping_abi.pth']
    paths+=list((REPO/'agents').rglob('*.py'))+list((REPO/'mapping').rglob('*.py'))+list((REPO/'decision').rglob('*.py'))
    paths+=list((REPO/'VGGT-SLAM/vggt_slam').rglob('*.py'))+[REPO/'runtime_paths.py']
    paths+=list((RT/'frozen_manyon').glob('*/*.json'))
    for name in ('salad/salad','VGGT_SPARK/vggt'):
        paths+=list((RT/'env_build/src'/name).rglob('*.py'))
    paths+=[Path(p) for p in [spec['source'],*spec['config'],*spec['checkpoints']]]
    return {str(p):digest(p) for p in sorted(set(paths))}
def verify_weights():
    weights=RT/'weights'
    for desc in data(weights/'verified_weights.json').values():
        entries=desc['files'] if 'files' in desc else {desc['local']:desc}
        for name,meta in entries.items():
            require(digest(weights/name)==meta['sha256'],'weight drift: '+name)
def initialize(out,spec_path):
    rows=panel();spec=config(spec_path);verify_weights()
    lock={'schema':1,'namespace':'formal_frozen','panel_sha256':PANEL_HASH,'model':MODEL,
          'policy_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
          'protocol':{'action_protocol':'goat','threshold':0.25,'goal_type':'description','instruction_policy':'first','seed':0,'budget':300},
          'weights_receipt_sha256':digest(RT/'weights/verified_weights.json'),'second':spec,'files':pinned_files(spec),'episodes':rows,'scene_files':scene_inventory()}
    out.mkdir(parents=True,exist_ok=False);write(out/'lock.json',lock)
    print('Locked 300 episodes, two arms, no evaluations executed.')
def verify_lock(out,full=True):
    lock=data(out/'lock.json')
    require(lock['namespace']=='formal_frozen' and lock['episodes']==panel() and lock['model']==MODEL,'formal campaign identity mismatch')
    require(lock['scene_files']==scene_inventory(),'scene assets changed since initialization')
    require(lock['files']==pinned_files(lock['second']),'code, input, configuration or checkpoint drift')
    if full:verify_weights()
    return lock
@contextmanager
def mutex(path):
    with path.open('a+') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield fd.fileno()
def check_health(state):
    s=state['semantic']
    require(all([state.get('ok'),state.get('sam_available'),state.get('loop_closure_enabled'),s.get('caption_enabled'),s.get('embedding_enabled')]),'mapping, caption, embedding, SAM or loop closure unavailable')
    return {'sam':True,'loop':True,'caption':True,'embed':True,'pointing':bool(s.get('pointing_enabled')),'caption_errors':int(s['caption_errors']),'caption_mode':s['caption_mode']}
def snapshot(port,drain=False):
    sys.path.insert(0,str(REPO))
    from mapping.client import MappingClient
    client=MappingClient(port=port,timeout=30)
    try:
        if drain:
            require(client.wait_idle(timeout=180) and client.wait_captions(timeout=180),'pending mapping/caption jobs')
        return check_health(client.get_state())
    finally:client.close()
def service(port):
    # Resolve the actual listener. A different process writing to the named log is insufficient.
    socket_ids=set()
    for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
        fields=line.split()
        if int(fields[1].split(':')[-1],16)==port and fields[3]=='0A':socket_ids.add('socket:['+fields[9]+']')
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if not (str(RT/'harness/start_mapping.py').encode() in (proc/'cmdline').read_bytes().split(b'\0')):continue
            if not any(os.readlink(d) in socket_ids for d in (proc/'fd').iterdir()):continue
            env=dict(x.split('=',1) for x in (proc/'environ').read_bytes().decode().strip('\0').split('\0') if '=' in x)
            require(env.get('NAV_CAPTION_API_MODEL')==MODEL,'caption model differs')
            run=env['NAV_RUN_ID'];folder=RT/'runs'/run
            log=folder/'server.log'
            require(Path(os.readlink(proc/'fd/1')).samefile(log),'server log binding mismatch')
            safe={k:v for k,v in env.items() if (k.startswith(('NAV_','VGGT_','MOST_')) or k=='TORCH_HOME') and k not in ('NAV_DEBUG_ROOT','NAV_RUN_ID') and not re.search('KEY|TOKEN|SECRET|PASSWORD',k)}
            require(env.get('VGGT_MODEL_CKPT')==str(RT/'weights/torch/hub/checkpoints/model.pt') and env.get('NAV_EMBED_MODEL_PATH')==str(RT/'weights/bge-m3') and env.get('NAV_SAM_CKPT')==str(RT/'weights/sam_vit_h_4b8939.pth'),'active weights differ')
            return folder,safe
        except (FileNotFoundError,ProcessLookupError):continue
    raise ValueError('expected original loopback mapping service missing')
def capabilities(health):
    return {k:v for k,v in health.items() if k!='caption_errors'}
def optional_tail(path,mark):
    return '' if mark==(None,0) and not path.exists() else tail(path,mark)
def caption_health(before,after,diagnostics):
    require(capabilities(before)==capabilities(after),'mapping capabilities changed mid-episode')
    failed=[x for x in diagnostics if x.get('event')=='caption_result' and x.get('status')=='failed']
    delta=after['caption_errors']-before['caption_errors']
    require(delta==len(failed),'caption errors have incomplete attribution')
    # A real model returning an empty caption is a policy outcome. Transport or encoder failure is infrastructure.
    require(all(str(x.get('error','')).strip()=='caption 为空' for x in failed),'caption transport/embedding runtime failure')
    return len(failed)
def offset(path):
    if not path.exists():return (None,0)
    st=path.stat();return (st.st_ino,st.st_size)
def tail(path,mark):
    require(path.is_file() and (mark[0] is None or path.stat().st_ino==mark[0]) and path.stat().st_size>=mark[1],'evidence log missing/truncated/rotated')
    with path.open('rb') as f:
        f.seek(mark[1]);return f.read().decode(errors='replace')
def protocol(result,row,agent,namespace):
    require(not result.get('journal_errors') and len(result['episodes'])==1,'bad evaluator or journal output')
    e=result['episodes'][0]; args=result['manifest']['arguments']
    require(e['episode_id']==row['episode_id']==args['episode_id'] and args['agent']==agent,'wrong episode or arm')
    if namespace=='formal_frozen':
        require(Path(args['dataset_dir']).resolve()==(RT/'frozen_manyon').resolve() and e['target_mode']=='many' and e['target_count']==row['quota'],'wrong data/quota')
    require(args['action_protocol']=='goat' and args['goal_type']=='description' and args['threshold']==0.25 and args['seed']==0 and args['instruction_policy']=='first' and e['max_steps_budget']==300,'wrong scoring protocol')
    require(not e['timed_out'] and e['failure_reason'] in (None,'incomplete_finish','max_steps','wrong_stop'),'engineering/unknown termination: '+str(e['failure_reason']))
    return e
def validate_trace(rows,model):
    require(any(r['kind']=='decision' for r in rows),'no model decision evidence')
    for r in rows:
        raw=r.get('raw_response')
        require(isinstance(raw,dict) and not r.get('error') and r['model']==raw.get('model')==model,'decision transport/model identity mismatch')
    # A returned model response with an invalid action remains a model outcome.
def accepted_path(out,arm,eid):return out/'accepted'/arm/(eid+'.json')
def accepted(out,lock,arm,row):
    ref=accepted_path(out,arm,row['episode_id'])
    if not ref.exists():return None
    a=data(ref)
    require(a['namespace']==lock['namespace'] and a['lock']==digest(out/'lock.json') and a['arm']==arm and a['episode_id']==row['episode_id'],'mixed campaign/arm')
    src=Path(a['result'])
    require(digest(src)==a['result_sha256'],'result drift')
    agent=lock.get('native_agent','agents.nav_agent:NavAgent') if arm=='native' else lock['second']['agent']
    result=data(src)
    require(result['manifest']['configuration_overrides']['NAV_RUN_ID']==a['run_id'],'wrong result run')
    return protocol(result,row,agent,lock['namespace'])
def commit(out,lock,arm,row,result,runid,evidence):
    eid=row['episode_id']
    require(not accepted_path(out,arm,eid).exists(),'episode already accepted')
    require(bool(evidence),'missing execution evidence')
    if lock['namespace']=='formal_frozen':require(evidence.get('server_run'),'missing service provenance')
    write(accepted_path(out,arm,eid),dict(namespace=lock['namespace'],arm=arm,episode_id=eid,lock=digest(out/'lock.json'),result=str(result.resolve()),result_sha256=digest(result),run_id=runid,evidence=evidence))
def score(out,lock):
    pairs={r:[] for r in ARMS}; groups={r:{} for r in ARMS}; missing={r:[] for r in ARMS}
    aligned=[]
    for row in lock['episodes']:
        e={k:accepted(out,lock,k,row) for k in ARMS}
        for k in ARMS:
            if e[k] is None:missing[k].append(row['episode_id'])
        if any(v is None for v in e.values()):continue
        aligned.append(row['episode_id'])
        for k,v in e.items():
            pairs[k].append(v);groups[k].setdefault(row['condition'],[]).append(v)
    return {'namespace':lock['namespace'],'campaign_sha256':digest(out/'lock.json'),'aligned_ids':aligned,'aligned_n':len(aligned),'total_n':len(lock['episodes']),
            'complete':len(aligned)==len(lock['episodes']),'missing_by_arm':missing,
            'official_metrics':{k:aggregate_metrics(pairs[k]) if pairs[k] else None for k in ARMS},
            'by_condition':{k:{g:aggregate_metrics(v) for g,v in groups[k].items()} for k in ARMS}}

def run_one(out,lock,arm,row):
    eid=row['episode_id']
    if accepted(out,lock,arm,row) is not None:return 'existing'
    rid='paired_'+arm+'_'+uuid.uuid4().hex
    att=out/'attempts'/arm/eid/rid;att.mkdir(parents=True)
    env={k:v for k,v in os.environ.items() if not k.startswith(('NAV_','VGGT_','MOST_','JEPA_')) and k not in ('PYTHONPATH','PYTHONHOME')}
    # Both arms run the same workspace entrypoint. Add only the frozen second-arm settings.
    agent='agents.nav_agent:NavAgent'
    if arm=='second':
        spec=lock['second'];agent=spec['agent'];env.update(spec['environment'])
        env['PYTHONPATH']=os.pathsep.join([spec['adapter_root'],*spec.get('module_roots',[])])
    env.update(MOST_FROZEN_AGENT=agent,MOST_EXPERIMENT_MODEL=MODEL,VGGT_SLAM_PORT='18555',MOST_BATCH_INHERIT_EVAL_FD='1')
    write(att/'started.json',{'episode':eid,'run_id':rid,'arm':arm,'agent':agent,'callback':env.get('MOST_EXEC_SCORER',env.get('MOST_FRONTIER_SCORER')),'lock':digest(out/'lock.json')})
    output=RT/'runs'/rid
    try:
        with mutex(RT/'eval.lock') as fd:
            folder,opts=service(18555)
            optsfile=out/'mapping_settings.json'
            before=snapshot(18555,drain=True)
            locked_service={'environment':opts,'capabilities':capabilities(before)}
            if not optsfile.exists():write(optsfile,locked_service)
            require(data(optsfile)==locked_service,'mapping service settings/capabilities differ from campaign')
            service_log=folder/'server.log'
            caption_log=folder/'mapping/diagnostics/vlm_caption.jsonl'
            frame_errors=folder/'mapping/diagnostics'/(eid+'_frame_captions.jsonl')
            fm=offset(frame_errors)
            gateway=BASE/'private_runtime/most_vlm/requests.jsonl'
            sm,cm,gm=offset(service_log),offset(caption_log),offset(gateway)
            # Preserve the shared lock across controller termination and exec.
            saved=None
            if fd!=9:
                try:saved=os.dup(9)
                except OSError:pass
                os.dup2(fd,9)
            try:
                result=subprocess.run(['bash',str(LAUNCH),'frozen-episode',rid,eid],env=env,pass_fds=(9,),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            finally:
                if fd!=9:
                    if saved is None:os.close(9)
                    else:os.dup2(saved,9);os.close(saved)
            (att/'launcher.log').write_bytes(result.stdout)
            require(result.returncode==0,'evaluator process exited '+str(result.returncode))
            doc=data(output/'result.json')
            protocol(doc,row,agent,lock['namespace'])
            manifest=doc['manifest']
            expected=REPO/'agents/nav_agent.py' if arm=='native' else Path(lock['second']['source'])
            require(manifest['agent']['source_sha256']==digest(expected),'agent source checksum mismatch')
            m=manifest['configuration_overrides'];require(m['NAV_RUN_ID']==rid and m['NAV_VLM_MODEL']==m['NAV_CAPTION_API_MODEL']==MODEL,'evaluator model/run mismatch')
            require(Path(manifest['arguments']['config']).resolve()==(BENCH/'evaluation/main/hm3d_config.yaml').resolve(),'sensor config drift')
            for key in ('NAV_VLM_ENABLED','NAV_REQUIRE_DECISION_PREFLIGHT','NAV_REQUIRE_CAPTION_PREFLIGHT'):require(m.get(key)=='1','VLM disabled')
            after=snapshot(18555,drain=True)
            frame_evidence=[json.loads(s) for s in optional_tail(frame_errors,fm).splitlines() if s.strip()]
            model_caption_errors=caption_health(before,after,frame_evidence)
            again,_=service(18555)
            require(again==folder,'mapping service restarted mid-episode')
            log=tail(service_log,sm)+(output/'evaluator.log').read_text()
            require(not re.search(r'Traceback \(most recent call last\)|CUDA out of memory|CAPTION_BACKEND_UNAVAILABLE|API 不可用，回退确定性决策',log,re.I),'runtime degradation in episode logs')
            dec=lines(output/'agent/vlm_calls.jsonl');validate_trace(dec,MODEL)
            cap=[json.loads(s) for s in optional_tail(caption_log,cm).splitlines() if s.strip()]
            require(all(r.get('episode')==eid and r.get('ok') and r.get('model')==MODEL for r in cap),'caption call unavailable, mixed episode or unexpected model')
            gw=[json.loads(s) for s in tail(gateway,gm).splitlines() if s.strip()]
            gs=[r for r in gw if r.get('event')=='success' and r.get('model')==MODEL]
            require(len(gs)>=len(dec)+sum(not r.get('cache_hit') for r in cap),'incomplete downstream VLM identity evidence')
            require(all(r.get('model')==r.get('reported_model')==MODEL and r.get('status')==200 for r in gs),'relay reported a different model')
            evidence={'server_run':folder.name,'mapping_environment_sha256':digest(optsfile),'health':after,'caption_calls':len(cap),'decision_trace_calls':len(dec),'relay_successes':len(gs),'model_empty_captions':model_caption_errors}
            commit(out,lock,arm,row,output/'result.json',rid,evidence)
        return 'accepted'
    except Exception as exc:
        write(att/'quarantined.json',{'class':type(exc).__name__,'reason':str(exc)[:400],'score_included':False})
        raise
def schedule(out,lock,limit,execute):
    require(limit is None or limit>0,'limit must be positive')
    n=0
    for i,row in enumerate(lock['episodes']):
        for arm in (ARMS if i%2==0 else ARMS[::-1]):
            if (out/'PAUSE').exists():return 'paused',n
            if accepted(out,lock,arm,row) is not None:continue
            execute(out,lock,arm,row);n+=1
            if limit is not None and n>=limit:return 'limit',n
    return 'complete',n
def main():
    os.umask(0o077)
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    init=sub.add_parser('init');init.add_argument('directory',type=Path);init.add_argument('--second',type=Path,required=True)
    for op in ('run','score','pause','resume'):
        q=sub.add_parser(op);q.add_argument('directory',type=Path)
        if op=='run':q.add_argument('--limit',type=int)
    args=p.parse_args();out=args.directory.resolve()
    if args.command=='init':initialize(out,args.second);return
    require(out.is_dir() and (out/'lock.json').is_file(),'missing campaign')
    if args.command=='pause':
        (out/'PAUSE').touch(exist_ok=True);print('Pause requested after current episode.');return
    if args.command=='resume':
        (out/'PAUSE').unlink(missing_ok=True);print('Ready to continue with run.');return
    lock=verify_lock(out,full=args.command=='run')
    if args.command=='score':
        snapshot=score(out,lock);path=out/'reports'/('report_'+uuid.uuid4().hex+'.json');write(path,snapshot)
        print(json.dumps({'paired':snapshot['aligned_n'],'total':snapshot['total_n'],'complete':snapshot['complete'],'report':str(path)}));return
    with mutex(out/'campaign.lock'):
        state,count=schedule(out,lock,args.limit,run_one)
    print(json.dumps({'state':state,'new_accepted':count}))
if __name__=='__main__':main()
