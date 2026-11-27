"""Prepare or run a single paired clock episode in a separate campaign directory."""
import argparse
import ast
import fcntl
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PILOTS = ROOT / 'mapping_runtime/pilots'
PARENT = PILOTS / 'identity_stage8_weak_relook'
STUDY = PILOTS / 'identity_stage9_spatial'
STAGE8 = ROOT / 'identity_stage8_relook'
ENTRY = 'spatial_agent:SpatialAgent'
WEIGHT = HERE / 'weights/view_jepa.npz'
PARTS = ('prepare_evaluation.py', 'run_environment.sh', 'spatial_agent.py', 'spatial_identity.py', 'view_jepa.py')
sys.path.append(str(ROOT / 'identity_stage5_confirmation'))
from development_run import SOURCES, digest, load, replace_once, save


def factory_check():
    from verify_rank import deny_ip_sockets
    deny_ip_sockets()
    from run_eval import make_agent
    from view_jepa import load_predictor
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        os.environ.update(NAV_DEBUG_ROOT=tmp, NAV_RUN_ID='factory', MAPPING_DEBUG_DIR=tmp)
        args = SimpleNamespace(seed=0, legacy_agent_api=False, log_actions=False,
                               log_action_limit=0, save_api_images_dir=None)
        agent = make_agent(ENTRY, None, args)
        cls = type(agent)
        assert cls.__module__ + ':' + cls.__name__ == ENTRY and agent._benchmark_observation_api
        assert Path(inspect.getfile(cls)).samefile(HERE/'spatial_agent.py')
        scores = load_predictor().score_views(np.zeros((96,96,3),np.uint8),[20,20,60,68],[0.,.5],[1.,.8])
        assert scores.shape==(2,) and np.isfinite(scores).all()


def adapted(name):
    source=(PARENT/name).read_text().replace('stage8_weak_relook','stage9_spatial')
    source=source.replace('18601','18605').replace('28983','28987')
    if name in ('run_five.py','pilot_failfast.py'):
        source=replace_once(source,'weak_route_agent:WeakRouteAgent',ENTRY)
    if name=='run_five.py':
        source=replace_once(source,repr(str(STAGE8/'weak_route_agent.py')),repr(str(HERE/'spatial_agent.py')))
        prefix='adapter_root='+repr(str(STAGE8))+',module_roots=['
        source=replace_once(source,prefix,'adapter_root='+repr(str(HERE))+',module_roots=['+repr(str(STAGE8))+', ')
        assert 'for cycle in range(1)' in source and '== [1]' in source
    if name=='finish_five.py':
        source=source.replace('weak_route','spatial_identity')
    if name.endswith('.py'):
        ast.parse(source)
    return source


def check_sources():
    """Read-only dry check; no frozen lock, mapper, relay or evaluator is created."""
    parent=load(PARENT/'launch_receipt.json')
    assert (PARENT/'closed.json').is_file()
    assert digest(PARENT/'panel.json')==parent['panel_sha256']
    row,=load(PARENT/'panel.json')['rows']
    assert row['ordinal']==1 and row['quota']==2
    assert load(Path(row['ref']))['result_sha256']==row['result_sha256']
    for path,sha in parent['sha256'].items():
        assert digest(path)==sha,path
    sources={name:adapted(name) for name in SOURCES}
    return parent,row,sources


def freeze():
    parent,row,sources=check_sources()
    subprocess.run(['bash',str(HERE/'run_environment.sh'),str(Path(__file__).resolve()),'factory-check'],check=True)
    files={p:v for p,v in parent['sha256'].items() if not Path(p).is_relative_to(PARENT)}
    own=[*(HERE/n for n in PARTS),WEIGHT,STAGE8/'delayed_relook_agent.py',*sorted(HERE.glob('test_*.py'))]
    files.update({str(p):digest(p) for p in own})
    save(STUDY/'panel.json',(PARENT/'panel.json').read_bytes())
    for name,source in sources.items():
        save(STUDY/name,source)
    (STUDY/'transport/pilot_entry.sh').chmod(0o700)
    files.update({str(p):digest(p) for p in [*(STUDY/n for n in SOURCES),STUDY/'panel.json']})
    checkpoints=sorted(set(parent['checkpoints'])|{str(WEIGHT)})
    preflight=STUDY/'preflight_receipt.json'
    save(preflight,json.dumps(dict(parent,sha256=files,checkpoints=checkpoints,
            config=sorted(set(files)-set(checkpoints))),sort_keys=True,indent=2)+'\n')
    sys.path.insert(0,str(STUDY))
    import run_five
    # Stop before the original runner first opens a mapper connection.
    setup,boundary,_=inspect.getsource(run_five.setup).partition('    service, env = p.service(PORT)')
    assert boundary
    setup=replace_once(setup,'gate=p.data(D/"launch_receipt.json")','gate=p.data(D/"preflight_receipt.json")')
    scope=vars(run_five).copy()
    exec(compile(setup+'    return lock\n','<offline preparation>','exec'),scope)
    lock=scope['setup']()
    assert lock['second']['agent']==ENTRY and lock['model']=='gpt-4o-2024-11-20'
    assert lock['protocol']['budget']==300
    assert all(lock['files'][str(p)]==files[str(p)] for p in own)
    assert run_five.p.accepted(run_five.C,run_five.p.data(run_five.C/'lock.json'),'native',row)
    launch=STUDY/'launch_receipt.json'
    if launch.exists():
        assert launch.samefile(preflight)
    else:
        os.link(preflight,launch)
    print('Verified one paired clock task, model identity, pretrained weights, new view head and 300-action protocol.')


def start_once():
    assert (STUDY/'launch_receipt.json').is_file()
    with (STUDY/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        submission=STUDY/'driver.log'
        if submission.exists() or (STUDY/'closed.json').exists():
            print('Submission already exists; inspect status and driver.log.')
            return
        subprocess.run([sys.executable,'-Bu',str(STUDY/'campaign.py'),'check'],check=True)
        with submission.open('xb') as log:
            child=subprocess.Popen([sys.executable,'-Bu',str(STUDY/'campaign.py'),'run'],cwd=STUDY,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('Started isolated one-episode campaign',child.pid,flush=True)


if __name__=='__main__':
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check-sources','factory-check','freeze','status','start'])
    action=parser.parse_args().action
    if action=='check-sources':
        check_sources()
        print('Parent hashes, original clock panel, adapted one-attempt driver and isolated ports verified.')
    elif action=='factory-check':
        factory_check()
    elif action=='freeze':
        freeze()
    elif action=='start':
        start_once()
    else:
        subprocess.run([sys.executable,'-Bu',str(STUDY/'campaign.py'),action],check=True)
