"""Freeze or run the single paired clock episode with the Stage 7 agent."""
import argparse
import ast
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
PARENT = PILOTS / 'identity_stage6_active_report'
STUDY = PILOTS / 'identity_stage7_arrival_transfer'
STAGE6 = ROOT / 'identity_stage6_approach'
ENTRY = 'transfer_agent:ArrivalTransferAgent'
sys.path.append(str(ROOT / 'identity_stage5_confirmation'))
from development_run import SOURCES, digest, load, replace_once, save


def factory_check():
    from verify_rank import deny_ip_sockets
    deny_ip_sockets()
    from run_eval import make_agent
    spec = load(STUDY / 'spec.json')
    with tempfile.TemporaryDirectory() as tmp:
        os.environ.update(NAV_DEBUG_ROOT=tmp, NAV_RUN_ID='factory', MAPPING_DEBUG_DIR=tmp)
        args = SimpleNamespace(seed=0, legacy_agent_api=False, log_actions=False,
                               log_action_limit=0, save_api_images_dir=None)
        agent = make_agent(spec['agent'], None, args)
        cls = type(agent)
        assert cls.__module__ + ':' + cls.__name__ == ENTRY and agent._benchmark_observation_api
        assert Path(inspect.getfile(cls)).samefile(spec['source'])


def freeze():
    parent = load(PARENT / 'launch_receipt.json')
    panel = PARENT / 'panel.json'
    assert digest(panel) == parent['panel_sha256']
    row, = load(panel)['rows']
    assert row['ordinal'] == 1 and row['quota'] == 2
    assert load(Path(row['ref']))['result_sha256'] == row['result_sha256']
    assert all(digest(p) == v for p,v in parent['sha256'].items())
    save(STUDY / 'panel.json', panel.read_bytes())
    for name in SOURCES:
        source = (PARENT / name).read_text()
        source = source.replace('stage6_active_report', 'stage7_arrival_transfer')
        source = source.replace('18593', '18597').replace('28975', '28979')
        if name == 'run_five.py':
            source = replace_once(source, 'approach_agent:ApproachAgent', ENTRY)
            source = replace_once(source, repr(str(STAGE6 / 'approach_agent.py')), repr(str(HERE / 'transfer_agent.py')))
            source = replace_once(source, 'adapter_root='+repr(str(STAGE6))+',module_roots=[',
                                  'adapter_root='+repr(str(HERE))+',module_roots=['+repr(str(STAGE6))+', ')
        if name.endswith('.py'):
            ast.parse(source)
        save(STUDY / name, source)
    (STUDY / 'transport/pilot_entry.sh').chmod(0o700)
    files = {p:v for p,v in parent['sha256'].items() if not Path(p).is_relative_to(PARENT)}
    files.update({str(p):digest(p) for p in [
        *(STUDY / n for n in SOURCES), STUDY / 'panel.json', Path(__file__).resolve(),
        *(HERE / n for n in ('transfer_agent.py','local_match.py','run_environment.sh',
                             'test_local_match.py','test_transfer_gate.py','test_native_leg.py'))]})
    preflight = STUDY / 'preflight_receipt.json'
    manifest = dict(parent, sha256=files, config=sorted(set(files)-set(parent['checkpoints'])))
    save(preflight, json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    sys.path.insert(0,str(STUDY))
    import run_five
    # Run the same setup up to its first live mapper call, including all lock
    # and policy checks. No controller, simulator or service starts here.
    setup, boundary, _ = inspect.getsource(run_five.setup).partition('    service, env = p.service(PORT)')
    assert boundary
    setup = replace_once(setup,'gate=p.data(D/"launch_receipt.json")','gate=p.data(D/"preflight_receipt.json")')
    scope = vars(run_five).copy()
    exec(compile(setup+'    return lock\n','<offline preparation>','exec'),scope)
    lock = scope['setup']()
    assert lock['second']['agent'] == ENTRY and lock['model'] == 'gpt-4o-2024-11-20'
    assert lock['protocol']['budget'] == 300 and lock['files'][str(HERE / 'local_match.py')] == files[str(HERE / 'local_match.py')]
    assert run_five.p.accepted(run_five.C,run_five.p.data(run_five.C/'lock.json'),'native',row)
    subprocess.run(['bash',str(HERE/'run_environment.sh'),str(Path(__file__).resolve()),'factory-check'],check=True)
    launch = STUDY / 'launch_receipt.json'
    if launch.exists():
        assert launch.samefile(preflight)
    else:
        os.link(preflight,launch)
    print('Verified Stage 7 factory, source, model, 300 actions and the identical clock panel.')


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',default='freeze',choices=['freeze','factory-check','status','run'])
    action = parser.parse_args().action
    if action == 'freeze':
        freeze()
    elif action == 'factory-check':
        factory_check()
    else:
        subprocess.run([sys.executable,'-Bu',str(STUDY/'campaign.py'),action],check=True)
