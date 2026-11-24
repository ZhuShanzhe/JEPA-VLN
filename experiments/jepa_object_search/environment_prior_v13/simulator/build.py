"""Freeze the retained cohort first, then physically traverse every planned origin."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from tree import cohorts


def build(project,asset,hab,workers):
    d=project/'jepa_env_v13/simulator'
    frozen=cohorts(project);path=d/'cohort.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('cohort mismatch')
    path.write_text(json.dumps(frozen,indent=2));(d/'logs').mkdir(exist_ok=True)
    def task(item):
        split,house=item
        with (d/'logs'/(house+'.log')).open('ab') as out:
            p=subprocess.run([str(hab),'-Bu',str(d/'tree.py'),str(project),str(asset),split,house],
                  stdout=out,stderr=out,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                                                MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet'))
        print(split,house,p.returncode,flush=True)
        return p.returncode
    with ThreadPoolExecutor(workers) as ex:
        result=list(ex.map(task,[(s,h) for s,hs in frozen.items() for h in hs]))
    if any(result):raise RuntimeError('house process failure, see retained logs')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('project','asset','hab'):p.add_argument(k,type=Path)
    p.add_argument('--workers',type=int,default=6);a=p.parse_args();build(a.project,a.asset,a.hab,a.workers)
