"""Verify all frozen houses and produce aggregate, public receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def verify(project,asset,hab):
    d=project/'jepa_env_v13/simulator'
    houses=json.loads((d/'cohort.json').read_text())
    def one(pair):
        split,house=pair
        while not (d/'trees'/split/house/'receipt.json').is_file():time.sleep(5)
        log=d/'logs'/('replay_'+house+'.log')
        with log.open('ab') as f:
            subprocess.run([str(hab),'-Bu',str(d/'replay.py'),str(project),str(asset),split,house],
                           stdin=subprocess.DEVNULL,stdout=f,stderr=f,check=True,
                           env=dict(os.environ,MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet',
                                    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(one,[(s,h) for s,group in houses.items() for h in group]))
    subprocess.run([sys.executable,'-B',str(d/'audit.py'),str(d),str(d/'coverage.json')],check=True)
    stats={}
    for split,group in houses.items():
        items=[json.loads((d/'replay'/split/(h+'.json')).read_text()) for h in group]
        stats[split]={k:(max(x[k] for x in items) if k=='max_rgb_mae' else sum(x[k] for x in items)) for k in items[0]}
    (d/'replay_summary.json').write_text(json.dumps(stats,indent=2))
    print(stats,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('project','asset','hab'):p.add_argument(n,type=Path)
    a=p.parse_args();verify(a.project,a.asset,a.hab)
