"""Additional training trees and labels on separate HSSD training houses only."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import subprocess
from tree import seed


def build(project,asset,hab):
    d=project/'jepa_env_v13/simulator'
    old=json.loads((project/'jepa_hidden_v11/hssd/cohort.json').read_text())
    extra=json.loads((project/'jepa_route_v12/expansion/hssd/cohort.json').read_text())
    test=json.loads((d/'cohort.json').read_text())
    training=sorted(set(old['train']+extra['train']),key=lambda h:seed('v13-long-train',h))
    assert len(training)==40 and set(training).isdisjoint(test['calibration']+test['fresh'])
    split={'train':training[:12],'origins_per_house':4}
    dest=d/'training_cohort.json'
    if dest.exists() and json.loads(dest.read_text())!=split:raise RuntimeError('changed cohort')
    dest.write_text(json.dumps(split,indent=2))
    def one(house):
        with (d/'logs'/('train_'+house+'.log')).open('ab') as f:
            subprocess.run([str(hab),'-Bu',str(d/'tree.py'),str(project),str(asset),'train',house,'--origins','4'],
                          stdout=f,stderr=f,check=True,env=dict(os.environ,OMP_NUM_THREADS='1',
                          OPENBLAS_NUM_THREADS='1',MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet'))
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one,split['train']))
    files=list((d/'trees/train').glob('*/*/training_targets.json'))
    rows=[g for f in files for g in json.loads(f.read_text())['targets']]
    stat=dict(houses=len(training[:12]),origins=len(files),queries=len(rows),
              encounterable=sum(x['encounterable'] for x in rows),absent=sum(x['house_count']==0 for x in rows))
    (d/'train_summary.json').write_text(json.dumps(stat,indent=2))
    print(stat)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('project','asset','hab'):p.add_argument(n,type=Path)
    a=p.parse_args();build(a.project,a.asset,a.hab)
