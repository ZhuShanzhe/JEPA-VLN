"""Score a frozen room prior on physically distinct calibration rooms."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from train_prior import source, stats


def brier(n,s,p):
    return ((s*(1-p)**2+(n[:,None]-s)*p**2).sum(0)/n.sum())


def run(project,assets,dst):
    prior=json.loads((project/'jepa_env_v13/data/prior.json').read_text())
    words=list(prior['p_target_given_room'])
    prob=np.array([prior['p_target_given_room'][w] for w in words]).T
    baseline=np.array([(sum(prior['rooms_with_target'][w])+.5)/(sum(prior['room_counts'])+1.)
                       for w in words])[None,:]
    with (assets/'metadata/fpmodels-with-decomposed.csv').open() as f:
        table={r['id']:r['main_wnsynsetkey'] or r['wnsynsetkey'] for r in csv.DictReader(f)}
    bins=[];counts=[];assigned=unassigned=0
    for h in source(project)['calibration'][1]:
        n,s,ok,missing=stats(assets,h,words,table)
        assigned+=ok;unassigned+=missing
        bins.append(n);counts.append(s)
    n=np.sum(bins,axis=0);s=np.sum(counts,axis=0)
    room=brier(n,s,prob);plain=brier(n,s,baseline)
    by_house=[float(np.mean(brier(h,t,prob)-brier(h,t,baseline))) for h,t in zip(bins,counts)]
    dst.write_text(json.dumps(dict(calibration_houses=len(bins),rooms=int(n.sum()),assigned=assigned,
                unassigned=unassigned,category_count=len(words),mean_brier_room_conditioned=float(room.mean()),
                mean_brier_global_prevalence=float(plain.mean()), improved_categories=int((room<plain).sum()),
                improved_houses=sum(x<0 for x in by_house),
                brier_by_category={w:[float(room[i]),float(plain[i])] for i,w in enumerate(words)}),indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser()
    for k in ('project','assets','out'):
        a.add_argument(k,type=Path)
    p=a.parse_args();run(p.project,p.assets,p.out)
