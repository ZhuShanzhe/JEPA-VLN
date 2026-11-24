"""Privileged hindsight bounds across the complete physical tree, independent of policies."""
from collections import defaultdict
from itertools import combinations
import argparse
import json
from pathlib import Path
import numpy as np


def one(meta,truth):
    words=truth['instance_words'];nodes=truth['nodes'];result={}
    for goal in meta['targets']:
        w=goal['word'];n=goal['house_count']
        options={'r':set()}
        for budget in (28,56,84,112):
            candidates={}
            for name,seen in options.items():
                s=nodes[name];scan={i for v in s['scan'].values() for i in v if words.get(str(i))==w}
                for b,obs in enumerate(s['walk']):
                    discovered={i for values in obs.values() for i in values if words.get(str(i))==w}
                    candidates[name+str(b)]=seen|scan|discovered
            gains={a:len(ids) for a,ids in candidates.items()}
            best=max(gains.values());assert best<=n
            result[w,budget]=dict(n=n,stratum=goal['stratum'],best=best,
                                  first_choices={a[1] for a,v in gains.items() if v>0})
            options=candidates
    return result


def report(folder):
    result={}
    for split in ('calibration','fresh'):
        groups=defaultdict(list);pairs=defaultdict(lambda:defaultdict(int))
        files=sorted((folder/'trees'/split).glob('*/*/episode.json'))
        for f in files:
            meta=json.loads(f.read_text());rid=meta['origin'];house=meta['house']
            rows=one(meta,json.loads((f.parent/'truth.json').read_text()))
            for (w,b),x in rows.items():
                labels=[x['stratum']]
                if x['n']>0:labels.append('all_present')
                for g in labels:groups[g,b].append((house,rid,x['best'],x['n']))
            for budget in (28,56,84,112):
                goals=[x for (_,b),x in rows.items() if b==budget and x['n']>0]
                for u,v in combinations(goals,2):
                    p=pairs[budget];p['all_present_goal_pairs']+=1
                    a,b=u['first_choices'],v['first_choices']
                    if a and b:
                        p['both_encounterable']+=1
                        if a.isdisjoint(b):p['disjoint_needed_first_choice']+=1
        out={}
        for (g,b),vs in sorted(groups.items()):
            origins=defaultdict(list);houses=defaultdict(list)
            for h,o,count,n in vs:
                origins[h,o].append([count>0,count>=2,count, count>=min(2,n) if n else False])
            for (h,_),values in origins.items():houses[h].append(np.mean(values,axis=0))
            means=np.mean([np.mean(means,axis=0) for means in houses.values()],axis=0)
            out[g+'|'+str(b)]=dict(houses=len(houses),origins=len(origins),queries=len(vs),
                can_find_one=sum(x[2]>=1 for x in vs),can_find_two=sum(x[2]>=2 for x in vs),
                no_opportunity=sum(x[2]==0 for x in vs),max_distinct_sum=sum(x[2] for x in vs),
                macro_any=means[0],macro_two=means[1],macro_max_distinct=means[2],
                macro_reach_inventory_capped_quota=means[3] if g!='absent' else None)
        result[split]=dict(groups=out,exclusive_first_directions_between_target_pairs=dict(pairs))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();a.out.write_text(json.dumps(report(a.folder),indent=2))
