"""Audit clustered success, matching instance IDs and triply distinct candidate actions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from train_compare import write


def main(root):
    exp=root/'jepa_hidden_v11';d=exp/'models'
    groups=defaultdict(lambda:defaultdict(list));meta={}
    for source in ('hssd','blender'):
        for p in (exp/source/'test').glob('*/*.npz'):
            with np.load(p) as z:
                key=source,str(z['house']),str(z['origin'])
                distinctive=len({tuple(x) for x in z['actions']})==len(z['actions'])
                md=dict(distinct=distinctive,words=z['words'].tolist())
                if 'hit_instance_ids' in z:md.update(ids=z['hit_instance_ids'])
                meta[key]=md
    for line in (d/'test_predictions.jsonl').read_text().splitlines():
        r=json.loads(line);md=meta[r['source'],r['house'],r['origin']]
        y=np.array(r['gain']);f=r['first'];s=r['second'];q=md['words'].index(r['query'])
        tags=[]
        if max(y)>0:
            tags+=['positive']
            if md['distinct']:tags+=['positive_distinct_actions']
            tags+=[f'profitable_{sum(y>0)}']
        if r['source']=='blender':
            ids=[set(i for i in row[q].tolist() if i>=0) for row in md['ids']]
            assert [len(i) for i in ids]==y.tolist()
            target=set.union(*ids)
            if target:
                assert s is not None
                found=ids[f]|ids[s];total=len(target)
                tags+=['multiple_instances' if total>1 else 'single_instance']
            else:
                tags+=['absent'];found=set();total=0
            if total>1:more=[float(len(found)),len(found)/total,float(len(found)==total)]
            else:more=[None,None,None]
        else:more=[None,None,None]
        for tag in tags:
            k=(r['source'],r['mode'],tag,r['arm'])
            groups[k][r['house']].append([float(y[f]>0),float(y[f]),float(max(y[f],y[s])>0) if s is not None else None,*more])
    output={}
    for k,h in groups.items():
        items=list(h.values());avg=[]
        for i in range(6):
            scores=[[row[i] for row in rows if row[i] is not None] for rows in items]
            avg.append(float(np.mean([np.mean(xs) for xs in scores if xs])) if any(scores) else None)
        output['|'.join(k)]=dict(houses=len(items),n_query_origins=sum(map(len,items))//3,
            first_hit=avg[0],first_distinct_gain=avg[1],two_probe_hit=avg[2],multi_unique_found=avg[3],
            multi_all_instance_recall=avg[4],multi_all_completed=avg[5])
    write(d/'sensitivity.json',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
