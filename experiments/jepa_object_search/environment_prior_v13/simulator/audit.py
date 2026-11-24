"""Measure attainable opportunities, including physically present but unreachable goals."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path


def profile(folder,expected):
    counts=Counter();categories=Counter();by_stratum=defaultdict(Counter)
    for h in expected:
        d=folder/h
        receipt=json.loads((d/'receipt.json').read_text())
        episodes=list(d.glob('*/episode.json'))
        assert receipt['origins']==len(episodes)
        counts['houses']+=1
        counts['origins']+=len(episodes)
        for f in episodes:
            meta=json.loads(f.read_text());truth=json.loads((f.parent/'truth.json').read_text())
            node=truth['nodes'];inv=truth['instance_words']
            assert len(node)==len(list(f.parent.glob('node_*.npz')))==40
            counts['nodes']+=len(node);counts['physical_legs']+=3*len(node)
            counts['primitive_actions_across_shared_tree']+=(4+3*24)*len(node)
            counts['fallback_plans']+=sum(v['fallback_count'] for v in node.values())
            for target in meta['targets']:
                q=target['word'];n=target['house_count'];label=target['stratum']
                assert sum(v==q for v in inv.values())==n
                assert not any(inv.get(str(i))==q for x in node['r']['scan'].values() for i in x)
                categories[q]+=1
                best=[0,0,0,0];routes={'r':set()}
                for length in range(1,5):
                    advance={}
                    for path,past in routes.items():
                        one=node[path]
                        prior=past|{i for ids in one['scan'].values() for i in ids if inv.get(str(i))==q}
                        for j,leg in enumerate(one['walk']):
                            found=prior|{i for ids in leg.values() for i in ids if inv.get(str(i))==q}
                            best[length-1]=max(best[length-1],len(found));advance[path+str(j)]=found
                    routes=advance
                assert all(x<=y for x,y in zip(best,best[1:])) and best[-1]<=n
                tab=by_stratum[label];tab['queries']+=1
                for i,budget in enumerate((28,56,84,112)):
                    tab[f'any_attainable_{budget}']+=int(best[i]>0)
                    tab[f'two_attainable_{budget}']+=int(best[i]>=2)
                    tab[f'max_distinct_sum_{budget}']+=best[i]
    return dict(counts=counts,per_goal=categories,by_inventory=by_stratum)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();cohort=json.loads((a.root/'cohort.json').read_text())
    out={s:profile(a.root/'trees'/s,hs) for s,hs in cohort.items()}
    a.output.write_text(json.dumps(out,indent=2))
