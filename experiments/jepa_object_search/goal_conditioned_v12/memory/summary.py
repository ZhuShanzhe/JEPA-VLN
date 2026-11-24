"""Summarize physically confirmed, instance-deduplicated cumulative discoveries."""
from pathlib import Path
import argparse
import json
import numpy as np


def compare(a,b,index):
    assert sorted(a['by_house'])==sorted(b['by_house'])
    d=np.array([a['by_house'][h][index]-b['by_house'][h][index] for h in sorted(a['by_house'])])
    boot=np.random.default_rng(1319).choice(d,(20000,len(d))).mean(1)
    return dict(delta=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())


def run(root):
    out={}
    for panel,filename,method,direct,replay in [
            ('first_panel','fresh_memory_summary','v12_jepa_crop','v12_direct',root),
            ('second_panel','second_fresh_memory_summary','rv_jepa','rv_direct',root/'expansion')]:
        table=json.loads((root/'memory'/(filename+'.json')).read_text())
        m=table[method]; base=table['v11_jepa']['shared_online_memory']
        rows=[json.loads(p.read_text()) for p in (replay/'validation/returns').glob('*/receipt.json')]
        receipts={k:sum(r[k] for r in rows) for k in ('origins','branches','successful_returns','alternatives_from_return')}
        assert all(r['exact_first_gain'] for r in rows)
        summaries={arm:{key:{k:v for k,v in result.items() if k!='by_house'} for key,result in stats.items()}
                   for arm,stats in table.items()}
        comparisons={}
        for label,(a,b) in {
                'first_head_with_shared_memory_vs_v11':(m['shared_online_memory'],base),
                'first_head_with_learned_memory_vs_v11':(m['trained_update'],base),
                'same_first_head_effect_of_learned_memory':(m['trained_update'],m['shared_online_memory']),
                'same_memory_vs_direct':(m['trained_update'],table[direct]['trained_update'])}.items():
            comparisons[label]={name:compare(a,b,idx) for idx,name in [(1,'any_hit'),(2,'new_discovery'),(3,'instances')]}
        out[panel]=dict(physical=receipts,arms=summaries,comparisons=comparisons)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.write_text(json.dumps(run(a.root),indent=2))
