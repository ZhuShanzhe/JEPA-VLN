"""Audit official HSSD-OV validation house exposure independently of policy scores."""
from pathlib import Path
import json
import re
import hashlib

HERE = Path(__file__).resolve().parent
P = HERE.parents[1]
BASE = P / 'approach_search_20261128'
DATA = P / 'a_hssdov_external'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    houses = {x.name.removesuffix('.json.gz') for x in (DATA / 'eval/hssdov/content').glob('*.json.gz')}
    previous = {}
    for name in ('eval', 'eval2','eval3','eval4'):
        path=BASE / name / 'panel.json'
        previous[name] = sorted({r['scene'] for r in read(path)})
    path=BASE / 'eval4/eligibility.json'
    previous['geometric_eligibility'] = sorted({r['scene'] for r in read(path)})
    previous['v13_scene_split'] = sorted({s for a in read(P/'jepa_env_v13/simulator/cohort.json').values() for s in a})
    previous['v13_training'] = sorted(set(read(P/'jepa_env_v13/simulator/training_cohort.json')['train']))
    previous['learned_prior_training'] = sorted(set(read(BASE/'eval4/locked/learned_prior.json')['train_houses']))
    for path in [P/'a_exploration/paired_branches_v2/cohort.json', P/'jepa_hidden_v11/hssd/cohort.json',
                 *sorted((P/'jepa_route_v12').rglob('*cohort.json')),
                 P/'a_hssdov_external/runtime_check/receipt.json', P/'a_exploration/hssd_smoke/receipt.json']:
        content=path.read_text()
        if path == P/'jepa_hidden_v11/hssd/cohort.json':
            historical=json.loads(content)
            for root,values in historical['footprints'].items():
                previous['v11-project:'+root]=sorted(houses.intersection(values)) if root.startswith('jepa_object_search/') else []
            content=json.dumps({k:historical[k] for k in ('train','calibration','test')})
        ids=set(re.findall(r'(?<!\d)\d{9}(?:_\d{9})?(?!\d)', content))
        previous[str(path.relative_to(P))]=sorted(ids & houses)
    used = set().union(*map(set,previous.values()))
    return houses, previous, used


if __name__=='__main__':
    houses, previous, used=inventory()
    print(json.dumps({'houses':len(houses), 'previous':{k:len(set(v) & houses) for k,v in previous.items()},'remaining':sorted(houses-used)}))
