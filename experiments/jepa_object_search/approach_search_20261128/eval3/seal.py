"""Make independent, read-only copies of navigation and placement modules before evaluation."""
import hashlib
import json
from pathlib import Path
from shutil import copyfile
here=Path(__file__).resolve().parent
project=here.parents[1]
lock=here/'locked';lock.mkdir(exist_ok=True)
paths={k:here.parent/k for k in ('map_views.py','global_frontiers.py','search_policy.py','enhanced_search.py')}
paths.update({k:project/'prior_approach_20261128'/k for k in ('placement_prior.py','context_calibration.json','learned_prior.json')})
paths.update({'point_executor.py':project/'route_execution_v1/point_executor.py',
              'agents/__init__.py':project/'private_sources/MOST-Nav/agents/__init__.py',
              'agents/navigator.py':project/'private_sources/MOST-Nav/agents/navigator.py'})
for k,v in paths.items():
    dst=lock/k;dst.parent.mkdir(exist_ok=True)
    if dst.exists() and dst.read_bytes()!=v.read_bytes():raise RuntimeError(f"Sealed source already exists: {k}")
    if not dst.exists():
        copyfile(v,dst);dst.chmod(0o444)
assert hashlib.sha256((lock/'enhanced_search.py').read_bytes()).hexdigest()=="9520ff58bb3c3a93c5b246defc163ca6f32e26ed8e51976f72fcca0d2b9f3558"
files=[lock/k for k in paths]+[here/f for f in ('accept.py','panel.json','physical_panel.py','inference_worker.py','seal.py')]
files.extend(project/f for f in ('jepa_variable_v16/model/variable_evidence.py','jepa_variable_v16/model/calibration.json',
                                 'jepa_evidence_v15/train/frozen.pt','jepa_env_v13/models/selection.json'))
record={str(f.relative_to(project)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
path=here/'locked_sources.json'
if path.exists() and json.loads(path.read_text())!=record:raise RuntimeError('source changes after freeze')
path.write_text(json.dumps(record,indent=2))
