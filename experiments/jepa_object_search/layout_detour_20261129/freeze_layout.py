"""Record weights, navigation code and the original compact geometry panel before testing."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
entries=['collect_layout.py','encode_future.py','layout_paths.py','layout_model.py','fit_layout.py',
         'layout_policy.py','layout_worker.py','run_layout.py','private_layout.pt',
         '../layout_map_bridge_20261129/layout_memory.py','../reactive_view_20261129/rollout.py',
         '../new_view_20261129/view_design.json']
d={'sha256':{name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in entries},
   'panel':'All prior 94 groups, 3 evaluation houses, each with seven original plans blocked and initially small target. Original 22/26/30 primitive action budgets; two scans charged. No oracle policy input.',
   'arms':'Compare previous frozen routes vs learned branch; each evaluated with old and retrained verifier.'}
out=here/'freeze.json'
if out.exists():assert json.loads(out.read_text())==d
else:out.write_text(json.dumps(d,indent=2)+'\n')
print(d['sha256']['private_layout.pt'])
