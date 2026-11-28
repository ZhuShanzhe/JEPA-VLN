"""Freeze the route model, its clearance bridge, validation settings and heading-aware runtime."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
files=['grid_policy.py','layout_worker.py','run_layout.py','calibrate_grid.py','calibration.json',
       '../layout_detour_20261129/private_layout.pt','../layout_detour_20261129/layout_paths.py',
       '../layout_detour_20261129/layout_policy.py','../layout_detour_20261129/layout_model.py',
       '../layout_map_bridge_20261129/layout_memory.py','../reactive_view_20261129/rollout.py',
       '../new_view_20261129/view_design.json']
d={'sha256':{k:hashlib.sha256((here/k).read_bytes()).hexdigest() for k in files},
   'selection':'Six calibration houses; physical feasible-and-visible endpoints used for ranking. 94 old groups are a regression set. Separate official validation houses to measure transfer.'}
out=here/'freeze.json'
if out.exists():assert json.loads(out.read_text())==d
else:out.write_text(json.dumps(d,indent=2)+'\n')
print(d['sha256']['grid_policy.py'])
