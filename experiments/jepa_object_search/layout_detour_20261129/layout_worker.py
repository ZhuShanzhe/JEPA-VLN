"""Stream sensory history to a layout planner isolated from simulator IDs and target masks."""
import contextlib,json,sys,traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'reactive_view_20261129'))
from reactive_route import RoutePlanner
from layout_policy import LayoutPlanner
if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2),contextlib.redirect_stdout(sys.stderr):
        planner=LayoutPlanner(RoutePlanner.load())
    for cmd in sys.stdin:
        if cmd.strip()!='predict':break
        try:
            with np.load(sys.argv[1]) as d,threadpool_limits(limits=2),contextlib.redirect_stdout(sys.stderr):
                history=[dict(x=float(p[0]),z=float(p[1]),yaw=float(p[2]),bump=bool(b),rgb=rgb,depth=depth)
                         for p,b,rgb,depth in zip(d['pose'],d['bump'],d['rgb'],d['depth'])]
                decision=planner.choose(d['context'],d['box'],history[0]['depth'],history,int(d['remaining']))
            print(json.dumps(decision,allow_nan=False),flush=True)
        except Exception:
            traceback.print_exc(file=sys.stderr);print(json.dumps({'error':'inference exception; see worker log'}),flush=True)
