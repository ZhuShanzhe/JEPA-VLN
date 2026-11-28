"""Inference worker. Its request packet contains only on-board observations and the cue."""
import contextlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


class ProjectedArea:
    """Geometric visibility prior frozen with the previous benchmark."""
    def predict(self,context,pose,observed_map):
        from view_model import current_inputs
        _,prior=current_inputs(context,pose,observed_map)
        return {"p_visible":1./(1.+np.exp(-prior))}


def main():
    import torch
    torch.set_num_threads(2)
    with contextlib.redirect_stdout(sys.stderr):
        from reactive_route import RoutePlanner
        model = RoutePlanner.load()
        if sys.argv[2]=="projected_area":
            model.view=ProjectedArea()
    packet = Path(sys.argv[1])
    for line in sys.stdin:
        if line.strip() != "predict":
            break
        try:
            with np.load(packet) as data:
                history = [dict(x=float(p[0]),z=float(p[1]),yaw=float(p[2]),bump=bool(b),
                           rgb=rgb,depth=depth) for p,b,rgb,depth in zip(
                           data["pose"],data["bump"],data["rgb"],data["depth"])]
                with contextlib.redirect_stdout(sys.stderr):
                    decision = model.choose(data["context"], data["box"], history[0]["depth"],
                                            history, int(data["remaining"]))
            print(json.dumps(decision, default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x),
                             allow_nan=False),flush=True)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"error":"inference exception; see private worker log"}),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=2):
        main()
