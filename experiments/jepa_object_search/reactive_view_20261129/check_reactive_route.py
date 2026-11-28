"""Geometry and frozen actor acceptance checks; no simulator labels required."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from reactive_route import HERE, VIEWS, RoutePlanner, axes, routes, contact_edges, local_depth_map, keyframes


def main():
    torch.set_num_threads(2)
    plane=np.full((192,192),2.,np.float32)
    def frame(x,z,yaw,bump=False):
        return dict(x=x,z=z,yaw=yaw,depth=plane,bump=bump)
    origin=frame(0,0,0)
    left=frame(0,0,math.pi/6)
    assert np.allclose(axes(math.pi/6)@axes(math.pi/6).T,np.eye(2))
    assert np.array_equal(local_depth_map([left,origin]).shape,(192,192))
    # A complete native +30 degree source frame has valid coverage in the virtual left camera.
    assert (local_depth_map([left,origin])[3:-3,3:-3]>.05).mean()>.96
    route_anchor=np.array([0.,.1,2.])
    horizontal=(1,1)
    assert any(horizontal in ps for _,ps in routes(route_anchor,26,[origin]))
    collided=[origin,frame(0,0,0,True)]
    assert np.allclose(contact_edges(collided)[0][0],[0.,.25])
    assert all(horizontal not in ps for _,ps in routes(route_anchor,26,collided))
    assert len(keyframes(collided))==1
    route=RoutePlanner.load()
    file=next(iter(sorted((VIEWS/"private_data").glob("val_*.npz"))))
    folder=VIEWS/"private_render"/file.stem
    with np.load(file) as z: ctx=z["context"][0]
    with np.load(folder/"0_sensor.npz") as z:
        depth=z["depth"];scan=z["left_depth"];box=z["box"]
    h=[dict(origin,depth=depth),dict(left,depth=scan),dict(origin,depth=depth),
       dict(origin,depth=depth,bump=True)]
    op=route.choose(ctx,box,depth,h,18)
    assert len(op["actions"])<=18 and set(op["actions"])<={1,2,3}
    assert all(len(p)<=5 for _,ps in routes(route_anchor,5,h) for p in ps)
    assert route.choose(ctx,box,depth,h,0)["actions"]==[]
    result=dict(frozen_checker=True,deterministic_contact_edge=True,legal_actions=True,
                translated_geometry=True,virtual_depth_projection=True,calibration_smoke=op["reason"])
    (HERE/"route_interface_check.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)

if __name__=="__main__":
    with threadpool_limits(limits=2):main()
