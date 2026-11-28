"""Current observations -> predicted future scene embedding, traversability and target visibility."""
import numpy as np
from layout_paths import path,axes,layout_features
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
sys.path[:0]=[str(PROJECT/'new_view_20261129'),str(PROJECT/'layout_map_bridge_20261129')]
from view_model import sketch
from layout_memory import LayoutMap


def encode_current(context,anchor,box):
    return np.r_[sketch(context[:1024]),sketch(context[1024:]),anchor,box].astype(np.float32)


def describe(plan,patch,xy,yaw,goal_local,box):
    global_goal=goal_local[[0,2]]@axes(yaw)+xy
    lay=layout_features(patch,plan,xy,yaw,global_goal)
    end=path(plan)[-1]
    delta=goal_local[[0,2]]-end
    heading=(plan.count(2)-plan.count(3))*np.pi/6
    side,depth=delta@axes(heading).T
    z=max(.15,depth)
    est=box[2]*box[3]*(box[4]/z)**2
    geometry=np.r_[end,side/z,goal_local[1]/z,depth,np.sin(heading),np.cos(heading),
                   np.log(max(1.e-5,est)/.01)]
    return np.r_[np.clip(geometry,-15,15),lay].astype(np.float32)


def learned_net(dim):
    import torch.nn as nn
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden=nn.Sequential(nn.Linear(dim,96),nn.LayerNorm(96),nn.GELU(),nn.Linear(96,48),nn.GELU())
            self.future=nn.Linear(48,24)
            self.out=nn.Sequential(nn.Linear(24+16,32),nn.GELU(),nn.Linear(32,2))
        def forward(self,x):
            pred=self.future(self.hidden(x))
            return pred,self.out(__import__('torch').cat([pred,x[:,-16:]],dim=1))
    return Head()
