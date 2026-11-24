"""Policy-side RGB/action replay. Instance identities and counterfactual views stay private."""
from pathlib import Path
import numpy as np


class TreeEpisode:
    def __init__(self, origin_folder):
        self.origin=Path(origin_folder)
        self.path='r'
        self.steps=0
        self.observed=False

    def scan(self):
        if self.observed or len(self.path)>4:
            raise RuntimeError('scan is available once per decision, up to four decisions')
        with np.load(self.origin/('node_'+self.path+'.npz'),allow_pickle=False) as record:
            images=record['ctx']
            plans=record['plans']
        self.steps+=4
        self.observed=True
        return images,plans

    def execute(self, choice):
        if not self.observed or choice not in (0,1,2):
            raise ValueError('scan first, then choose one route')
        with np.load(self.origin/('node_'+self.path+'.npz'),allow_pickle=False) as record:
            plan=record['plans'][choice]
            observations=record['walk_rgb'][choice]
            collisions=record['collided'][choice]
        self.path+=str(choice)
        self.steps+=len(plan)
        self.observed=False
        return plan,observations,collisions
