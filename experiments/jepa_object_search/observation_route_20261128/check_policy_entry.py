"""End-to-end decision-contract fixtures with real geometry and controlled visual scores."""
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from check_routes import MeasuredGrid
from host_dispatch import ObservationActions

root = Path(__file__).resolve().parents[1]
for stage in ("multiview_nav_20261128","committed_inspect_20261128","gpt_inspect_only_20261128",
              "local_target_20261128","approach_search_20261128"):
    sys.path.insert(0,str(root/stage))
from local_search import LocalTargetTracker
from policy import ObservationRoutePolicy


class FixtureGate:
    def __init__(self,mode):
        self.mode = mode
    def choose(self,**kw):
        return dict(mode=self.mode,region_id="r0")


class FixtureCue:
    last_receipt = {"event":"fixture_choice"}
    calls = 0
    def select(self,views,goal):
        self.calls += 1
        return dict(view_index=2,bbox_norm=[.38,.4,.62,.6],category=goal,candidate_id="L1")


class FixtureMemory:
    def __init__(self):
        self.inspected = set()
    def update_context(self,*_):
        pass
    def choose(self,rows):
        return 0,dict(candidates=[dict(prior=0.,novelty=0.,unseen=0.,steps=len(rows[0]["actions"]),blocked=False)])


def exercise(upper, valid_depth=True, tied=False):
    grid, pose = MeasuredGrid(),(4.,4.,0.)
    views = [np.zeros((96,96,3),np.uint8) for _ in range(3)]
    distances = [2.,2.,.61 if valid_depth else float("nan")]
    geometry = [((*pose[:2],math.radians(deg)),np.full((96,96),d,np.float32))
                for deg,d in zip((-30,0,30),distances)]
    row = dict(key="regional",actions=[2,2,1,1],position=(4.,5.),heading=0.)
    cue = FixtureCue()
    obj = ObservationRoutePolicy(FixtureGate(upper),cue=cue)
    calls = []
    def scores(_,mode,plans,**kw):
        calls.append(mode)
        ranking = [0.]*len(plans) if tied else list(range(len(plans)))
        if mode=="selected_box":
            assert kw["view_index"]==2 and kw["bbox_norm"]==[.38,.4,.62,.6]
            return ranking
        return dict(score=[0.]*len(plans),evidence=ranking,current=[0.]*3,patches=[])
    with patch("policy.request",scores):
        decision,region=obj.pick(LocalTargetTracker(),FixtureMemory(),grid,pose,1.,[row],views,
                                 "clock",40,None,scan_geometry=geometry)
    assert region is row and decision["region_key"]=="regional"
    if (upper=="explore" or valid_depth) and not tied:
        assert isinstance(decision["actions"],ObservationActions) and decision["action_source"]=="local"
        options = decision["observation_routes"]["candidates"]
        assert list(decision["actions"])==next(r["actions"] for r in options if r["id"]==decision["route_selected"])
    else:
        assert decision["action_source"]=="region" and decision["actions"]==row["actions"]
    return decision,calls,cue.calls


if __name__ == "__main__":
    d,seq,boxes=exercise("explore")
    assert (d["observation_mode"],seq,boxes)==("discover",["observe","observe"],0)
    d,seq,boxes=exercise("inspect")
    assert (d["observation_mode"],seq,boxes)==("verify",["observe","selected_box"],1)
    d,seq,_=exercise("inspect",valid_depth=False)
    assert (d["gateway_reason"],seq)==("visual_cue_unavailable",["observe"])
    d,_,_=exercise("explore",tied=True)
    assert d["route_selected"]=="continue"
    print("PASS actual policy: no-cue discovery, left-camera verification, missing source depth, tie to regional fallback")
