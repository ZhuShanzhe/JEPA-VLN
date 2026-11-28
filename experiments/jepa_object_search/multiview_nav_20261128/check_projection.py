"""Isolate camera binding; use different known geometry for each scan view."""
import math
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
for d in ("committed_inspect_20261128","gpt_inspect_only_20261128","local_target_20261128","approach_search_20261128"):
    sys.path.insert(0,str(HERE.parent/d))
from multi_policy import located_cue


class Grid:
    res=.1


def main():
    pose=(.3,-.4,.42)
    headings=[pose[2]+i*math.pi/6 for i in (-1,0,1)]
    views=np.zeros((3,192,192,3),np.uint8)
    geometry=[((*pose[:2],a),np.full((192,192),d,np.float32)) for a,d in zip(headings,(1.,2.,3.))]
    for idx in range(3):
        choice=dict(view_index=idx, bbox_norm=[.4,.4,.6,.6], candidate_id=str(idx),category="clock")
        arr,src,yaw=located_cue(choice,views,Grid(),pose,1.,geometry,90.)
        assert len(arr)==1 and src=="selected_view_aligned_rgbd" and yaw==headings[idx]
        expect=np.array(pose[:2])+(idx+1)*np.array([math.cos(yaw),math.sin(yaw)])
        assert np.linalg.norm(np.array(arr[0]["anchor_xy"])-expect)<.025
        corrupted=list(geometry)
        corrupted[idx]=(geometry[idx][0],np.full((192,192),np.nan,np.float32))
        assert not located_cue(choice,views,Grid(),pose,1.,corrupted,90.)[0]
    print("three distinct metric depths and three camera headings; invalid selected depth rejected")


if __name__=="__main__":
    main()
