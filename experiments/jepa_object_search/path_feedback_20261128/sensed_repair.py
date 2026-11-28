"""A bounded observation loop for same-waypoint local repairs. Source target/visual memory stays anchored."""
import numpy as np
from path_geometry import path_features,from_depth,NAMES
from replan import ProposalSet,decide

# Start straight ahead, then 30 degrees left, then 30 degrees right.
LOOK_ACTIONS=((),(2,),(3,3))
UNDO=((),(3,),(2,))


def consider(plans,depth,undo,visual,checker):
    options=[undo+tuple(plan) for plan in plans]
    width=max(map(len,options))
    a=np.zeros((len(options),width),np.int8)
    x=np.zeros((len(options),width,len(NAMES)),np.float32)
    observed=from_depth(depth)
    for j,p in enumerate(options):
        a[j,:len(p)]=p;x[j,:len(p)]=path_features(p,observed)
    check=checker.predict(a,x)
    return ProposalSet(a,x,check["p_reach"],check["threshold"],check["suspect_step"],
        check["suspect_side"],check["decision"],visual["sketch"],visual["old_joint"],visual["p_visible_if_reach"])


def choose_with_observation(plans,visual,observe,checker,scorer,initial,budget,arm,max_views=3):
    """observe(turns) executes just those turns and returns current depth, or a physical prerecorded equivalent."""
    if not 1<=max_views<=len(LOOK_ACTIONS):raise ValueError("unsupported number of observation views")
    cost=0
    checks=0
    for view in range(max_views):
        turns=LOOK_ACTIONS[view]
        if cost+len(turns)+len(UNDO[view])+len(plans[initial])>budget:break
        depth=observe(turns)
        cost+=len(turns)
        s=consider(plans,depth,UNDO[view],visual,checker)
        result=decide(s,initial,budget-cost,arm,scorer,max_checks=3)
        checks+=len(result["history"])
        if result["index"] is not None:
            return dict(view=view,index=result["index"],actions=result["actions"],sensing_actions=cost,
                        checks=checks,status="execute")
    return dict(view=None,index=None,actions=[],sensing_actions=cost,checks=checks,status="defer")
