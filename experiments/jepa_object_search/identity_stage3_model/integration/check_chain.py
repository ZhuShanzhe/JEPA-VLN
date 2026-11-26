"""Exercise existing report staging, rival checks and memory writes with controlled evidence."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from identity_agent import IdentityMemoryAgent, mapped_frame
from test_chain import ReportChainTests, REPORT, TURN


def run(out):
    os.environ["NAV_STAGE3_IDENTITY"]="jepa"
    fixtures=ReportChainTests()
    fixtures.setUp()
    def example(verdict):
        agent=fixtures.make_agent(IdentityMemoryAgent)
        owner=fixtures.register(agent,3,(0.,.5,.5))
        agent.target_instance_id=owner.iid
        agent._category=lambda _: "clock"
        agent._boxes=Mock(return_value=[dict(bbox=[80,80,240,240])])
        agent._recover_report=lambda *args: (TURN,"defer")
        agent._identity_observations=lambda *args: (None,None)
        agent._stage3_decide=lambda *args:dict(state=verdict,index=0 if verdict=="same" else None)
        return agent,owner
    cases={}
    for state in ("different","uncertain"):
        agent,owner=example(state)
        action=agent._report_found(owner.iid)
        cases[state+"_retains_pending_instance"]=action!=REPORT and not owner.reported and agent._pending_report is None
    agent,owner=example("same")
    agent._identity_observations=lambda *args:None
    cases["missing_geometry_deferred"]=agent._report_found(owner.iid)!=REPORT and not owner.reported
    agent,owner=example("same")
    cases["same_stages_action_only"]=agent._report_found(owner.iid)==REPORT and not owner.reported
    agent._settle_report(fixtures.observation(21,TURN))
    cases["wrong_receipt_leaves_owner_open"]=not owner.reported and agent._pending_report is None
    agent.now=21
    agent._last_observation=fixtures.observation(21,TURN)
    cases["fresh_report_retry"]=agent._report_found(owner.iid)==REPORT
    agent._settle_report(fixtures.observation(22,REPORT))
    agent._settle_report(fixtures.observation(22,REPORT))
    cases["exactly_one_acknowledged_write"]=owner.reported and agent._reported_count==1
    cases["reported_instance_stays_closed"]=agent._report_found(owner.iid)!=REPORT
    sibling=fixtures.register(agent,4,(.18,.5,.5))
    cases["separate_nearby_instance_stays_open"]=sibling in agent.instance_store.available() and not sibling.reported
    agent,owner=example("same")
    fixtures.register(agent,4,(.18,.5,.5))
    cases["two_references_supporting_one_live_box_defer"]=agent._report_found(owner.iid)!=REPORT and not owner.reported
    h=w=320
    yy,xx=np.mgrid[:h:8,:w:8]
    z=np.ones_like(yy)*1.6
    k=np.array([[170,0,160],[0,170,160],[0,0,1]])
    pts=np.stack(((xx+.5-160)/170*z,(yy+.5-160)/170*z,z),axis=-1)
    pose=np.eye(4)
    pos={"frame_id":3,"rows":yy.flatten(),"pose":pose,"points":pts.reshape(-1,3),
         "colors":fixtures.rgb[::8,::8].reshape(-1,3)}
    m=mapped_frame(pos,fixtures.rgb,fixtures.rgb,[[80,80,240,240]],1.)
    cases["estimated_map_depth_grid"]=bool(m and np.allclose(m[3],k,atol=1e-4) and np.allclose(m[2][::8,::8],1.6)
                                         and np.isfinite(m[2]).sum()==1600)
    wrong=np.full_like(fixtures.rgb,230)
    cases["unrelated_rgb_rejected"]=mapped_frame(pos,fixtures.rgb,wrong,[[80,80,240,240]],1.) is None
    agent,owner=example("same")
    del agent._identity_observations
    agent._source=lambda *unused:(fixtures.rgb,[80,80,240,240])
    agent.supply_relook_geometry(3,20,fixtures.rgb,wrong,m[2:],m[2:])
    cases["stale_supplied_pose_does_not_certify_image"]=agent._report_found(owner.iid)!=REPORT
    assert all(cases.values()),cases
    out.write_text(json.dumps(dict(paid_calls=0,synthetic_decision_contracts=cases),indent=2,sort_keys=True)+"\n")
    print("passed",len(cases))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("out",type=Path)
    run(ap.parse_args().out)
