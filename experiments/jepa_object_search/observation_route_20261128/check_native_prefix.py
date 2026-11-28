"""Exercise the actual optional native carrier without sensors, model, paid calls or simulator."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

import numpy as np


def main():
    from benchmark_api import Action
    from evaluation.main import run_eval
    from host import ObservationRouteAgent
    from multi_agent import MultiViewAgent
    from host_dispatch import ObservationActions

    with tempfile.TemporaryDirectory() as td, patch.dict(os.environ,{
        "NAV_DEBUG_ROOT":td,"NAV_RUN_ID":"native_route_check","MAPPING_DEBUG_DIR":td,
        "NAV_REQUIRE_DECISION_PREFLIGHT":"0"
    }):
        args = SimpleNamespace(seed=0,legacy_agent_api=False,log_actions=False,log_action_limit=0,
                               save_api_images_dir=None)
        agent = run_eval.make_agent(os.environ["MOST_FROZEN_AGENT"],None,args)
        assert isinstance(agent,ObservationRouteAgent)
        agent.output_dir = td
        obs = SimpleNamespace(episode_id="native_route_check",step_count=0,goal_text="find clocks",
                              max_steps=300,previous_action=None,rgb=np.zeros((192,192,3),np.uint8))
        agent.init_episode(obs)
        agent._estimated_current_pose = lambda: (0.,0.,0.)
        agent.mode = "explore"
        agent._explore_follower = SimpleNamespace(scale=1.,anchor_frame=1)
        first = [2,1,3]
        assert agent.start_focus_view(obs,ObservationActions(first),"local/obs1/1")
        carrier = agent._explore_follower
        receipt = [carrier.next_action() for _ in range(len(first)+1)]
        assert receipt == [(a,False) for a in first]+[(None,True)]
        assert agent.start_focus_view(obs,ObservationActions([1,1]),"local/contact/1")
        assert agent._explore_follower.next_action()==(1,False)
        obs.previous_action = 1
        agent._last_motion_failed = True
        # The original native obstacle recovery decides its own subsequent action.
        # Isolate that downstream call and check what state it receives.
        def recovered(native, observation):
            return native._focus_view is None and native._explore_follower is None
        with patch.object(MultiViewAgent, "_explore_action", recovered):
            assert agent._explore_action(obs) and agent._local_blocked
        agent._last_motion_failed = False
        agent._explore_follower = SimpleNamespace(scale=1.,anchor_frame=1)
        assert agent.start_focus_view(obs,ObservationActions([3,1]),"local/obs2/1")
        assert agent._explore_follower.next_action()==(3,False)
        result = agent._execute_decider_result(obs,SimpleNamespace(action="FINISH"),"controlled")
        assert result==int(Action.FINISH) and agent._focus_view is None and agent._explore_follower is None
        print(json.dumps(dict(native_factory=True,scored=first,sent=[a for a,done in receipt if not done],
                              contact_cancelled=True,finish_preempted=True,offline=True,paid_calls=0)))

if __name__=="__main__":
    main()
