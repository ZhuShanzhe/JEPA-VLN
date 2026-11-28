"""Regression fixtures that exercise map replan and native control handoff."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from agents.navigator import OccupancyGrid, PathFollower
from point_executor import FocusViewMixin


def fixture():
    free = np.ones((41,65), dtype=bool)
    return OccupancyGrid(.1, np.array([-1., -2.]), free, ~free)


class Host:
    def _estimated_current_pose(self):
        return self._frontier_pose

    def _plan_exploration(self, observation, select):
        assert select is False
        self._frontier_revision += 1

    def _block_failed_nav_direction(self, observation, follower):
        self._nav_blocked_points.append(((follower.x+.25,follower.y), observation.step_count+20))

    def _explore_action(self, observation):
        if self._last_motion_failed:
            self._explore_follower=None
            return "native_escape"
        return self._explore_follow(observation)

    def _explore_follow(self, observation):
        action, arrived = self._explore_follower.next_action()
        if action is None or arrived:
            self._explore_follower = None
        else:
            self._explore_follower.dead_reckon(action)
        return action


class Rig(FocusViewMixin, Host):
    def __init__(self):
        self.mode = "explore"
        self._explore_follower=PathFollower(1.)
        self._frontier_grid=fixture()
        self._frontier_pose=(0.,0.,0.)
        self._frontier_revision=1
        self._nav_blocked_points=[]
        self._last_motion_failed=False
        self.notes=[]
        self._last_feed_info={"frame_id":5}
        self.client=SimpleNamespace(get_all_poses=lambda:(np.asarray([np.eye(4)]),[5]))
        self.emit=lambda event,**kw:self.notes.append((event,kw))


def run():
    obs=SimpleNamespace(max_steps=300,step_count=1)
    outcomes={}
    host=Rig()
    assert host.start_focus_view(obs,[1,1,1,1,1,1,1,1,2],"wall")
    host._last_motion_failed=True
    command=host._explore_action(obs)
    assert command in (2,3) and host._focus_view[0].repairs==1, command
    assert not host._frontier_grid.obstacle.any()
    outcomes["collision_replan_on_copy_of_observed_map"]="passed"
    host=Rig()
    host._last_feed_info={"frame_id":15}
    host.start_focus_view(obs,[1,1,1,2],"wall")
    host._last_motion_failed=True
    assert host._explore_action(obs)=="native_escape" and host._focus_view is None
    outcomes["stale_anchor_returns_to_native_recovery"]="passed"
    host=Rig()
    host.start_focus_view(obs,[1,1,1,2],"wall")
    for _ in range(7):
        if host._explore_follow(obs) is None:
            break
    assert host.notes[-1][-1]["reason"]=="estimated_view_reached"
    outcomes["odometry_alone_never_certified_as_map_arrival"]="passed"
    host=Rig()
    host.start_focus_view(obs,[1,1,1,2],"wall")
    host._explore_follower=PathFollower(1.)
    host._explore_follow(obs)
    assert host._focus_view is None and host.notes[-1][-1]["reason"]=="native_recovery_or_replacement"
    outcomes["higher_priority_follower_replacement_preserved"]="passed"
    return outcomes


if __name__=="__main__":
    result=run()
    print(result)
    if len(sys.argv)>1:
        Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+"\n")
