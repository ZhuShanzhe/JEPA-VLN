"""Episode-local odometry and measured RGB stream at the original navigation interface."""
import json
import math
from pathlib import Path

from benchmark_api import Action
from v8_extended import JepaFrontier as RouteAgent


class JepaFrontier(RouteAgent):
    def __init__(self):
        self.mem_step = None
        self.mem_query = None
        self.xy = [0., 0.]
        self.angle = 0.
        super().__init__()

    def _feed_frame(self, obs):
        # MOST-Nav calculates motion failure from successive observed frames here.
        result = super()._feed_frame(obs)
        step = int(obs.step_count)
        query = str(obs.goal_text)
        reset = self.mem_step is None or self.mem_query != query or step < self.mem_step
        if not reset and step == self.mem_step:
            return result
        if reset:
            self.xy, self.angle = [0., 0.], 0.
        elif step != self.mem_step + 1:
            raise ValueError("unobserved gap in action history")
        action = None if reset else obs.previous_action
        blocked = bool(action == int(Action.MOVE_FORWARD) and self._last_motion_failed)
        if action == int(Action.MOVE_FORWARD) and not blocked:
            self.xy[0] += .25 * math.cos(self.angle)
            self.xy[1] += .25 * math.sin(self.angle)
        elif action == int(Action.TURN_LEFT):
            self.angle += math.pi/6
        elif action == int(Action.TURN_RIGHT):
            self.angle -= math.pi/6
        self.mem_query, self.mem_step = query, step
        if self.channel is None:
            self.connect_worker()
        current = self.views.get(self.yaw_bin)
        image = current[2] if step % 3 == 0 and current and current[1] == step else None
        self.channel.send(dict(task="observe", reset=reset, step=step, query=query,
                               xz=self.xy, heading=[math.cos(self.angle),math.sin(self.angle)],
                               collided=blocked, frame=image))
        if not self.channel.poll(180):
            raise RuntimeError("current observation timed out")
        record = self.channel.recv()
        if record["step"] != step:
            raise RuntimeError("out-of-order memory update")
        self.emit("memory_updates.jsonl", **record)
        return result
