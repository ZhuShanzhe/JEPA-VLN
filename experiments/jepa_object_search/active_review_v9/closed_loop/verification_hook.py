"""Supplement active exploration with goal-conditioned, interruptible verification."""
from benchmark_api import Action
from base_v8 import JepaFrontier as RouteAgent


class JepaFrontier(RouteAgent):
    def __init__(self):
        super().__init__()
        self.last_probe = -100
        self.allowed_after = 0
        self.turns = self.reviews = self.balance = 0
        self.unwind = False
        self.controller = None

    def forget(self):
        self.balance = 0
        self.unwind = False
        self.controller = None

    def _execute_decider_result(self, observation, result, event, images=None):
        self.forget()
        return super()._execute_decider_result(observation, result, event, images)

    def turn(self, step, offset, reason):
        action = int(Action.TURN_LEFT if offset > 0 else Action.TURN_RIGHT)
        self._explore_follower.dead_reckon(action)
        self.balance += offset
        self.emit("verification_events.jsonl", event=reason, step=step, action=action,
                  balance=self.balance)
        return action

    def _caption_hit_decision(self, obs):
        native = super()._caption_hit_decision(obs)
        if native is not None:
            self.forget()
            return native
        step = int(obs.step_count)
        if self.controller is not self._explore_follower:
            self.forget()
        if self.unwind and self.balance:
            action = self.turn(step, -1 if self.balance > 0 else 1, "restore_heading")
            if not self.balance:
                self.forget()
            return action
        if (self._explore_follower is None or self.mode != "explore" or self._scanning
                or self._adjusting or self._last_motion_failed or self.pitch not in (0, 1)
                or step == self.last_probe or step + 5 >= int(obs.max_steps)):
            return None
        if not self.balance and (step < self.allowed_after or step - self.last_probe < 3):
            return None
        if self.turns >= 12 and self.reviews >= 6 and not self.balance:
            return None
        # Only frames observed from this exact location, sorted oldest to newest.
        views = sorted([(stamp, (b-self.yaw_bin+6)%12-6, image) for b,(moves,stamp,image)
                        in self.views.items() if moves == self.forward_count and stamp <= step])
        views = views[-3:]
        if not views or views[-1][:2] != (step, 0):
            return None
        self.last_probe = step
        if self.channel is None:
            self.connect_worker()
        self.channel.send(dict(task="review_v9", context=[x[2] for x in views], headings=[x[1] for x in views],
                               query=str(obs.goal_text)))
        if not self.channel.poll(180):
            raise RuntimeError("V9 scoring worker timed out")
        response = self.channel.recv()
        choice = response["action"]
        if choice not in ("left", "right", "review", "continue"):
            raise ValueError("Invalid V9 decision")
        self.emit("verification_events.jsonl", event="predict", step=step,
                  branch=choice, canonical=response.get("canonical"), supported=response.get("supported"), source_steps=[x[0] for x in views],
                  headings=[x[1] for x in views],
                  current_score=response.get("current_score"),turn_score=response.get("turn_score"))
        if choice == "review" and self.reviews < 6:
            self.reviews += 1
            self.allowed_after = step + 12
            self.forget()
            self.close_motion("supplementary_review")
            self._explore_follower = None
            self._active_frontier_key = self._active_branch_key = None
            self.emit("verification_events.jsonl",event="request_native_review",step=step)
            # Native decision prompt has the complete goal, current image and mapping tools.
            return self._choose_high_level_target(obs, "world_state_updated")
        direction = {"left":1, "right":-1}.get(choice, 0)
        if (direction and self.turns < 12 and abs(self.balance) < 2
                and (not self.balance or self.balance * direction > 0)):
            self.controller = self._explore_follower
            self.turns += 1
            self.allowed_after = step + 12
            return self.turn(step, direction, "verification_turn")
        if self.balance:
            self.unwind = True
            action = self.turn(step, -1 if self.balance > 0 else 1, "restore_heading")
            if not self.balance:
                self.forget()
            return action
        return None
