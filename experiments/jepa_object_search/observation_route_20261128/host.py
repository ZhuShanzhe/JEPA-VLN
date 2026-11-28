"""Optional native MOST-Nav arm. The native report/adjust/finish priority is inherited."""
from host_dispatch import ObservationActions, Prefix
from multi_agent import MultiViewAgent
from policy import ObservationRoutePolicy


class ObservationRouteAgent(MultiViewAgent):
    policy_class = ObservationRoutePolicy

    def start_focus_view(self, observation, actions, tag):
        bound = super().start_focus_view(observation, actions, tag)
        if not bound or not isinstance(actions, ObservationActions):
            return bound
        follower = self._focus_view[0]
        prefix = Prefix(actions)
        follower.observation_route_prefix = True

        def take_scored_action():
            action, done = prefix.next()
            if done:
                follower.set_path(None)
                follower.outcome = "observation_prefix_finished"
            return action, done
        follower.next_action = take_scored_action
        return True

    def _explore_action(self, observation):
        active = self._focus_view
        if (active and getattr(active[0],"observation_route_prefix",False)
                and self._last_motion_failed and observation.previous_action==1):
            self._local_blocked = True
            self._release_owned_carrier("observation_prefix_blocked")
        return super()._explore_action(observation)
