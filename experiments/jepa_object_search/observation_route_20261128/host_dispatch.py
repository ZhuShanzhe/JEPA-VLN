"""Execute only observation prefixes literally; native regional paths keep their follower."""
from contextlib import contextmanager
from unittest.mock import patch


class ObservationActions(list):
    """Locally scored primitive sequence (1 forward, 2 left, 3 right)."""
    def __init__(self, actions):
        if not 1 <= len(actions) <= 4 or any(type(a) is not int or a not in (1,2,3) for a in actions):
            raise ValueError("observation route must contain one to four native actions")
        super().__init__(actions)


class Prefix:
    def __init__(self, actions):
        self.actions = iter(ObservationActions(actions))

    def next(self):
        action = next(self.actions, None)
        return action, action is None


@contextmanager
def exact_physical_follow(physical):
    original = physical.ViewFollower

    class WithObservationPrefix(original):
        def __init__(self, pose, scale, actions, *args, **kwargs):
            super().__init__(pose,scale,actions,*args,**kwargs)
            self.observation_prefix = Prefix(actions) if isinstance(actions,ObservationActions) else None

        def next_action(self):
            if self.observation_prefix is not None:
                return self.observation_prefix.next()
            return super().next_action()

    with patch.object(physical, "ViewFollower", WithObservationPrefix):
        yield
