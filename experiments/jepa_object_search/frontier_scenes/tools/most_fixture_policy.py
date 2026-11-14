"""Two deterministic action sequences exercising the unmodified public MOST agent API."""
import numpy as np

class ScriptedProbe:
    def __init__(self, **kwargs):
        self.reset()

    def reset(self):
        self.i=0

    def act(self, observation):
        assert observation.rgb.shape[:2] == (1280,1280) and observation.rgb.std()>3
        assert observation.depth is not None and np.isfinite(observation.depth).all()
        assert observation.goal_text and observation.target_count == 2
        # Start looking along +X. At each report location turn 90 degrees to face its chair.
        repeat=[6] if observation.episode_id.endswith("repeat") else []
        mid_report=[6] if repeat else []
        seq=[1]*2+[2]*3+[6]+repeat+[3]*3+[1]*9+mid_report+[1]*9+[2]*3+[6,0]
        action=seq[self.i]
        self.i+=1
        return action
