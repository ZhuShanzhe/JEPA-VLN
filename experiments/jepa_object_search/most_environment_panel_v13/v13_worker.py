"""Private process for the frozen HSSD scorer; callers supply only observed inputs."""
import json
from multiprocessing.connection import Connection
from pathlib import Path
import sys
import numpy as np


class Worker:
    def __init__(self, project, target):
        root = project / "jepa_env_v13"
        sys.path[:0] = [str(root / n) for n in ("models", "online", "horizon")]
        from env_routes import EnvironmentRoutes
        from run_search import SETTINGS, pick_route
        from scene_memory import SceneMemory
        from train_value import ValueResidual
        self.model = EnvironmentRoutes(project, 11)
        if target not in self.model.raw["p_target_given_room"]:
            raise ValueError("target has no trained room prior")
        self.head = ValueResidual.load(project)
        self.config = SETTINGS[json.loads((root / "online/online_selection.json").read_text())["setting"]]
        calibration = json.loads((root / "online/cue_calibration.json").read_text())
        self.memory = SceneMemory(self.model.old.remem(project), calibration)
        self.choose = pick_route
        self.target = target
        self.history = []
        self.step = 0

    def rank(self, msg):
        trace = msg["history"]
        if not trace or [int(t[0]) for t in trace] != list(range(self.step + 1, msg["step"] + 1)):
            raise ValueError("missing or duplicated observation steps")
        images = [t[3] for t in trace if t[3] is not None]
        sensed, cue = self.model.observe_batch(np.asarray(images), self.target)
        itr = iter(zip(sensed, cue))
        for step, act, blocked, image in trace:
            env, c = next(itr) if image is not None else (None, None)
            self.memory.observe(act, blocked, env, None if c is None else float(c))
        self.step = msg["step"]
        plans = np.asarray(msg["plans"], dtype=np.int64) - 1
        views = np.asarray(msg["views"])
        obs, signals = self.model.observe_batch(views, self.target)
        out = self.model.live(views, self.target, plans)
        room, prior = out["env_pred_jepa"], out["prior"]
        past = np.mean([h[0] for h in self.history], 0) if self.history else obs.mean(0)
        past_c = float(np.mean([h[1] for h in self.history])) if self.history else signals.mean()
        horizon = min(4, max(1, int(msg["remaining"]) // 24))
        delta = self.head.route_delta(out["jepa"], room, prior, obs, signals, plans, horizon, past, past_c)
        terms = self.memory.components(plans)
        base = out["jepa"] + delta
        alpha, gain, explore, _ = self.config
        affinity = room @ prior
        final = self.memory.apply(base + alpha*(affinity-affinity.mean()), room, terms,
                                  gain=gain, exploration=explore)
        choice, scout = self.choose(base, room, prior, terms, len(self.history), self.config)
        self.history.append((obs.mean(0), float(signals.mean())))
        return dict(index=int(choice), expected_rooms=[self.model.rooms[k] for k in room.argmax(1)],
                    prior_room=self.model.rooms[int(prior.argmax())], ranking_turn=len(self.history),
                    model_horizon=horizon, v11_scores=out["v11"].tolist(),
                    direct_scores=out["direct"].tolist(), jepa_scores=out["jepa"].tolist(),
                    horizon_delta=delta.tolist(), final_scores=final.tolist(), scout_changed=bool(scout))


if __name__ == "__main__":
    channel = Connection(int(sys.argv[1]))
    agent = Worker(Path(__file__).resolve().parents[2], sys.argv[2])
    while True:
        try:
            request = channel.recv()
        except EOFError:
            break
        channel.send(agent.rank(request))
