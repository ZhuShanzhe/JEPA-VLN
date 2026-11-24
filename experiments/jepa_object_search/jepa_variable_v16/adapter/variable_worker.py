"""Read actual camera observations and rank reachable variable-length native paths."""
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import sys
import numpy as np


PROJECT = Path(os.environ.get("NAV_ROOT", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(PROJECT / p) for p in ("jepa_variable_v16/model", "jepa_evidence_v15/adapter",
                                          "jepa_env_v13/models", "jepa_env_v13/online")]
from env_routes import EnvironmentRoutes
from evidence_policy import claim_rays, clue_deficit
from scene_memory import SceneMemory
from variable_evidence import VariableEvidence
from variable_policy import score_routes


class VariableWorker:
    def __init__(self, project, target):
        self.model = EnvironmentRoutes(project, 11)
        if target not in self.model.raw["p_target_given_room"]:
            raise ValueError("unsupported target category")
        calibration = json.loads((project / "jepa_env_v13/online/cue_calibration.json").read_text())
        self.memory = SceneMemory(self.model.old.remem(project), calibration)
        self.evidence = VariableEvidence(project)
        self.target = target
        self.step = 0
        self.turn = 0
        self.claimed = set()
        self.last_report = -1

    def rank(self, msg):
        trace = msg["history"]
        first, last = self.step + 1, int(msg["step"])
        reports = msg["report_steps"]
        if not trace or [row[0] for row in trace] != list(range(first, last + 1)):
            raise ValueError("missing or duplicated observations")
        if any(t < first or t > last for t in reports):
            raise ValueError("report outside new observation window")
        self.claimed.update(claim_rays(self.memory, trace, reports))
        if reports:
            self.last_report = max(reports)
        images = [row[3] for row in trace if row[3] is not None]
        observations = iter(())
        if images:
            rooms, cues = self.model.observe_batch(np.asarray(images), self.target)
            observations = iter(zip(rooms, cues))
        for _, action, blocked, image in trace:
            room, cue = next(observations) if image is not None else (None, None)
            self.memory.observe(action, blocked, room, None if cue is None else float(cue))
        self.step = last
        self.turn += 1
        plans = [np.asarray(plan, dtype=np.int64) - 1 for plan in msg["plans"]]
        forecast = self.evidence.predict(self.model, np.asarray(msg["views"]), self.target, plans)
        # SceneMemory accepts rectangular batches; evaluate each real path in isolation.
        components = [self.memory.components(plan[None, :]) for plan in plans]
        memory = [float(item[0][0]) for item in components]
        supply = [float(item[3][0]) for item in components]
        deficit, rays = clue_deficit(self.memory, self.claimed, self.last_report)
        scores = score_routes(memory, forecast["evidence_rate"], supply, deficit, int(msg["remaining"]))
        return {**scores, "evidence_probability": np.asarray(forecast["evidence_probability"]).tolist(),
                "evidence_rate": np.asarray(forecast["evidence_rate"]).tolist(),
                "evaluated_steps": np.asarray(forecast["evaluated_steps"]).tolist(),
                "tap_steps": forecast["tap_steps"],
                "forecast_source": forecast["source"], "ranking_turn": self.turn,
                "available_clue_rays": rays, "reported_ray_count": len(self.claimed),
                "last_report_step": self.last_report, "evidence_policy": "variable_goal_evidence_v16"}


if __name__ == "__main__":
    stream = Connection(int(sys.argv[1]))
    worker = VariableWorker(PROJECT, sys.argv[2])
    while True:
        try:
            message = stream.recv()
        except EOFError:
            break
        stream.send(worker.rank(message))
