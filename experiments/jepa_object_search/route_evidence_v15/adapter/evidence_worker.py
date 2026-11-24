"""Score the same observed scan and executable candidate plans with the V15 head."""
from multiprocessing.connection import Connection
from pathlib import Path
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT/"jepa_env_v13/most_integration"), str(PROJECT/"jepa_evidence_v15/train")]
from evidence_head import EvidencePredictor
from evidence_policy import claim_rays, clue_deficit, score_routes
from v13_worker import Worker


class EvidenceWorker(Worker):
    def __init__(self, project, target):
        super().__init__(project, target)
        self.evidence = EvidencePredictor(project)
        self.claimed = set()
        self.last_report = -1

    def rank(self, msg):
        reports = msg["report_steps"]
        first, last = self.step + 1, int(msg["step"])
        if not all(first <= t <= last for t in reports):
            raise ValueError("report outside new observation window")
        self.claimed.update(claim_rays(self.memory, msg["history"], reports))
        if reports:
            self.last_report = max(reports)
        old = super().rank(msg)
        plans = np.asarray(msg["plans"], dtype=np.int64) - 1
        p = self.evidence.predict(self.model, np.asarray(msg["views"]), self.target, plans)
        supply = self.memory.components(plans)[3]
        deficit, recent_rays = clue_deficit(self.memory, self.claimed, self.last_report)
        new = score_routes(old["final_scores"], p, supply, deficit, int(msg["remaining"]))
        return {**old, **new, "v13_index": old["index"], "v13_same_input_index": old["index"],
                "v13_scores": old["final_scores"], "available_clue_rays": recent_rays,
                "reported_ray_count": len(self.claimed), "last_report_step": self.last_report,
                "evidence_policy": "goal_evidence_v15"}


if __name__ == "__main__":
    stream = Connection(int(sys.argv[1]))
    worker = EvidenceWorker(PROJECT, sys.argv[2])
    while True:
        try:
            message = stream.recv()
        except EOFError:
            break
        stream.send(worker.rank(message))
