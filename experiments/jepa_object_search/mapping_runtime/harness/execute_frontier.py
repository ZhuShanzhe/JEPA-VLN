"""Allow an external online scorer to choose among the author's reachable frontiers."""
import copy
import importlib
import json
import os
from pathlib import Path

import numpy as np

from most_frontier_probe import ProbeAgent, check_scores


class FrontierSelector(ProbeAgent):
    def __init__(self):
        super().__init__()
        if self.scorer or self.advise:
            raise ValueError("Configure the execution scorer separately from the VLM advisory probe")
        self.predict = None
        spec = os.environ.get("MOST_EXEC_SCORER")
        if spec:
            module, name = spec.split(":", 1)
            self.predict = getattr(importlib.import_module(module), name)

    def _select(self, observation, result, rows):
        if self.predict is None:
            return result.target_id
        scores = check_scores(self.predict(
            rgb=np.array(observation.rgb, copy=True), goal_text=str(observation.goal_text),
            candidates=copy.deepcopy(rows), vlm_selected=result.target_id,
            prior_report_steps=[int(c.step) for c in self.instance_store.report_claims]), rows)
        return max(scores, key=lambda k: (scores[k], k == result.target_id, -int(k[1:])))

    def _record_selection(self, observation, original, executed, rows, applied):
        self._write_probe(observation, "execution_selection", original=original, executed=executed,
                          candidate_count=len(rows), applied=bool(applied))

    def _apply_decider_steering(self, observation, result):
        if result.action != "GOTO_FRONTIER" or result.validation != "ok":
            return super()._apply_decider_steering(observation, result)
        rows = self._rows()
        ids = {r["id"] for r in rows}
        if len(rows) < 2 or result.target_id not in ids:
            return super()._apply_decider_steering(observation, result)
        chosen = self._select(observation, result, rows)
        if chosen not in ids:
            raise ValueError("Selected frontier is outside the author's current candidate table")
        command = copy.copy(result)
        command.target_id = chosen
        applied = super()._apply_decider_steering(observation, command)
        self._record_selection(observation, result.target_id, chosen, rows, applied)
        return applied


def engineering_alternate(*, rgb, goal_text, candidates, vlm_selected, prior_report_steps):
    if not goal_text or rgb.ndim != 3 or rgb.shape[-1] != 3 or not isinstance(prior_report_steps, list):
        raise ValueError("Incomplete online scorer inputs")
    other = next((r["id"] for r in candidates if r["id"] != vlm_selected and r["nominal_action_ids"]),
                 vlm_selected)
    return {r["id"]: float(r["id"] == other) for r in candidates}


class BranchExecutionCheck(FrontierSelector):
    def __init__(self):
        super().__init__()
        if self.predict is not engineering_alternate:
            raise ValueError("Set MOST_EXEC_SCORER=execute_frontier:engineering_alternate for this check")
        self.redirect = None
        self.released_actions = []
        self.start_gps = None

    def _select(self, observation, result, rows):
        return super()._select(observation, result, rows) if self.redirect is None else result.target_id

    def _record_selection(self, observation, original, executed, rows, applied):
        super()._record_selection(observation, original, executed, rows, applied)
        if applied and original != executed:
            self.start_gps = np.array(observation.gps, dtype=float, copy=True)
            self.redirect = dict(original=original, executed=executed,
                                 step=int(observation.step_count), candidate_count=len(rows))

    def act(self, observation):
        if self.redirect and 1 in self.released_actions:
            displacement = float(np.linalg.norm(np.asarray(observation.gps) - self.start_gps))
            if displacement > 0.1:
                receipt = dict(scope="engineering only; early finish; no model score",
                               selection=self.redirect, ending_step=int(observation.step_count),
                               actions=list(self.released_actions), displacement_m=displacement)
                path = Path(self.output_dir).parent / "branch_execution.json"
                path.write_text(json.dumps(receipt, indent=2) + "\n")
                return 0
        action = super().act(observation)
        if self.redirect:
            self.released_actions.append(int(action))
        return action
