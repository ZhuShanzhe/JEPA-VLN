"""MOST agent entry point: variable_agent:VariableFrontier."""
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import MethodType


PROJECT = Path(os.environ.get("NAV_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(PROJECT / "jepa_evidence_v15/adapter"))
from v15_agent import EvidenceChannel, EvidenceFrontier
from most_frontier_probe import ProbeAgent
from variable_routes import build_table
from delayed_forks import DeferredForks


class VariableFrontier(DeferredForks, EvidenceFrontier):
    def _activate_frontier(self, observation, cluster, source):
        self._cancel_delay("native_reactivation")
        self._close_commit("native_reactivation")
        self._end_scan("native_reactivation")
        self._end_route("native_reactivation")
        index = next((i for i, item in enumerate(self._last_frontier_clusters)
                      if item is cluster), None)
        applied = ProbeAgent._activate_frontier(self, observation, cluster, source)
        if not applied or index is None:
            return applied
        original = f"f{index}"
        rows = self._rows()
        available = int(observation.max_steps) - self.now - len(self.scan_actions)
        entries, reason = build_table(rows, original, available)
        if not (source.startswith("decider GOTO_FRONTIER ") or
                source in ("autonomous EXPLORE", "deterministic explorer")):
            reason = "native_action_not_explore"
        elif self.word is None:
            reason = "unsupported_target_category"
        elif self.pitch not in (0, 1):
            reason = "off_level"
        elif self._explore_follower is None:
            reason = "no_active_follower"
        ids = [entry["id"] for entry in entries]
        plans = [entry["plan"] for entry in entries]
        self.emit("eligibility", original=original, source=str(source), reason=reason, ids=ids,
                  native_candidate_count=len(rows), feasible_distinct_count=len(plans),
                  planned_steps=[len(plan) for plan in plans], post_scan_budget=available,
                  room_prior_supported=self.word is not None)
        if reason:
            if reason == "no_distinct_spatial_branch_in_horizon":
                self._schedule_delay(observation, index, source, available)
            return applied
        self._begin_scan(original, ids, plans, source)
        return applied

    def _begin_scan(self, original, ids, plans, source):
        carrier = self._explore_follower
        scan = dict(start=self.now, yaw=self.yaw, carrier=carrier, next=carrier.next_action,
                    original=original, ids=ids, plans=plans, source=source, received=0, sent=0,
                    snapshot=tuple(self._last_frontier_clusters), waiting=None, views=[], frame_steps=[])
        self.scan = scan

        def move(_):
            if self.scan is not scan or scan["waiting"] is not None or scan["sent"] >= len(self.scan_actions):
                raise RuntimeError("inconsistent native scan dispatch")
            action = self.scan_actions[scan["sent"]]
            scan["sent"] += 1
            scan["waiting"] = (self.now, action)
            return action, False

        carrier.next_action = MethodType(move, carrier)
        self.emit("scan_start", original=original, ids=ids, scan_actions=list(self.scan_actions))

    def _connect(self):
        left, right = socket.socketpair()
        try:
            env = dict(os.environ)
            for key in ("PYTHONPATH", "PYTHONHOME"):
                env.pop(key, None)
            env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            python = PROJECT.parent / "jepa_instance_bench_20261016/policy_env/bin/python"
            worker = Path(__file__).with_name("variable_worker.py")
            with (Path(self.output_dir) / "v16_worker.log").open("ab") as log:
                self.worker = subprocess.Popen([str(python), "-B", str(worker), str(right.fileno()), self.word],
                                               env=env, pass_fds=(right.fileno(),), stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT)
            self.channel = EvidenceChannel(Connection(left.detach()), self.observed_reports)
        finally:
            left.close()
            right.close()
