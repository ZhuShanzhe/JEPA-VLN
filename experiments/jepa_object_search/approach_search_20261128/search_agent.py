"""MOST exploration hook using the native decision/report loop and a local prior scorer."""
from multiprocessing.connection import Connection
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import MethodType

from map_views import candidates, visible_map_surfaces
from enhanced_search import PriorEvidenceMemory
from global_frontiers import frontier_plans
from motion_memory import MotionMemory
from jepa_pilot_agent import JepaFrontier
from most_frontier_probe import ProbeAgent, candidate_rows
from point_executor import FocusViewMixin
from target_words import requested_class


class PriorApproachAgent(FocusViewMixin, JepaFrontier):
    def init_episode(self, obs):
        super().init_episode(obs)
        self.search = PriorEvidenceMemory()
        self.motion_memory = MotionMemory()
        self._motion_source = None
        self.map_frame = None
        self._selected_site = None
        from placement_prior import PlacementPrior
        db = PlacementPrior.load(Path(os.environ["NAV_ROOT"]) / "prior_approach_20261128/learned_prior.json")
        self.word = requested_class(obs.goal_text, db.targets) if getattr(obs, "goal_type", "") != "image" else None
        self.emit("prior_target", target=self.word, active=self.word is not None)

    def _connect(self):
        project = Path(os.environ["NAV_ROOT"])
        python = project.parent / "jepa_instance_bench_20261016/policy_env/bin/python"
        left, right = socket.socketpair()
        try:
            env = dict(os.environ)
            for name in ("PYTHONPATH", "PYTHONHOME"):
                env.pop(name, None)
            env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            with (Path(self.output_dir) / "prior_worker.log").open("ab") as log:
                self.worker = subprocess.Popen([str(python), "-B", str(Path(__file__).with_name("prior_worker.py")),
                                                str(right.fileno()), self.word], env=env, pass_fds=(right.fileno(),),
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            self.channel = Connection(left.detach())
        finally:
            left.close()
            right.close()

    def _current_frame(self):
        scale = self._metric_scale_value()
        if not scale or not math.isfinite(scale) or scale <= 0:
            return None
        pose = self._estimated_current_pose()
        if pose is None or not all(map(math.isfinite, pose)):
            return None
        rev = self._map_revision_tuple(self._frontier_server_revision)
        loop = rev[2] if rev else None
        if self.map_frame is not None:
            old_scale, old_loop = self.map_frame
            if not math.isclose(old_scale, scale, rel_tol=.05) or (old_loop is not None and loop is not None and old_loop != loop):
                self.search = PriorEvidenceMemory()
                self.motion_memory = MotionMemory()
                self._motion_source = None
                self._selected_site = None
                self.map_frame = None
                self.emit("prior_map_reset", reason="scale_or_loop_closure")
        if self.map_frame is None:
            self.map_frame = (scale, loop)
        elif self.map_frame[1] is None and loop is not None:
            self.map_frame = (self.map_frame[0], loop)
        return pose, scale

    def _feed_frame(self, obs):
        old = self.observed_step
        result = super()._feed_frame(obs)
        self.history.clear()  # this worker uses actual scan frames, not the historical V13 message stream
        if old < self.observed_step and self.word and self.pitch in (0, 1) and obs.rgb is not None:
            frame = self._current_frame()
            blocked = getattr(self, "_last_motion_failed", False) and obs.previous_action == 1
            if blocked and self._motion_source is not None:
                start, old_scale = self._motion_source
                self.motion_memory.record(start, old_scale, self.now)
            self._motion_source = frame
            if frame is not None:
                (x, y, yaw), scale = frame
                grid = getattr(self, "_frontier_grid", None)
                surfaces = visible_map_surfaces(grid, (x,y,yaw), scale) if grid is not None else None
                self.search.mark_view((x * scale, y * scale), yaw, surfaces)
            if self._selected_site and blocked:
                self.search.mark_blocked(*self._selected_site)
        return result

    def _finish_focus(self, reason):
        self._selected_site = None
        return super()._finish_focus(reason)

    def _activate_frontier(self, obs, cluster, source):
        self._end_scan("native_reactivation")
        self._finish_focus("native_reactivation")
        applied = ProbeAgent._activate_frontier(self, obs, cluster, source)
        if not applied or self.word is None or self.pitch not in (0, 1) or self._frontier_grid is None:
            return applied
        if not (source.startswith("decider GOTO_FRONTIER ") or source in ("autonomous EXPLORE", "deterministic explorer")):
            return applied
        if obs.max_steps - self.now <= len(self.scan_actions) or self._current_frame() is None:
            return applied
        fl = self._explore_follower
        scan = dict(start=self.now, yaw=self.yaw, carrier=fl, next=fl.next_action,
                    original_obj=cluster, received=0, sent=0, waiting=None, views=[], frame_steps=[])
        self.scan = scan
        def move(_):
            if self.scan is not scan or scan["waiting"] is not None:
                raise RuntimeError("invalid scan action acknowledgement")
            action = self.scan_actions[scan["sent"]]
            scan["sent"] += 1
            scan["waiting"] = self.now, action
            return action, False
        fl.next_action = MethodType(move, fl)
        return applied

    def _rank_after_scan(self, obs, scan):
        self._end_scan("complete")
        if self.yaw != scan["yaw"] or len(scan["views"]) != 3:
            return
        frame = self._current_frame()
        if frame is None:
            return
        pose, scale = frame
        fl = self._explore_follower
        grid = self._frontier_grid
        if fl is None or grid is None or not math.isclose(scale, fl.scale, rel_tol=.03):
            return
        clusters = self._last_frontier_clusters
        table = candidate_rows(clusters, pose, scale, self.reach_m)
        native = next((f"f{i}" for i, c in enumerate(clusters) if c is scan["original_obj"]), None)
        plans = [(r["id"], r["nominal_action_ids"]) for r in sorted(table, key=lambda r: r["id"] != native)
                 if r["nominal_action_ids"]]
        grid = self.motion_memory.grid(grid, scale, self.now)
        remaining = obs.max_steps - self.now
        plans.extend(frontier_plans(grid, pose, scale, remaining))
        rows = candidates(grid, pose, scale, plans, remaining)
        if not rows:
            self.emit("prior_route_skip", reason="no_known_free_approach")
            return
        if self.channel is None:
            self._connect()
        self.channel.send(dict(views=scan["views"], plans=[r["actions"] for r in rows]))
        if not self.channel.poll(240):
            raise TimeoutError("prior worker did not reply")
        scores = self.channel.recv()
        if len(scores["priors"]) != len(rows) or len(scores["current"]) != 3 or len(scores["evidence"]) != len(rows):
            raise ValueError("prior worker returned different candidate table")
        x, y, yaw = pose
        for offset, p in zip((-1, 0, 1), scores["current"]):
            self.search.update_context((x * scale, y * scale), yaw + offset * math.pi / 6, p)
        for row, p, e in zip(rows, scores["priors"], scores["evidence"]):
            row["prior"] = p
            row["evidence"] = e
        index, audit = self.search.choose(rows)
        choice = rows[index]
        bound = self.start_focus_view(obs, choice["actions"], choice["key"])
        if bound:
            self._selected_site = (choice["position"], choice["heading"])
            if native is None or choice["key"].split("/")[0] != native:
                self._active_frontier_key = self._active_branch_key = None
                # The original frontier was provisionally activated only for the scan.
                if self._recent_frontiers and self._recent_frontiers[-1][1] == scan["start"]:
                    self._recent_frontiers.pop()
        self.emit("prior_route_choice", selected=choice["key"], source=choice["source"], bound=bound,
                  phase=audit["phase"], candidates=audit["candidates"], goal=self.word)
