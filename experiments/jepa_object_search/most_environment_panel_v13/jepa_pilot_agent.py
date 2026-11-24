"""Rerank three genuine MOST exploration frontiers after an interruptible physical scan."""
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import MethodType

from benchmark_api import Action
from most_frontier_probe import ProbeAgent
from v13_inputs import camera, category, plan_table


class JepaFrontier(ProbeAgent):
    scan_actions = (3, 2, 2, 3)

    def __init__(self):
        super().__init__()
        self.episode = None
        self.scan = self.route = self.channel = self.worker = None
        self.observed_step = -1

    def emit(self, event, **fields):
        with (Path(self.output_dir) / "v13_routes.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(event=event, step=self.now, **fields), allow_nan=False) + "\n")

    def init_episode(self, o):
        if self.channel is not None:
            self.channel.close()
            self.worker.wait(timeout=30)
        self.channel = self.worker = self.scan = self.route = None
        self.episode, self.now = str(o.episode_id), int(o.step_count)
        if self.now != 0:
            raise ValueError("missing episode start")
        self.word = category(str(o.goal_text))
        self.pitch = self.yaw = 0
        self.history = []
        self.observed_step = -1
        self.emit("episode", model_target=self.word, room_prior_supported=self.word is not None,
                  attribute_unmodelled=True, camera="native rectified RGB, 128x128, 90 deg")

    def _connect(self):
        left, right = socket.socketpair()
        try:
            env = dict(os.environ)
            for name in ("PYTHONPATH", "PYTHONHOME"):
                env.pop(name, None)
            env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            python = Path(__file__).resolve().parents[3] / "jepa_instance_bench_20261016/policy_env/bin/python"
            with (Path(self.output_dir) / "v13_worker.log").open("ab") as log:
                self.worker = subprocess.Popen([str(python), "-B", str(Path(__file__).with_name("v13_worker.py")),
                                               str(right.fileno()), self.word], env=env,
                                              pass_fds=(right.fileno(),), stdin=subprocess.DEVNULL,
                                              stdout=log, stderr=subprocess.STDOUT)
            self.channel = Connection(left.detach())
        finally:
            left.close()
            right.close()

    def _end_scan(self, reason):
        if self.scan is None:
            return
        s = self.scan
        s["carrier"].next_action = s["next"]
        self.scan = None
        self.emit("scan_end", start=s["start"], reason=reason, dispatched=s["sent"],
                  physical_turns=s["received"], heading_restored=self.yaw == s["yaw"])

    def _end_route(self, reason):
        if self.route is not None:
            r = self.route
            self.emit("route_end", start=r["start"], reason=reason, original=r["original"],
                      selected=r["selected"], matched_steps=r["matched"],
                      first_difference=r["difference"], divergence_reached=r["diverged"])
            self.route = None

    def _feed_frame(self, o):
        result = super()._feed_frame(o)
        if self.now == self.observed_step:
            return result
        if self.now != self.observed_step + 1:
            raise ValueError("missing simulator observation")
        self.observed_step = self.now
        if self.now == 0:
            return result
        a = int(o.previous_action)
        self.yaw = (self.yaw + int(a == 2) - int(a == 3)) % 12
        self.pitch += int(a == int(Action.LOOK_UP)) - int(a == int(Action.LOOK_DOWN))
        scan = self.scan
        fresh = scan is not None and scan["waiting"] is not None
        if fresh:
            ts, expected = scan["waiting"]
            if self.now != ts + 1 or a != expected:
                self._end_scan("native_interrupted_scan")
                fresh = False
        assign = fresh and scan["received"] < 3
        frame = None
        if self.word and self.pitch in (0, 1) and (assign or self.now % 6 == 0):
            frame = camera(o.rgb, self.pitch)
        if self.word:
            self.history.append((self.now, a - 1 if a in (1, 2, 3) else -1,
                                 bool(a == 1 and self._last_motion_failed), frame))
        if fresh:
            scan["received"] += 1
            scan["waiting"] = None
            if assign:
                if frame is None:
                    self._end_scan("off_level")
                else:
                    scan["views"].append(frame)
                    scan["frame_steps"].append(self.now)
        r = self.route
        if r is not None and self.now > r["start"]:
            k = r["matched"]
            if a != r["plan"][k]:
                self._end_route("native_command_differs_from_nominal")
            else:
                r["matched"] += 1
                if k == r["difference"]:
                    r["diverged"] = True
                    self.emit("route_progress", start=r["start"], offset=k, action=a,
                              divergence_reached=True, observed_motion_blocked=bool(a == 1 and self._last_motion_failed))
                if r["matched"] == len(r["plan"]):
                    self._end_route("observed_all_24_commands")
        return result

    def act(self, observation):
        self.now = int(observation.step_count)
        if self.episode != str(observation.episode_id) or self.now < self.observed_step:
            self.init_episode(observation)
        action = super().act(observation)
        if self.scan and self._explore_follower is not self.scan["carrier"]:
            self._end_scan("native_follower_replaced")
        if self.route and self._explore_follower is not self.route["carrier"]:
            self._end_route("native_follower_replaced")
        if int(action) == int(Action.FINISH) or self.now + 1 >= int(observation.max_steps):
            self._end_scan("episode_end")
            self._end_route("episode_end")
        return action

    def _execute_decider_result(self, observation, result, event, images=None):
        self._end_scan("native_decision_" + str(result.action))
        self._end_route("native_decision_" + str(result.action))
        return super()._execute_decider_result(observation, result, event, images)

    def _activate_frontier(self, observation, cluster, source):
        self._end_scan("native_reactivation")
        self._end_route("native_reactivation")
        idx = next((i for i, c in enumerate(self._last_frontier_clusters) if c is cluster), None)
        applied = super()._activate_frontier(observation, cluster, source)
        if not applied or idx is None:
            return applied
        host = "f%d" % idx
        rows = self._rows()
        ids, plans, reason = plan_table(rows, host)
        if not (source.startswith("decider GOTO_FRONTIER ") or source == "autonomous EXPLORE"
                or source == "deterministic explorer"):
            reason = "native_action_not_explore"
        elif self.word is None:
            reason = "unsupported_target_category"
        elif self.pitch not in (0, 1):
            reason = "off_level"
        elif int(observation.max_steps) - self.now < 28:
            reason = "remaining_budget_below_28"
        self.emit("eligibility", original=host, source=str(source), reason=reason, ids=ids,
                  native_candidate_count=len(rows), room_prior_supported=self.word is not None)
        if reason:
            return applied
        carrier = self._explore_follower
        self.scan = dict(start=self.now, yaw=self.yaw, carrier=carrier, next=carrier.next_action,
                         original=host, ids=ids, plans=plans, source=source, received=0, sent=0,
                         snapshot=tuple(self._last_frontier_clusters), waiting=None, views=[], frame_steps=[])
        def move(_):
            s = self.scan
            if s is None or s["waiting"] is not None or s["sent"] >= 4:
                raise RuntimeError("inconsistent native scan dispatch")
            cmd = self.scan_actions[s["sent"]]
            s["sent"] += 1
            s["waiting"] = (self.now, cmd)
            return cmd, False
        carrier.next_action = MethodType(move, carrier)
        self.emit("scan_start", original=host, ids=ids, scan_actions=list(self.scan_actions))
        return applied

    def _explore_follow(self, observation):
        s = self.scan
        if s and self._explore_follower is not s["carrier"]:
            self._end_scan("native_follower_replaced")
        elif s and s["received"] == 4:
            self._rank_after_scan(observation, s)
        return super()._explore_follow(observation)

    def _rank_after_scan(self, observation, s):
        if self.yaw != s["yaw"] or len(s["views"]) != 3:
            self._end_scan("scan_pose_or_frames_invalid")
            return
        current = self._last_frontier_clusters
        if len(s["snapshot"]) != len(current) or any(a is not b for a, b in zip(s["snapshot"], current)):
            self._end_scan("native_frontiers_refreshed")
            return
        self._end_scan("complete")
        if self.channel is None:
            self._connect()
        self.channel.send(dict(history=self.history, step=self.now, views=s["views"], plans=s["plans"],
                               remaining=int(observation.max_steps)-self.now))
        if not self.channel.poll(240):
            raise TimeoutError("frozen route scorer did not reply")
        scores = self.channel.recv()
        self.history = []
        k = int(scores["index"])
        pick, original = s["ids"][k], s["original"]
        chosen = s["plans"][k]
        difference = next((i for i, (a, b) in enumerate(zip(chosen, s["plans"][0])) if a != b), None)
        if k:
            # The original activation was provisional. Remove its scan-only cooldown entry.
            if (not self._recent_frontiers or self._recent_frontiers[-1][1] != s["start"] or
                not all(abs(float(a)-float(b)) < 1e-7 for a,b in
                        zip(self._recent_frontiers[-1][0],s["snapshot"][int(original[1:])]["world"][:2]))):
                raise RuntimeError("native frontier cooldown changed during scan")
            self._recent_frontiers.pop()
            if not super()._activate_frontier(observation, current[int(pick[1:])], s["source"]):
                raise RuntimeError("selected native frontier rejected")
        self.route = dict(carrier=self._explore_follower,start=self.now,matched=0,plan=chosen,
                          difference=difference,diverged=False,original=original,selected=pick)
        self.emit("rank", original=original, selected=pick, ids=s["ids"], plans_native=s["plans"],
                  frame_steps=s["frame_steps"], first_difference=difference, model_target=self.word,
                  room_prior_supported=True, attribute_unmodelled=True, **scores)
