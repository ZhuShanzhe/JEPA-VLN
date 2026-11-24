"""Reconsider genuine distant native forks only after traversing their common trunk."""
import math

from variable_routes import FORECAST_STEPS, branches_differ, build_table, floor_trace


def follow(follower_type, path, pose, scale, reach, steps):
    follower = follower_type(scale=scale, reach_m=reach)
    follower.set_path(path)
    follower.x, follower.y, follower.yaw = pose
    result = []
    for _ in range(steps):
        action, arrived = follower.next_action()
        if arrived or action not in (1, 2, 3):
            break
        result.append(int(action))
        follower.dead_reckon(action)
    return result


def trunk_plan(clusters, index, pose, follower, budget):
    """Use projected native geometry solely to schedule a later observation."""
    if budget <= FORECAST_STEPS:
        return None
    original = clusters[index]
    if original.get("key") is None:
        return None
    def project(cluster):
        return follow(type(follower), cluster["path"], pose, cluster["scale"], follower.reach_m, budget)
    host = project(original)
    if len(host) <= FORECAST_STEPS:
        return None
    trace = floor_trace(host)
    alternatives = set()
    forks = []
    for i, candidate in enumerate(clusters):
        if i == index or candidate.get("key") is None or candidate["key"] == original["key"]:
            continue
        distance = math.dist(candidate["world"][:2], original["world"][:2]) * follower.scale
        if distance < follower.reach_m:
            continue
        other = project(candidate)
        if other[:FORECAST_STEPS] != host[:FORECAST_STEPS] or not branches_differ(trace, floor_trace(other)):
            continue
        fork = next(i for i, (a, b) in enumerate(zip(host, other)) if a != b)
        forks.append(fork)
        alternatives.add(tuple(candidate["key"]))
    if not forks:
        return None
    # At the new viewpoint the nearest fork falls around the middle of the
    # calibrated window, leaving actual translation after the first turn.
    advance = min(forks) - FORECAST_STEPS // 2
    prefix = host[:advance]
    return (prefix, alternatives) if prefix.count(1) >= 2 else None


def same_pose(first, second, scale):
    return (math.dist(first[:2], second[:2]) * scale <= .25 and
            abs(math.remainder(first[2] - second[2], 2*math.pi)) <= math.radians(20))


def moved(pose, action, scale):
    x, y, yaw = pose
    if action == 1:
        x += .25 * math.cos(yaw) / scale
        y += .25 * math.sin(yaw) / scale
    else:
        yaw += math.pi / 6 if action == 2 else -math.pi / 6
    return x, y, yaw


class DeferredForks:
    deferred = None

    def _cancel_delay(self, reason):
        state = self.deferred
        if state is not None:
            self.deferred = None
            self.emit("deferred_end", reason=reason, start=state["start"], confirmed=state["cursor"])

    def _schedule_delay(self, observation, index, source, available):
        pose = self._frontier_pose or self._frontier_slam_pose
        follower = self._explore_follower
        if pose is None or follower is None:
            return
        clusters = self._last_frontier_clusters
        delayed = trunk_plan(clusters, index, pose, follower, available)
        if delayed is None:
            return
        prefix, keys = delayed
        cluster = clusters[index]
        revision = self._map_revision_tuple(self._frontier_server_revision)
        if revision is None:
            return
        self.deferred = dict(carrier=follower, start=self.now, source=source, prefix=prefix,
                             keys=keys, original=tuple(cluster["key"]), world=tuple(cluster["world"]),
                             pose=tuple(pose), expected=tuple(pose), cursor=0, sent=None,
                             revision=self._frontier_revision, metric=self._metric_snapshot.get("revision"),
                             loop_closures=revision[2])
        self.emit("deferred_start", original=f"f{index}", reachable_alternative_keys=len(keys),
                  common_steps_before_recheck=len(prefix), scan_steps_spent=0)

    def _feed_frame(self, o):
        previous = self.observed_step
        result = super()._feed_frame(o)
        state = self.deferred
        if state is None or self.now <= previous or self.now <= state["start"]:
            return result
        k = state["cursor"]
        if (state["sent"] != self.now - 1 or k >= len(state["prefix"]) or
                o.previous_action != state["prefix"][k]):
            self._cancel_delay("common_action_not_acknowledged")
        elif o.previous_action == 1 and self._last_motion_failed:
            self._cancel_delay("forward_blocked")
        else:
            state["expected"] = moved(state["expected"], o.previous_action, state["carrier"].scale)
            state["cursor"] += 1
            state["sent"] = None
        return result

    def _native_context(self, state):
        return (self._explore_follower is state["carrier"] and self.mode == "explore" and
                not self._adjusting and not self._scanning and
                self._active_frontier_key == state["original"] and
                self._metric_snapshot.get("revision") == state["metric"])

    def _explore_follow(self, observation):
        state = self.deferred
        if state is not None:
            if not self._native_context(state) or self._frontier_revision != state["revision"]:
                self._cancel_delay("native_path_or_map_changed")
            elif state["cursor"] == len(state["prefix"]):
                self._start_deferred_scan(observation, state)
        action = super()._explore_follow(observation)
        state = self.deferred
        if state is not None:
            k = state["cursor"]
            if (not self._native_context(state) or k >= len(state["prefix"]) or
                    action != state["prefix"][k]):
                self._cancel_delay("native_action_changed")
            else:
                state["sent"] = self.now
        return action

    def _execute_decider_result(self, observation, result, event, images=None):
        self._cancel_delay("native_decision_" + str(result.action))
        return super()._execute_decider_result(observation, result, event, images)

    def init_episode(self, observation):
        self._cancel_delay("episode_reset")
        return super().init_episode(observation)

    def act(self, observation):
        action = super().act(observation)
        state = self.deferred
        if state is not None and (not self._native_context(state) or state["sent"] != self.now or
                                  action != state["prefix"][state["cursor"]]):
            self._cancel_delay("higher_priority_action")
        return action

    def _fresh_deferred_table(self, observation, state, budget):
        """Rebuild with native A* at the observed pose. The current carrier stays live."""
        if budget < 1 or not self._native_context(state):
            return None
        follower = state["carrier"]
        if follower.anchor_frame < 0:
            return None
        # A read-only decision-map refresh must not postpone the next native
        # autonomous planning opportunity if the optional handoff is declined.
        last_plan = self._last_explore_plan
        try:
            refreshed = self._plan_exploration(observation, select=False)
        finally:
            self._last_explore_plan = last_plan
        if not refreshed:
            return None
        if (not self._native_context(state) or self._last_frontier_step != self.now or
                not math.isclose(follower.scale, self._frontier_scale, rel_tol=.01)):
            return None
        pose = self._estimated_current_pose()
        if pose is None or not same_pose(pose, state["expected"], follower.scale):
            return None
        state_revision = self._map_revision_tuple(self._frontier_server_revision)
        if state_revision is None or state_revision[2] != state["loop_closures"]:
            return None
        grid = self._frontier_grid
        window = min(budget, FORECAST_STEPS)
        def replan(goal):
            raw = grid.astar(pose[:2], goal)
            return grid.shortcut(raw) if raw is not None and len(raw) >= 2 else None
        origin_path = replan(state["world"])
        if origin_path is None or not follower.path:
            return None
        def prediction(path, scale):
            return follow(type(follower), path, pose, scale, follower.reach_m, window)
        original_plan = prediction(follower.path, follower.scale)
        certified = prediction(origin_path, self._frontier_scale)
        if original_plan != certified:
            return None
        rows = [dict(id="active", nominal_action_ids=original_plan)]
        lookup = {}
        for i, cluster in enumerate(self._last_frontier_clusters):
            key = cluster.get("key")
            if key is None or tuple(key) not in state["keys"]:
                continue
            path = replan(cluster["world"])
            if path is None:
                continue
            key_id = f"f{i}"
            rows.append(dict(id=key_id, nominal_action_ids=prediction(path, cluster["scale"])))
            lookup[key_id] = dict(cluster, path=path)
        entries, reason = build_table(rows, "active", budget)
        if reason:
            return None
        return entries, lookup, pose, state_revision

    def _checked_snapshot(self, observation, state, budget):
        try:
            return self._fresh_deferred_table(observation, state, budget)
        except (OSError, RuntimeError, ValueError, IndexError, ArithmeticError) as exc:
            self.emit("deferred_map_error", error_type=type(exc).__name__)
            return None

    def _start_deferred_scan(self, o, state):
        pose = self._estimated_current_pose()
        if (pose is None or math.dist(pose[:2], state["pose"][:2]) * state["carrier"].scale < .5 or
                self.pitch not in (0, 1) or self.word is None):
            self._cancel_delay("insufficient_confirmed_displacement_or_view")
            return
        fresh = self._checked_snapshot(o, state, int(o.max_steps) - self.now - len(self.scan_actions))
        if fresh is None:
            self._cancel_delay("no_fresh_reachable_fork")
            return
        entries, _, _, revision = fresh
        self._cancel_delay("fork_available_after_common_motion")
        self._begin_scan("active", [e["id"] for e in entries], [e["plan"] for e in entries], state["source"])
        self.scan["deferred_state"] = state
        self.scan["before_scan_revision"] = revision

    def _rank_after_scan(self, o, scan):
        state = scan.get("deferred_state")
        if state is None:
            return super()._rank_after_scan(o, scan)
        if (self.yaw != scan["yaw"] or len(scan["views"]) != 3 or
                not self._native_context(state) or self._explore_follower is not scan["carrier"]):
            self._end_scan("deferred_scan_interrupted")
            return
        self._end_scan("complete")
        remaining = int(o.max_steps) - self.now
        fresh = self._checked_snapshot(o, state, remaining)
        if fresh is None or fresh[3][2] != scan["before_scan_revision"][2]:
            self.emit("deferred_bypass", reason="post_scan_path_or_loop_closure_changed")
            return
        entries, lookup, pose, revision = fresh
        plans = [e["plan"] for e in entries]
        ids = [e["id"] for e in entries]
        if self.channel is None:
            self._connect()
        self.channel.send(dict(history=self.history, step=self.now, views=scan["views"],
                               plans=plans, remaining=remaining))
        if not self.channel.poll(240):
            raise TimeoutError("variable route scorer did not reply")
        scores = self.channel.recv()
        self.history = []
        k = scores["index"]
        if not isinstance(k, int) or not 0 <= k < len(ids):
            raise ValueError("invalid deferred route index")
        # A mapper can finish a keyframe or loop closure while the local worker
        # is scoring. Require the same atomic snapshot before activating a path.
        try:
            current_revision = self._map_revision_tuple(self.client.get_state())
        except Exception:
            current_revision = None
        if not self._native_context(state) or revision != current_revision:
            self.emit("deferred_bypass", reason="mapper_changed_during_scoring")
            return
        pick = ids[k]
        if k:
            from most_frontier_probe import ProbeAgent
            anchor = state["carrier"].anchor_frame
            # The original frontier was actually followed; preserve its cooldown.
            if not ProbeAgent._activate_frontier(self, o, lookup[pick], state["source"]):
                self.emit("deferred_bypass", reason="updated_frontier_rejected")
                return
            self._explore_follower.x, self._explore_follower.y, self._explore_follower.yaw = pose
            self._explore_follower.anchor_frame = anchor
        difference = next((i for i, (a, b) in enumerate(zip(plans[0], plans[k])) if a != b), None)
        self.emit("rank", original="active", selected=pick, ids=ids, plans_native=plans,
                  frame_steps=scan["frame_steps"], first_difference=difference, model_target=self.word,
                  room_prior_supported=True, attribute_unmodelled=True, delayed=True,
                  confirmed_common_steps=state["cursor"], **scores)
        self.accept_selection(o, "active", pick, plans[0], plans[k])
