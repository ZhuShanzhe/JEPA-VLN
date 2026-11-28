"""GPT-4o assigns exploration intents, JEPA selects a reachable view for a concrete cue."""
import math
from pathlib import Path

from local_agent import LocalApproachAgent
from most_frontier_probe import candidate_rows
from global_frontiers import frontier_plans
from map_views import candidates
from arm_policy import GatedPolicy
from gpt_gate import GPTGate


class GPTRegionAgent(LocalApproachAgent):
    def init_episode(self, obs):
        super().init_episode(obs)
        self._gate = None
        self._gated = None
        self._instruction = getattr(obs, "goal_text", "") or self.word

    def _current_frame(self):
        old = self.search
        frame = super()._current_frame()
        if self.search is not old:
            # Map-position hints are meaningful only in the matching map revision.
            self._gated = None
        return frame

    def _rank_after_scan(self, obs, scan):
        self._end_scan("complete")
        frame = self._current_frame()
        if frame is None or self.yaw != scan["yaw"] or len(scan["views"]) != 3:
            return
        pose, scale = frame
        follower = self._explore_follower
        if follower is None or self._frontier_grid is None or not math.isclose(scale, follower.scale, rel_tol=.03):
            return
        clusters = self._last_frontier_clusters
        native = next((f"f{i}" for i,c in enumerate(clusters) if c is scan["original_obj"]), None)
        table = candidate_rows(clusters,pose,scale,self.reach_m)
        plans = [(r["id"],r["nominal_action_ids"]) for r in sorted(table,key=lambda r:r["id"]!=native)
                 if r["nominal_action_ids"]]
        grid = self.motion_memory.grid(self._frontier_grid,scale,self.now)
        remaining = obs.max_steps-self.now
        plans.extend(frontier_plans(grid,pose,scale,remaining))
        rows = candidates(grid,pose,scale,plans,remaining)
        if not rows and self.local.target is None:
            self.emit("local_skip", reason="no_known_free_regional_approach")
            return
        if self.channel is None:
            self._connect()
        if self._gate is None:
            self._gate = GPTGate(self.episode, Path(self.output_dir)/"gpt_region_private")
        if self._gated is None:
            self._gated = GatedPolicy(self._gate, instruction=self._instruction)
        decision, region = self._gated.pick(self.local,self.search,grid,pose,scale,rows,scan["views"],
                                           self.word,remaining,self.channel,blocked=self._local_blocked)
        self._local_blocked = False
        self._bind_choice(obs,scan,native,decision,region)
