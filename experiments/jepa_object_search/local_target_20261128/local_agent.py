"""MOST-Nav exploration hook for persistent, visually grounded observation poses."""
import math
import os
from pathlib import Path

from global_frontiers import frontier_plans
from map_views import candidates
from most_frontier_probe import candidate_rows
from search_agent import PriorApproachAgent

from local_runtime import LocalTargetTracker, fast_continue, pick, start_worker
from observation_sites import plan_to_site


class LocalApproachAgent(PriorApproachAgent):
    def init_episode(self, obs):
        super().init_episode(obs)
        self.local = LocalTargetTracker()
        self._local_blocked = False
        self._local_forward_at = None

    def _connect(self):
        project = Path(os.environ['NAV_ROOT'])
        self.channel, self.worker = start_worker(project, Path(self.output_dir)/'local_worker.log', cpu=True)

    def _release_owned_carrier(self, reason):
        active = self._focus_view
        if active is not None:
            follower, tag = active[:2]
            if tag.startswith(('local/', 'region/')):
                if self._explore_follower is follower:
                    self._explore_follower = None
                self._finish_focus(reason)

    def _current_frame(self):
        prior = self.search
        frame = super()._current_frame()
        # The parent creates a new prior memory after scale changes or loop closures.
        # Positions saved in a superseded map cannot be used in the revised frame.
        if prior is not self.search:
            self._release_owned_carrier('revised_map_frame')
            self.local = LocalTargetTracker()
            self._local_blocked = False
            self._local_forward_at = None
            self.emit('local_map_reset', reason='scale_or_loop_closure')
        return frame

    def _feed_frame(self, obs):
        old = self.observed_step
        out = super()._feed_frame(obs)
        if old >= self.observed_step or self.local.target is None:
            return out
        if (self._local_forward_at == self.now-1 and obs.previous_action == 1 and
                getattr(self,'_last_motion_failed',False)):
            self._local_blocked = True
        frame = self._current_frame()
        grid = self._frontier_grid
        if frame is not None and grid is not None and self.local.target is not None:
            pose, scale = frame
            online = self.motion_memory.grid(grid, scale, self.now)
            reached = self.local.arrival((pose[0]*scale, pose[1]*scale, pose[2]),
                                         lambda t: plan_to_site(online,pose,scale,t,
                                                                budget=obs.max_steps-self.now))
            if reached:
                self._release_owned_carrier('local_observation_reached')
                self._local_blocked = False
                self.emit('local_site_observation', observation_source='estimated_map_pose', **{
                    k:v for k,v in reached.items() if k != 'event'})
        return out

    def _explore_follow(self, obs):
        act = super()._explore_follow(obs)
        active = self._focus_view
        if act == 1 and active is not None and active[1].startswith('local/'):
            self._local_forward_at = self.now
        return act

    def _execute_decider_result(self, observation, result, event, images=None):
        if result.action in ('START_ADJUST','GOTO_INSTANCE','REPORT_FOUND','FINISH'):
            self._release_owned_carrier('native_priority')
            if self.local.target or self.local.region:
                self.emit('local_priority_interrupt', native_action=str(result.action))
                self.local.target = self.local.region = None
            self._local_forward_at = None
            self._local_blocked = False
        return super()._execute_decider_result(observation,result,event,images)

    def _activate_frontier(self, obs, cluster, source):
        applied = super()._activate_frontier(obs,cluster,source)
        # The inherited eligibility check installs a scan only on a native exploration
        # command with enough budget and an active carrier. No other mode enters here.
        scan = self.scan
        if not applied or scan is None or self.local.target is None:
            return applied
        frame = self._current_frame()
        if frame is None or self.local.target is None or self.scan is not scan:
            return applied
        pose, scale = frame
        grid = self.motion_memory.grid(self._frontier_grid,scale,self.now)
        decision = fast_continue(self.local,grid,pose,scale,obs.max_steps-self.now,
                                 blocked=self._local_blocked)
        if decision is not None and decision['actions']:
            self._end_scan('retained_goal_geometry_check')
            self._bind_choice(obs,scan,None,decision,None)
        return applied

    def _rank_after_scan(self, obs, scan):
        self._end_scan('complete')
        frame = self._current_frame()
        if frame is None or self.yaw != scan['yaw'] or len(scan['views']) != 3:
            return
        pose, scale = frame
        fl, grid = self._explore_follower, self._frontier_grid
        if fl is None or grid is None or not math.isclose(scale, fl.scale, rel_tol=.03):
            return
        clusters = self._last_frontier_clusters
        native = next((f'f{i}' for i,c in enumerate(clusters) if c is scan['original_obj']),None)
        table = candidate_rows(clusters,pose,scale,self.reach_m)
        plans = [(r['id'],r['nominal_action_ids']) for r in sorted(table,key=lambda r:r['id']!=native)
                 if r['nominal_action_ids']]
        grid = self.motion_memory.grid(grid,scale,self.now)
        remaining = obs.max_steps-self.now
        plans.extend(frontier_plans(grid,pose,scale,remaining))
        rows = candidates(grid,pose,scale,plans,remaining)
        if not rows and self.local.target is None and not self.local.inside((pose[0]*scale,pose[1]*scale,pose[2])):
            self.emit('local_skip', reason='no_known_free_regional_approach')
            return
        if self.channel is None:
            self._connect()
        # The native scan RGB is yaw/pitch-rectified. MOST's public observation has
        # no aligned metric depth; pass its current observed 2-D map via the proxy.
        decision, region = pick(self.local,self.search,grid,pose,scale,rows,scan['views'],
                                self.word,remaining,self.channel,blocked=self._local_blocked)
        self._local_blocked = False
        self._bind_choice(obs,scan,native,decision,region)

    def _bind_choice(self,obs,scan,native,decision,region):
        selected = region['key'] if region else None
        local = decision['action_source'] == 'local'
        tag = f"local/{decision['goal_id']}/{decision['age']}" if local else f"region/{selected}"
        bound = bool(decision['actions']) and self.start_focus_view(obs,decision['actions'],tag)
        if bound:
            # A local displacement is accounted as a measured contact in MotionMemory,
            # without penalizing an entire anchor or unrelated regional candidate.
            if not local:
                self._selected_site = region['position'], region['heading']
            if local or native is None or selected.split('/')[0] != native:
                self._active_frontier_key = self._active_branch_key = None
                if self._recent_frontiers and self._recent_frontiers[-1][1] == scan['start']:
                    pt, _ = self._recent_frontiers[-1]
                    if math.dist(pt[:2],scan['original_obj']['world'][:2]) < 1.e-6:
                        self._recent_frontiers.pop()
        fields = dict(decision)
        fields['transition'] = fields.pop('event')
        self.emit('local_route_choice', bound=bound, **fields)
