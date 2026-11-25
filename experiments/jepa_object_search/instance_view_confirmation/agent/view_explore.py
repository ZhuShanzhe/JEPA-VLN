"""Rank feasible local observation views using the agent's causal map and frozen JEPA."""
import math

import numpy as np


def cone(grid, pose, scale):
    """First uninspected cells in a 120-degree camera cone, before any occluder."""
    x, y, yaw = pose
    inspected = grid.semantic_inspected if grid.semantic_coverage_enabled else grid.geometry_observed
    spacing = max(.001, min(grid.res * .5, .25 / scale))
    unseen = []
    for angle in np.linspace(yaw-math.pi/3, yaw+math.pi/3, 13):
        heading = np.array([math.cos(angle), math.sin(angle)])
        first = None
        for d in np.arange(spacing, 2.0/scale + spacing*.01, spacing):
            c = grid.world_to_cell((x,y) + heading*d)
            if not grid.in_bounds(c) or grid.obstacle[c[1],c[0]]:
                break
            if not inspected[c[1],c[0]]:
                first = c
                break
            if not grid.geometry_observed[c[1],c[0]]:
                break
        unseen.append(first)
    return unseen


def trace(grid, pose, scale, plan, seen):
    x, y, yaw = pose
    safe = 0
    headings, moving = [], []
    samples = []
    reason = "route_endpoint"
    for i, a in enumerate(plan, 1):
        if a == 1:
            dx,dy = math.cos(yaw)*.25/scale, math.sin(yaw)*.25/scale
            steps = max(1, math.ceil(math.hypot(dx,dy)/(.5*grid.res)))
            if any(not grid.traversable(grid.world_to_cell((x+dx*t/steps,y+dy*t/steps)))
                   for t in range(1,steps+1)):
                reason = "movement_leaves_traversable_map"
                break
            x, y = x+dx, y+dy
            headings.append(round((yaw-pose[2])/(math.pi/6)) % 12)
            moving.append(i)
        else:
            yaw += math.pi/6 if a==2 else -math.pi/6
        safe = i
        if moving and (i%6 == 0 or i==len(plan)):
            samples.append((i,(x,y,yaw)))
    if moving and (not samples or samples[-1][0]!=safe):
        samples.append((safe,(x,y,yaw)))
    gains = []
    for t, camera in samples:
        rays = cone(grid,camera,scale)
        gains.append((t,sum(c is not None and c not in seen for c in rays)/len(rays)))
    return dict(safe=safe, reason=reason, headings=headings, moving=moving, samples=gains)


def view_profiles(grid, pose, scale, plans):
    if grid is None or pose is None or scale is None or not np.all(np.isfinite(pose)) or not (scale>0 and np.isfinite(scale)):
        return []
    initial = set(cone(grid,pose,scale))
    states = [trace(grid,pose,scale,p,initial) for p in plans]
    # Alternative viewpoints must lie beyond an actual change of travel direction.
    # Compare against the native nominal direction even when its own prefix is blocked.
    native_directions = []
    turn = 0
    for a in plans[0]:
        if a==1:
            native_directions.append(turn)
        else:
            turn=(turn+(1 if a==2 else -1))%12
    profiles=[]
    for k,state in enumerate(states):
        minimum=1
        eligible=bool(state["moving"])
        if k and eligible:
            divergent=next((j for j,(a,b) in enumerate(zip(native_directions,state["headings"])) if a!=b),None)
            eligible = divergent is not None
            if eligible:
                minimum = state["moving"][divergent]
        candidates=[(gain,t) for t,gain in state["samples"] if t>=minimum]
        gain,t=max(candidates, default=(0.,0))
        profiles.append(dict(allowed=bool(eligible and t>0),view_steps=t,free_steps=state["safe"],
                             new_view_fraction=gain,reach_check=state["reason"]))
    return profiles


class ViewChannel:
    def __init__(self, stream, host):
        self.stream, self.host = stream,host

    def send(self, msg):
        plans=msg["plans"]
        rows=self.host._view_profiles(plans)
        self.rows=rows
        self.score_step=msg["step"]
        prefixes=[p[:r["view_steps"]] if r["allowed"] else p for p,r in zip(plans,rows)] if rows else plans
        self.stream.send(dict(msg,plans=prefixes))

    def recv(self):
        reply=self.stream.recv()
        result=adjust(reply,self.rows)
        self.last_choice=(self.score_step,result["index"],self.rows)
        return result

    def __getattr__(self,name):
        return getattr(self.stream,name)


def adjust(reply, rows):
    native_scores=np.asarray(reply["final_scores"],dtype=float)
    n=len(native_scores)
    allow=np.array([r["allowed"] for r in rows],dtype=bool) if rows else np.zeros(n,dtype=bool)
    gain=np.array([r["new_view_fraction"] for r in rows]) if rows else np.zeros(n)
    if len(allow)!=n or not np.isfinite(native_scores).all():
        raise ValueError("mismatched map and JEPA candidate table")
    reason="view_ranked"
    if not allow[1:].any():
        allow[:]=False
        allow[0]=True
        reason="native_only_after_map_check"
    deficit=reply["clue_deficit"]
    # Reuse the existing deficit-weighted exploration term; no new fit on evaluation tasks.
    novelty=np.asarray(reply["exploration_bonus"])*(1-deficit+deficit*gain)
    score=native_scores-np.asarray(reply["exploration_bonus"])+novelty
    no_evidence=np.asarray(reply["memory_scores"])+np.asarray(reply["unvisited_fraction"])*reply["adaptive_weight"]*.35*(1-deficit+deficit*gain)
    mask=lambda s:np.where(allow,s,float(s.min())-1.)
    final=mask(score)
    return dict(reply, index=int(final.argmax()),final_scores=final.tolist(),exploration_bonus=novelty.tolist(),
                no_exploration_index=int(mask(score-novelty).argmax()),no_evidence_index=int(mask(no_evidence).argmax()),
                view_profiles=rows,view_reason=reason,pre_view_index=int(reply["index"]),
                pre_view_scores=native_scores.tolist())


class ReachableViewMixin:
    _view_watch=None

    def _end_view_watch(self, reason):
        watch=self._view_watch
        if watch is not None:
            self.emit("view_track_end",start=watch["start"],confirmed_commands=watch["cursor"],
                      planned_commands=len(watch["prefix"]),reason=reason)
            self._view_watch=None

    def accept_selection(self, observation, original_id, selected_id, original_plan, selected_plan):
        result=super().accept_selection(observation,original_id,selected_id,original_plan,selected_plan)
        self._end_view_watch("superseded")
        choice=getattr(self.channel,"last_choice",None)
        if choice and choice[0]==self.now:
            _,index,rows=choice
            if rows and rows[index]["allowed"]:
                prefix=list(selected_plan[:rows[index]["view_steps"]])
                self._view_watch=dict(start=self.now,cursor=0,prefix=prefix)
                self.emit("view_track_start",selected=selected_id,original=original_id,prefix=prefix)
        return result

    def _feed_frame(self,o):
        old=self.observed_step
        result=super()._feed_frame(o)
        watch=self._view_watch
        if watch and old<self.now and self.now>watch["start"]:
            if watch["prefix"][watch["cursor"]]!=o.previous_action:
                self._end_view_watch("different_executed_command")
            elif o.previous_action==1 and self._last_motion_failed:
                self._end_view_watch("native_forward_stall")
            else:
                watch["cursor"]+=1
                if watch["cursor"]==len(watch["prefix"]):
                    self._end_view_watch("observed_command_prefix_complete")
        return result

    def _execute_decider_result(self,o,result,event,images=None):
        self._end_view_watch("native_decision_"+str(result.action))
        return super()._execute_decider_result(o,result,event,images)

    def _activate_frontier(self,observation,cluster,source):
        self._end_view_watch("native_reactivation")
        return super()._activate_frontier(observation,cluster,source)

    def init_episode(self,o):
        self._end_view_watch("episode_reset")
        return super().init_episode(o)

    def _view_profiles(self,plans):
        pose=self._frontier_pose or self._frontier_slam_pose
        return view_profiles(self._frontier_grid,pose,self._frontier_scale,plans)

    def _begin_scan(self, original, ids, plans, source):
        rows=self._view_profiles(plans)
        viable=any(r["allowed"] for r in rows[1:])
        self.emit("view_filter", ids=ids,view_profiles=rows,viable_alternative=viable,source=str(source))
        # DeferredForks attaches an additional snapshot to the scan, then repeats
        # its own A* and loop closure checks before the channel is called.
        if not viable and original!="active":
            return
        return super()._begin_scan(original,ids,plans,source)

    def _connect(self):
        super()._connect()
        self.channel=ViewChannel(self.channel,self)
