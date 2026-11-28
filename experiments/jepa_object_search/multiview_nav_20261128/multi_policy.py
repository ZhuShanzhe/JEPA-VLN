"""Approve a target cue on one of three actual observations and reach its viewing site."""
import math
from pathlib import Path
import numpy as np

from commit_policy import CommittedInspectPolicy
from local_runtime import geometric_plan, project_patches, propose_sites, request
from map_views import visible_map_surfaces
from trigger_policy import relative


def located_cue(choice, views, grid, pose, scale, geometry, hfov):
    idx, box = choice["view_index"], choice["bbox_norm"]
    if type(idx) is not int or idx not in range(3) or len(box)!=4:
        raise ValueError("invalid selected camera or box")
    l,t,r,b = map(float,box)
    if not (0<=l<r<=1 and 0<=t<b<=1):
        raise ValueError("region outside source frame")
    if geometry is None:
        # Native scan is physically three rectified yaws with fixed camera position.
        # Its public observation has no aligned metric depth. Only a measured-map
        # obstacle ray along the selected source bearing can support this fallback.
        camera = (*pose[:2], pose[2]+(idx-1)*math.pi/6)
        depth = None
        source = "source_view_observed_map_column_proxy"
    else:
        if len(geometry)!=3:
            raise ValueError("three camera poses and aligned optical-axis depths required")
        camera, raw = geometry[idx]
        depth = np.asarray(raw, dtype=np.float32).squeeze()
        if depth.shape != np.asarray(views[idx]).shape[:2] or depth.ndim!=2:
            raise ValueError("source RGB and metric depth have different sizes")
        if not all(math.isfinite(float(v)) for v in camera) or len(camera)!=3:
            raise ValueError("invalid source camera pose")
        if abs(math.remainder(camera[2]-pose[2]-(idx-1)*math.pi/6, math.tau))>math.radians(2.):
            raise ValueError("selected camera pose inconsistent with captured turn sequence")
        # Keep the frozen central-window median in the selected box only; an adjacent
        # background wall must not supply depth to a tiny foreground detection.
        h,w = depth.shape
        yy,xx = np.ogrid[:h,:w]
        inside = ((xx+.5)>=l*w)&((xx+.5)<r*w)&((yy+.5)>=t*h)&((yy+.5)<b*h)
        depth = np.where(inside, depth, np.nan)
        source = "selected_view_aligned_rgbd"
    patch = dict(patch_id=idx, bbox_norm=list(box), view_index=idx, surface=choice["category"],
                 support=1., candidate_id=choice["candidate_id"])
    anchors = project_patches(grid, camera, scale, [patch], depth=depth, hfov=hfov)
    return anchors, source, float(camera[2])


class MultiViewPolicy(CommittedInspectPolicy):
    def __init__(self, gate, instruction=None, cue=None):
        super().__init__(gate, instruction)
        if cue is None:
            from cue_service import CueSelector
            cue = CueSelector(log_dir=Path(gate.path).parent/("cue_"+Path(gate.path).stem))
        self.cue = cue

    def pick(self, tracker, memory, grid, pose, scale, rows, views, goal, budget, channel,
             depth_m=None, hfov=90., blocked=False, replay_memory=False, audit_sites=None, scan_geometry=None):
        if depth_m is not None and scan_geometry is None:
            raise ValueError("single central depth cannot locate a multi-view detection")
        metric = (pose[0]*scale,pose[1]*scale,pose[2])
        prediction = request(channel, mode="observe", views=views, plans=[r["actions"] for r in rows], goal=goal)
        for a,p in zip((-math.pi/6,0.,math.pi/6), prediction["current"] if rows else ()):
            if replay_memory:
                memory.mark_view(metric[:2],pose[2]+a,observed_prior=p)
            else:
                memory.update_context(metric[:2],pose[2]+a,p)
        entered = tracker.inside(metric)
        def plan(site):
            return geometric_plan(grid,pose,scale,site,budget,tracker.chunk)
        regions = []
        previous_index = None
        region = None
        if rows:
            assert len(rows)==len(prediction["score"])==len(prediction["evidence"])
            for row,p,e in zip(rows,prediction["score"],prediction["evidence"]):
                row.update(prior=p,evidence=e)
            previous_index,table=memory.choose(rows)
            region = rows[previous_index]
            for j,(row,meta) in enumerate(zip(rows,table["candidates"])):
                distance,bearing=relative(metric,row["position"])
                regions.append(dict(id=f"r{j}",distance_m=distance,bearing_deg=bearing,
                                    turn_deg=round(math.degrees(math.remainder(row["heading"]-pose[2], math.tau))),
                                    prior=round(meta["prior"],2),novelty=round(meta["novelty"],2),
                                    unseen=round(meta["unseen"],2),steps=meta["steps"],blocked=bool(meta["blocked"])))
        patches=prediction["patches"]
        offered=[{k:p[k] for k in ("patch_id","bbox_norm","surface","support")} for p in patches]
        history=[]
        for pos,mode,surface in self.recent:
            dist,az=relative(metric,pos)
            history.append(dict(distance_m=dist,bearing_deg=az,mode=mode,surface=surface))
        active = tracker.target
        context=dict(remaining_steps=budget,region_entered=entered,previous_contact=bool(blocked),
                     checked_sites=sum(why=="arrived" for _,why,_ in tracker.history),
                     recent=history,active_local=bool(active))
        answer=self.gate.choose(views=views,goal=self.instruction or goal,regions=regions,patches=offered,
                                memory=context) if regions else None
        inspect = bool(answer and answer["mode"]=="inspect")
        checked = inspect and active is None
        choice = None
        anchors = []
        sites = []
        source = yaw = receipt = None
        status = "no_region" if answer is None else "explore"
        if checked:
            choice = self.cue.select(views,goal)
            receipt = self.cue.last_receipt["event"]
            if choice is None:
                status = "no_candidates" if receipt=="no_candidates" else "gate_abstain"
            else:
                anchors, source, yaw = located_cue(choice,views,grid,pose,scale,scan_geometry,hfov)
                sites = propose_sites(grid,pose,scale,anchors,budget=min(budget,24))
                if sites:
                    scores=request(channel,mode="selected_box",view_index=choice["view_index"],
                                   bbox_norm=choice["bbox_norm"],plans=[s["actions"] for s in sites])
                    if len(scores)!=len(sites) or not all(math.isfinite(float(x)) for x in scores):
                        raise ValueError("invalid selected-box JEPA site scores")
                    for s,x in zip(sites,scores):
                        s["score"]=float(x)
                status = "inspect_runnable" if sites else ("no_camera_site" if anchors else "cue_not_projectable")
        left = self._remaining(tracker,budget) if active else None
        if active and left==0:
            tracker.tick += 1
            rel = tracker.arrival(metric,plan)
            if not rel:
                tracker._remember(active,"inspection_budget")
                rel = dict(event="inspection_budget",goal_id=active["goal_id"],age=tracker.age,
                           elapsed=self.start_remaining-budget,distance_m=math.dist(metric[:2],active["site_xy"]))
                tracker.target = None
            decision = tracker._region_move(region,"region_after_inspection",rel)
            status = rel["event"]
        else:
            decision = tracker.decide(metric,region,sites,plan,region_entered=inspect,blocked=blocked)
            if active and decision["event"]=="retained":
                status = "committed_view_retained"
            elif status=="inspect_runnable" and decision["action_source"]!="local":
                status = "site_not_adopted"
        if region:
            pos=decision["site_xy"] if decision["action_source"]=="local" else region["position"]
            self.recent.append((tuple(pos),answer["mode"],decision.get("surface","")))
        tracker.scan_surfaces=visible_map_surfaces(grid,pose,scale)
        tracker.may_skip_scan=decision["action_source"]=="local"
        rid=f"r{previous_index}" if region else None
        decision.update(old_selected_region_id=rid,old_selected_region_key=region["key"] if region else None,
                        executed_region_id=rid,region_phase="frozen_region_multiview_inspection",region_entered=entered,
                        patches=len(patches),anchors=len(anchors),sites=len(sites),anchor_source=source,
                        region_key=region["key"] if region else None,gpt_called=bool(answer),
                        gpt_region_agrees_with_prior=(answer["region_id"]==rid) if answer else None,
                        gpt_region=answer["region_id"] if answer else None,gpt_mode=answer["mode"] if answer else None,
                        gpt_patch=answer.get("patch_id") if answer else None,gpt_cue=answer.get("cue") if answer else None,
                        gpt_evidence=answer.get("target_evidence") if answer else None,exec_status=status,
                        commit_previous_id=active["goal_id"] if active else None,commit_before_dispatch=left,
                        cue_checked=checked,cue_selected=choice is not None,cue_event=receipt,
                        cue_view_index=choice["view_index"] if choice else None,
                        cue_bbox_norm=choice["bbox_norm"] if choice else None,
                        cue_candidate_id=choice["candidate_id"] if choice else None,
                        cue_projection_yaw=yaw,cue_anchor_source=source,cue_rgbd_aligned=(source=="selected_view_aligned_rgbd"),
                        cue_anchor_xy=anchors[0]["anchor_xy"] if anchors else None,
                        cue_score_source="selected_box_and_ordered_scan_jepa" if sites else None)
        self._annotate(decision,tracker,budget)
        if audit_sites:
            audit_sites([r["position"] for r in rows],[s["site_xy"] for s in sites],region)
        return decision,region
