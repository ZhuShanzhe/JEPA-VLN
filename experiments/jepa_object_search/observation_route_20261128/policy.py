"""One live entry for pre-detection exploration and post-detection verification."""
import math
from pathlib import Path

from gateway import ObservationGateway
from host_dispatch import ObservationActions
from local_runtime import propose_sites, request
from multi_policy import MultiViewPolicy, located_cue
from routes import observation_routes, score_observation_routes
from terminal_prefix import with_terminal_prefixes
from trigger_policy import relative


class ObservationRoutePolicy(MultiViewPolicy):
    def __init__(self, gate, instruction=None, cue=None):
        super().__init__(gate, instruction=instruction, cue=cue)
        self.gateway = ObservationGateway()
        self.map_memory = None
        self.prefix_serial = 0

    def _regions(self, memory, rows, pose, prediction):
        if not rows:
            return None, None, []
        if len(rows)!=len(prediction["score"]) or len(rows)!=len(prediction["evidence"]):
            raise ValueError("bad regional evidence shape")
        for row,p,e in zip(rows, prediction["score"], prediction["evidence"]):
            row.update(prior=p,evidence=e)
        index, table = memory.choose(rows)
        regions = []
        for i,(row,meta) in enumerate(zip(rows, table["candidates"])):
            distance, bearing = relative(pose, row["position"])
            turn = round(math.degrees(math.remainder(row["heading"]-pose[2],math.tau)))
            regions.append(dict(id=f"r{i}",distance_m=distance,bearing_deg=bearing,turn_deg=turn,
                                prior=round(meta["prior"],2),novelty=round(meta["novelty"],2),
                                unseen=round(meta["unseen"],2),steps=meta["steps"],blocked=bool(meta["blocked"])))
        return rows[index],f"r{index}",regions

    def pick(self, tracker, memory, grid, pose, scale, rows, views, goal, budget,
             channel, depth_m=None, hfov=90., blocked=False, replay_memory=False,
             audit_sites=None, scan_geometry=None):
        # Existing long observations and their interruption/expiry retain native ownership.
        if tracker.target is not None:
            return super().pick(tracker,memory,grid,pose,scale,rows,views,goal,budget,channel,
                                depth_m,hfov,blocked,replay_memory,audit_sites,scan_geometry)
        if depth_m is not None and scan_geometry is None:
            raise ValueError("single central depth cannot locate a multiview detection")
        if self.map_memory is not memory:
            self.map_memory = memory
            self.gateway = ObservationGateway()
            self.recent.clear()
        metric = (pose[0]*scale,pose[1]*scale,pose[2])
        prediction = request(channel,mode="observe",views=views,goal=goal,plans=[r["actions"] for r in rows])
        for offset,p in zip((-math.pi/6,0,math.pi/6), prediction["current"] if rows else []):
            if replay_memory:
                memory.mark_view(metric[:2],pose[2]+offset,observed_prior=p)
            else:
                memory.update_context(metric[:2],pose[2]+offset,p)
        region,rid,regions = self._regions(memory,rows,metric,prediction)
        patches = [{k:p[k] for k in ("patch_id","bbox_norm","surface","support")} for p in prediction["patches"]]
        entered = tracker.inside(metric)
        history = []
        for xy, mode, surface in self.recent:
            distance, bearing = relative(metric,xy)
            history.append(dict(distance_m=distance,bearing_deg=bearing,mode=mode,surface=surface))
        context = dict(remaining_steps=budget,region_entered=entered,previous_contact=bool(blocked),
                       checked_sites=sum(why=="arrived" for _,why,_ in tracker.history),
                       recent=history,active_local=False)
        answer = self.gate.choose(views=views,goal=self.instruction or goal,regions=regions,
                                  patches=patches,memory=context) if regions else None
        upper = answer["mode"] if answer else None
        choice, anchors, anchor_source, cue_event = None, [], None, None
        if upper=="inspect" and not blocked and budget>0:
            choice = self.cue.select(views,goal)
            cue_event = self.cue.last_receipt["event"]
            if choice is not None:
                anchors,anchor_source,_ = located_cue(choice,views,grid,pose,scale,scan_geometry,hfov)
        mode, reason = self.gateway.admit(upper_mode=upper,accepted_cue=choice is not None,
                                         anchors=anchors,blocked=blocked,budget=budget,region=region)
        proposal = None
        if mode is not None:
            proposal = observation_routes(grid,pose,scale,mode=mode,region_actions=region["actions"],
                                          observed_surfaces=memory.inspected,budget=budget,
                                          anchor_xy=anchors[0]["anchor_xy"] if anchors else None)
            if mode=="verify":
                certified = propose_sites(grid,pose,scale,anchors,budget=min(budget,24))
                proposal = with_terminal_prefixes(proposal,certified,grid,pose,scale,anchors[0]["anchor_xy"],budget)
            score = len(proposal["candidates"]) >= 2
            if mode=="discover":
                score,reason = self.gateway.discoverable(proposal,pose,scale,budget)
            if score:
                def model_scores(plans):
                    if mode=="discover":
                        return request(channel,mode="observe",views=views,goal=goal,plans=plans)["evidence"]
                    return request(channel,mode="selected_box",view_index=choice["view_index"],
                                   bbox_norm=choice["bbox_norm"],plans=plans)
                proposal = score_observation_routes(proposal,model_scores)
        tracker.tick += 1
        decision = tracker._region_move(region,"regional_approach",None)
        selected = proposal["selected_id"] if proposal else "continue"
        if selected != "continue":
            row = next(r for r in proposal["candidates"] if r["id"]==selected)
            self.prefix_serial += 1
            decision.update(actions=ObservationActions(row["actions"]),event="observation_prefix",
                            action_source="local",goal_id=f"obs{self.prefix_serial}",age=1,
                            commit_actions_left=len(row["actions"]),commit_horizon=len(row["actions"]),
                            site_xy=row["site_xy"],site_yaw=row["face_yaw"])
        if region:
            self.recent.append((tuple(decision.get("site_xy",region["position"])),upper,
                                choice["category"] if choice else ""))
        key = region["key"] if region else None
        decision.update(old_selected_region_id=rid,executed_region_id=rid,old_selected_region_key=key,
                        region_key=key,region_phase="observation_routes",region_entered=entered,
                        patches=len(patches),anchors=len(anchors),sites=len(proposal["candidates"])-1
                        if proposal and proposal["candidates"] else 0,anchor_source=anchor_source,
                        gpt_called=answer is not None,gpt_mode=upper,gpt_region=answer.get("region_id") if answer else None,
                        cue_selected=choice is not None,cue_event=cue_event,
                        cue_view_index=choice["view_index"] if choice else None,
                        observation_mode=mode,gateway_reason=reason,observation_routes=proposal,
                        route_selected=selected,exec_status=decision["event"])
        if audit_sites:
            offered = [r["site_xy"] for r in proposal["candidates"] if r["id"]!="continue"] if proposal else []
            audit_sites([r["position"] for r in rows],offered,region)
        return decision,region
