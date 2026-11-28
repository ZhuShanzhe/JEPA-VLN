"""GPT chooses a reachable region or an image cue. JEPA ranks camera sites for that cue."""
import math
from collections import deque
from local_runtime import geometric_plan, request, project_patches, propose_sites
from map_views import visible_map_surfaces


def relative(pose, target):
    dx,dy=target[0]-pose[0],target[1]-pose[1]
    return round(math.hypot(dx,dy),2), round(math.degrees(math.remainder(math.atan2(dy,dx)-pose[2],math.tau)))


class GatedPolicy:
    def __init__(self,gate,instruction=None):
        self.gate=gate
        self.instruction=instruction
        self.recent=deque(maxlen=4)

    def pick(self,tracker,memory,grid,pose,scale,rows,views,goal,budget,channel,
             depth_m=None,hfov=90.,blocked=False,replay_memory=False,audit_sites=None):
        metric=(pose[0]*scale,pose[1]*scale,pose[2])
        prediction=request(channel,mode="observe",views=views,plans=[r["actions"] for r in rows],goal=goal)
        for a,p in zip((-math.pi/6,0.,math.pi/6), prediction["current"]):
            if replay_memory:
                memory.mark_view(metric[:2],pose[2]+a,observed_prior=p)
            else:
                memory.update_context(metric[:2],pose[2]+a,p)
        entered=tracker.inside(metric)
        def plan(target):
            return geometric_plan(grid,pose,scale,target,budget,tracker.chunk)
        patches=prediction["patches"]
        regions=[]
        previous_index=None
        if rows:
            assert len(rows)==len(prediction["score"])==len(prediction["evidence"])
            for row,p,e in zip(rows,prediction["score"],prediction["evidence"]):
                row.update(prior=p,evidence=e)
            previous_index,table=memory.choose(rows)
            for j,(row,meta) in enumerate(zip(rows,table["candidates"])):
                distance,bearing=relative(metric,row["position"])
                regions.append(dict(id=f"r{j}",distance_m=distance,bearing_deg=bearing,
                                    turn_deg=round(math.degrees(math.remainder(row["heading"]-pose[2], math.tau))),
                                    prior=round(meta["prior"],2),novelty=round(meta["novelty"],2),
                                    unseen=round(meta["unseen"],2),steps=meta["steps"],blocked=bool(meta["blocked"])))
        # Keep all nine observed patches available to the upper model. Visual labels are suggestions.
        offered=[{k:p[k] for k in ("patch_id","bbox_norm","surface","support")} for p in patches]
        history=[]
        for pos,mode,surface in self.recent:
            dist,az=relative(metric,pos)
            history.append(dict(distance_m=dist,bearing_deg=az,mode=mode,surface=surface))
        context=dict(remaining_steps=budget,region_entered=entered,previous_contact=bool(blocked),
                     checked_sites=sum(why=="arrived" for _,why,_ in tracker.history),
                     recent=history,active_local=bool(tracker.target))
        answer=self.gate.choose(views=views,goal=self.instruction or goal,regions=regions,patches=offered,memory=context) if regions else None
        region=None
        anchors=[]
        sites=[]
        status="no_region"
        release=None
        if answer:
            ids={r["id"]:i for i,r in enumerate(regions)}
            region=rows[ids[answer["region_id"]]]
            if answer["mode"]=="inspect":
                selected=[p for p in patches if p["patch_id"]==answer["patch_id"]]
                assert len(selected)==1
                anchors=project_patches(grid,pose,scale,selected,depth=depth_m,hfov=hfov)
                sites=propose_sites(grid,pose,scale,anchors,budget=min(budget,24))
                if sites:
                    scores=request(channel,mode="local_sites",plans=[s["actions"] for s in sites])
                    assert len(scores)==len(sites)
                    for s,sc in zip(sites,scores):
                        s["score"]=float(sc[s["patch_id"]])
                    if tracker.target and min(math.dist(s["anchor_xy"],tracker.target["anchor_xy"]) for s in sites)>.85:
                        release=dict(event="gpt_new_visual_anchor",goal_id=tracker.target["goal_id"],age=tracker.age)
                        tracker.target=None
                status="inspect_runnable" if sites else ("no_camera_site" if anchors else "cue_not_projectable")
            else:
                if tracker.target:
                    release=dict(event="gpt_resume_exploration",goal_id=tracker.target["goal_id"],age=tracker.age)
                    tracker.target=None
                status="explore"
        decision=tracker.decide(metric,region,sites,plan,
                                region_entered=bool(answer and answer["mode"]=="inspect"),blocked=blocked)
        if release and not decision["released"]:decision["released"]=release
        if status=="inspect_runnable" and decision["action_source"]!="local":status="site_not_adopted"
        if region:
            pos=decision["site_xy"] if decision["action_source"]=="local" else region["position"]
            self.recent.append((tuple(pos),answer["mode"],decision.get("surface","")))
        tracker.scan_surfaces=visible_map_surfaces(grid,pose,scale)
        tracker.may_skip_scan=decision["action_source"]=="local"
        decision.update(region_phase="gpt_upper",region_entered=entered,patches=len(patches),anchors=len(anchors),
                        sites=len(sites),anchor_source="rgbd_measured" if depth_m is not None else "observed_map_column_proxy",
                        region_key=region["key"] if region else None,gpt_called=bool(answer),
                        gpt_region_agrees_with_prior=(answer["region_id"]==f"r{previous_index}") if answer else None,
                        gpt_region=answer["region_id"] if answer else None,gpt_mode=answer["mode"] if answer else None,
                        gpt_patch=answer.get("patch_id") if answer else None,
                        gpt_cue=answer.get("cue") if answer else None,
                        gpt_evidence=answer.get("target_evidence") if answer else None,exec_status=status)
        if audit_sites:audit_sites([r["position"] for r in rows],[s["site_xy"] for s in sites],region)
        return decision,region
