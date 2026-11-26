"""Evaluate one bounded retry from the observed blocked edge in the archived small-clock case."""
from historical_trial import (P,ProposalWorker,ActiveViewScorer,audit,candidates,registered_image,
    semantic_scene,frozen,replay_until,quaternion_rotate_vector,math,np,relative_action,physical,pose_after,FORWARD,json)
from learned_map_views import blocked_edge


def avoids(commands,origin,edge):
    pose=origin
    for action in commands:
        if action==FORWARD and blocked_edge(pose,[edge],1.):
            return False
        pose=pose_after(pose,action,1.)
    return True


def main():
    doc,times,start,pose,ob,ref_step=frozen()
    env,item,data=semantic_scene(doc)
    det=ProposalWorker(P/"b_verification/env/bin/python",P,"cuda")
    try:
        source=replay_until(env,item,data,times,ref_step)
        image,ref=registered_image(source,ob)
        iy,ix=(np.array(ob["pixel"])[::-1]/1000*np.asarray(source["semantic"]).squeeze().shape).round().astype(int)
        target_id=int(np.asarray(source["semantic"]).squeeze()[iy,ix])
        frame=replay_until(env,item,data,times,start["step"])
        heading=quaternion_rotate_vector(env.sim.get_agent_state().rotation,np.array([0.,0.,-1.]))
        anchor,options,_,_=candidates(times,start,pose,math.atan2(heading[2],heading[0]))
        h,w=image.shape[:2]
        model=ActiveViewScorer(P/"jepa_reid_v25/active/weights/jepa.npz")
        scores=model.score(image,np.asarray(ref.roi)*[w,h,w,h],[relative_action(anchor,e,1.) for e,_,_ in options])
        rank=lambda i:(scores[i],-len(options[i][1]))
        first=max(range(len(options)),key=rank)
        assert options[first][1][0]==FORWARD
        frame,r=physical(env,FORWARD)
        assert r["collided"] and r["translation_m"]<.005
        remaining=12-1
        origin=(0.,0.,pose[2])
        eligible=[i for i,(_,commands,_) in enumerate(options) if i!=first and len(commands)<=remaining
                  and avoids(commands,origin,origin)]
        total=1; path_m=r["translation_m"];planned=0;reblocked=False
        if eligible:
            next_i=max(eligible,key=rank)
            commands=options[next_i][1];planned=len(commands)
            for a in commands:
                frame,r=physical(env,a)
                total+=1;path_m+=r["translation_m"]
                if a==FORWARD and r["translation_m"]<.005:
                    reblocked=True
                    break
        assert total<=12
        end=audit(frame,item,env,det,ref,target_id)
        result=dict(budget=12,pool=len(options),first_step_stalled=True,safe_unvisited_routes=len(eligible),
                    retries=int(bool(eligible)),fallback_planned_steps=planned,steps_used=total,
                    travel_m=round(path_m,3),fallback_stalled=reblocked,endpoint=end,paid_calls=0)
    finally:
        det.close();env.close()
    (P/"jepa_reid_v25/hardcase/retry_results.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result))


if __name__=="__main__":
    main()
