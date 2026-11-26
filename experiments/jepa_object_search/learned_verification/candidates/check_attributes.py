"""Score unchanged closed-loop source views and a separate, explicitly posed sensitivity control."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from replay_candidates import P,load,semantic_scene,state,ProposalWorker
from attribute_screen import AttributeWorker,local_crop
from source_quality import best,coarse_category,overlap


def assess(rgb,mask,records,ids,attr,text,spec):
    bestq,ix=best(spec,records,518,518)
    counts=[int(np.count_nonzero(mask==int(tid))) for tid in ids]
    result=dict(max_pixels=max(counts),visible=max(counts)>=103,admitted=bestq>=spec["threshold"],
        attribute_pass=False,localized=False,has_frame_target=bool(max(counts)))
    if not result["admitted"] or ix is None:return result
    box=records[ix]["bbox"]; crop=local_crop(rgb,box)
    if min(crop.shape[:2])<8:return result
    margin=attr.margin(crop,text)
    result.update(attribute_pass=margin>0,margin=margin)
    for tid,count in zip(ids,counts):
        if count<103:continue
        y,x=np.nonzero(mask==int(tid))
        q=np.array([x.min(),y.min(),x.max()+1,y.max()+1])*518/320
        result["localized"] |= bool(overlap(box,q)>=.5)
    return result


def summarize(rows):
    return dict(frames=len(rows),some_instance_pixels=sum(r["has_frame_target"] for r in rows),
        visible=sum(r["visible"] for r in rows),category_accepted=sum(r["admitted"] for r in rows),
        full_goal_accepted=sum(r["attribute_pass"] for r in rows),
        category_localized=sum(r["localized"] for r in rows),full_goal_localized=sum(r["localized"] and r["attribute_pass"] for r in rows))


def test(index,name,attr,worker,spec):
    dest=Path(__file__).resolve().parent
    cache=load(dest/"private_replay"/(name+".json"))
    frames={f["step"]:f for f in cache["frames"]}
    p=P/"mapping_runtime/pilots/jepa_reid_v24_first2"; row=load(p/"panel.json")["rows"][index]
    path=Path(load(p/"comparison/accepted/second"/(row["episode_id"]+".json"))["result"]);doc=load(path); ep,=doc["episodes"]
    env,item,data=semantic_scene(doc); ids=item.target_object_ids; goal=ep["goal_text"]
    hist=[]; controls=[]
    from habitat_sim.utils.common import quat_from_angle_axis
    try:
        f=_reset_environment(env,item,data)
        def sensors(f):
            rgb=np.asarray(Image.fromarray(np.asarray(f["rgb"])[...,:3]).resize((518,518),Image.Resampling.BICUBIC))
            return rgb,np.asarray(f["semantic"]).squeeze()
        for step,act in enumerate(ep["paper"]["steps"]):
            if step in frames:
                assert np.linalg.norm(state(env)[0]-act["position_before"])<.002
                rgb,mask=sensors(f); rr=assess(rgb,mask,frames[step]["boxes"],ids,attr,goal,spec)
                assert rr["visible"]==bool(frames[step]["truth"])
                hist.append(dict(fid=frames[step]["fid"],**rr))
            if act["action_role"]=="navigate":f=env.step(_env_action(act["action"]))
        # Oracle camera placement is used only to test the detector, never by the online policy.
        for tid in ids:
            points=ep["paper"]["episode_definition"]["view_points"][str(tid)]
            center=np.asarray(ep["paper"]["episode_definition"]["target_positions"][str(tid)])
            for j in np.linspace(0,len(points)-1, min(6,len(points))).round().astype(int):
                point=np.asarray(points[j])
                dx,dy,dz=center-point
                quat=quat_from_angle_axis(np.arctan2(-dx,-dz),np.array([0.,1.,0.]))
                f=env.sim.get_observations_at(position=point,rotation=quat,keep_agent_at_new_pose=False)
                rgb,mask=sensors(f); boxes=worker.detect(rgb,coarse_category(goal))
                controls.append(assess(rgb,mask,boxes,ids,attr,goal,spec))
    finally:env.close()
    (dest/"private_replay"/(name+"_attributes.json")).write_text(json.dumps(hist))
    return dict(frozen_trajectory=summarize(hist),oracle_viewpoint_stress_control=summarize(controls))


if __name__=="__main__":
    from evaluation.main.evaluator import _reset_environment,_env_action
    dest=Path(__file__).resolve().parent; selector=load(dest/"selector.json")
    attr=AttributeWorker(P,"cuda")
    worker=ProposalWorker(P/"b_verification/env/bin/python",P,"cuda")
    try: out={n:test(i,n,attr,worker,selector) for i,n in enumerate(("clock","painting"))}
    finally:attr.close();worker.close()
    (dest/"attribute_controls.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out))
