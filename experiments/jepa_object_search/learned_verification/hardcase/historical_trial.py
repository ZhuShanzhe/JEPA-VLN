"""Three closed physical branches from the first archived V24 clock inspection failure."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from habitat.utils.geometry_utils import quaternion_rotate_vector
from agents import navigator
from active_view_score import ActiveViewScorer
from appearance import build_reference,match_reference
from confirm_view import FORWARD, pose_after
from evaluation.main.evaluator import _near_target_ids
from learned_map_views import map_views,relative_action
from proposal_worker import ProposalWorker
from report_evidence import current_support,eligible_boxes,overlap
from verify_views import physical,semantic_scene,replay_until,rgb


P=Path(os.environ["NAV_ROOT"])
def load(p): return json.loads(p.read_bytes())
def rows(p): return [json.loads(line) for line in p.read_bytes().splitlines()]


def frozen():
    pilot=P/"mapping_runtime/pilots/jepa_reid_v24_first2"
    eid=load(pilot/"panel.json")["rows"][0]["episode_id"]
    ticket=load(pilot/"comparison/accepted/second"/(eid+".json"))
    result=Path(ticket["result"])
    assert hashlib.sha256(result.read_bytes()).hexdigest()==ticket["result_sha256"]
    doc=load(result); ep,=doc["episodes"]; log=result.parent/"agent"
    timeline=rows(log/"confirm_events.jsonl")
    check=next(x for x in timeline if x["event"]=="confirm_view_recheck" and x["reason"]=="no_current_instance_match")
    start=next(x for x in reversed(timeline[:timeline.index(check)]) if x["event"]=="confirm_view_start" and
               (x["frame"],x["instance"])==(check["frame"],check["instance"]))
    first=next(x for x in timeline if x["event"]=="confirm_view_action" and x["step"]==start["step"])
    trace=ep["paper"]["agent_trace"]
    n=next(n for n in trace["final_memory"]["instances"] if n["instance_id"]==start["instance"])
    ob,=[o for o in trace["observations"] if o["observation_id"] in n["observation_ids"] and o["frame_id"]==start["frame"] and o["step"]<=start["step"]]
    frame,=[r for r in rows(log/"action_trace.jsonl") if r.get("mapping_frame_id")==start["frame"] and (r.get("mapping") or {}).get("is_keyframe")]
    times=ep["paper"]["steps"]
    assert start["planned_actions"]==[x["action"] for x in times[start["step"]:check["step"]]]
    return doc,times,start,first["pose"],ob,frame["step"]


def registered_image(frame,ob):
    source=P/"private_sources/MOST-Nav/mapping/server.py"
    tree=ast.parse(source.read_text())
    fn,=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="_vggt_frame_grid"]
    fn.decorator_list=[]
    ns={"np":np,"Image":Image}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),"exec"),ns)
    mapped=ns["_vggt_frame_grid"](rgb(frame))
    ok,b=cv2.imencode(".jpg",cv2.cvtColor(mapped,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,85]);assert ok
    image=cv2.cvtColor(cv2.imdecode(b,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    ref=build_reference(b.tobytes(),ob["bbox"],point_1000=ob["pixel"])
    assert ref is not None
    return image,ref


def candidates(times,start,pose,physical_heading):
    # Known nominal metre displacement divided by online map displacement at the planned endpoint.
    commands=tuple(start["planned_actions"])
    nominal=tuple(pose)
    for a in commands: nominal=pose_after(nominal,a,1.)
    scale=math.dist(nominal[:2],pose[:2])/math.dist(pose[:2],start["waypoint"][:2])
    assert math.isfinite(scale) and scale>0
    source=start["reference_pose"]
    offset=lambda xy:(np.array(xy[:2])-pose[:2])*scale
    anchor=(*offset(source),source[2]);cur=(0.,0.,pose[2])
    target=offset(start["target"])
    prefix=times[:start["step"]+1]
    xz=np.asarray([x["position_before"] for x in prefix])[:,[0,2]]
    delta=xz-xz[-1]
    forward=delta@np.array([math.cos(physical_heading),math.sin(physical_heading)])
    right=delta@np.array([-math.sin(physical_heading),math.cos(physical_heading)])
    f=np.array([math.cos(cur[2]),math.sin(cur[2])]);rr=np.array([f[1],-f[0]])
    history=forward[:,None]*f+right[:,None]*rr
    original=[cur]
    for a in commands: original.append(pose_after(original[-1],a,1.))
    assert np.allclose(original[-1][:2],offset(start["waypoint"]),atol=1e-4)
    # Map consists only of odometry previously traversed and the corridor already proposed online.
    res=.05;low=np.array([-3.,-3.]);centres=low[0]+(np.arange(120)+.5)*res
    xx,yy=np.meshgrid(centres,centres)
    free=np.zeros(xx.shape,bool)
    for trace in (history,np.array([q[:2] for q in original])):
        for a,b in zip(trace[:-1],trace[1:]):
            steps=max(2,int(np.ceil(np.linalg.norm(b-a)/.10)))
            for t in np.linspace(0,1,steps):
                x,y=(1-t)*a+t*b
                free|=(xx-x)**2+(yy-y)**2<=.23**2
    grid=navigator.OccupancyGrid(res,low,free,np.zeros_like(free))
    routes=[(original[-1],commands,"archived_geometric")]
    for end,seq,kind in map_views(grid,cur,target,anchor,1.,[]):
        if FORWARD in seq and tuple(seq) not in {r[1] for r in routes}:
            routes.append((end,tuple(seq),kind))
    return anchor,routes,scale,target


def audit(frame,item,env,det,ref,target_id):
    im=rgb(frame); mask=np.asarray(frame["semantic"]).squeeze()
    boxes=eligible_boxes(det.detect(im,"distinct clock objects"),im.shape)
    patch=match_reference(ref,im); support,reason=current_support(ref,patch,im,boxes)
    true=mask==target_id; y,x=np.nonzero(true)
    gt=[x.min(),y.min(),x.max()+1,y.max()+1] if len(x) else None
    near=_near_target_ids(tuple(env.sim.get_agent_state().position),item.target_object_ids,item.target_positions,
        threshold=.25,pathfinder=env.sim.pathfinder,target_view_points=item.view_points)
    return dict(target_pixels=len(x),in_legal_range=target_id in set(map(int,near)),live_boxes=len(boxes),
                best_box_iou=float(max((overlap(b,gt) for b in boxes),default=0)) if len(x) else 0.,
                best_target_coverage=max((float(true[b[1]:b[3],b[0]:b[2]].sum())/len(x) for b in boxes),default=0.) if len(x) else 0.,
                appearance_support=bool(support is not None),appearance_reason=reason)


def main():
    doc,times,start,pose,ob,ref_step=frozen()
    env,item,data=semantic_scene(doc)
    detector=ProposalWorker(P/"b_verification/env/bin/python",P,"cuda")
    try:
        source=replay_until(env,item,data,times,ref_step)
        registered,ref=registered_image(source,ob)
        # Oracles below are used only by audit(), never by the proposal or ranking functions.
        ids=np.asarray(source["semantic"]).squeeze()
        iy,ix=(np.array(ob["pixel"])[::-1]/1000*np.array(ids.shape)).round().astype(int)
        target_id=int(ids[iy,ix]); assert target_id in set(map(int,item.target_object_ids))
        current=replay_until(env,item,data,times,start["step"])
        heading=quaternion_rotate_vector(env.sim.get_agent_state().rotation,np.array([0.,0.,-1.]))
        anchor,options,scale,target=candidates(times,start,pose,math.atan2(heading[2],heading[0]))
        assert len(options)>=2
        h,w=registered.shape[:2]; box=np.asarray(ref.roi)*[w,h,w,h]
        moves=[relative_action(anchor,o[0],1.) for o in options]
        choices={"geometric":0}
        for name in ("direct","jepa"):
            scorer=ActiveViewScorer(P/"jepa_reid_v25/active/weights"/(name+".npz"))
            s=scorer.score(registered,box,moves)
            choices[name]=max(range(len(options)),key=lambda i:(s[i],-len(options[i][1])))
        out=dict(actions_allowed=12,eligible_routes=len(options),source_box_fraction=float((box[2]-box[0])*(box[3]-box[1])/(w*h)),
                 controls="Fixed one historical start. Ideal prefix odometry and the already-planned old path supply a traversability proxy. Target estimate and reference pose from the old online agent. Every eligible path includes forward motion.",
                 paid_calls=0,initial=audit(current,item,env,detector,ref,target_id),branches={})
        for name,index in choices.items():
            frame=replay_until(env,item,data,times,start["step"])
            end,commands,kind=options[index]
            total=0.;blocked=0
            for steps,a in enumerate(commands,1):
                frame,receipt=physical(env,a)
                total+=receipt["translation_m"];blocked+=receipt["collided"]
                # Habitat may report contact even while sliding. Stop on failed
                # physical motion as in the previous real-view receipt fixture.
                if a==FORWARD and receipt["translation_m"]<.005: break
            out["branches"][name]=dict(steps=steps,collisions=blocked,
                 path_m=round(total,3),same_route_as_geometric=index==0,selected_route_index=index,
                 route_type=kind,endpoint=audit(frame,item,env,detector,ref,target_id))
        # Exact archived physical control is independently known from the V24 audit.
        print("Physical comparison",json.dumps(out),flush=True)
        assert out["branches"]["geometric"]["endpoint"]["target_pixels"]==17968
        assert not out["branches"]["geometric"]["endpoint"]["appearance_support"]
    finally:
        detector.close();env.close()
    (P/"jepa_reid_v25/hardcase/results.json").write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps(out),flush=True)


if __name__=="__main__":
    main()
