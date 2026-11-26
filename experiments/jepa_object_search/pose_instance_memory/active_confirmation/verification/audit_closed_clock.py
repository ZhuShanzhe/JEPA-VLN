"""Read-only post-hoc audit; keep the underlying frames, poses and IDs on the server."""
import os, sys, json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
P=Path(os.environ["NAV_ROOT"])
for relative in ("jepa_exec_v14","jepa_completion_v17/physical"):
    sys.path.insert(0,str(P/relative))
from verify_habitat import replay_until,deny_network,state
sys.addaudithook(deny_network)
from check_routes import semantic_scene
from evaluation.main.evaluator import _near_target_ids
def load(p):return json.loads(p.read_bytes())
def rows(p):return [json.loads(x) for x in p.read_bytes().splitlines() if x]
p=P/"mapping_runtime/pilots/jepa_reid_v24_first2"; assert (p/"closed.json").is_file()
eid=load(p/"panel.json")["rows"][0]["episode_id"]
receipt=load(p/"comparison/accepted/second"/(eid+".json")); path=Path(receipt["result"])
assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt["result_sha256"]
doc=load(path); ep,=doc["episodes"]; hist=ep["paper"]["agent_trace"]; times=ep["paper"]["steps"]
d=path.parent/"agent"; c=rows(d/"confirm_events.jsonl"); a=rows(d/"action_trace.jsonl")
end=next(e for e in c if e["event"]=="confirm_view_recheck" and e["reason"]=="no_current_instance_match")
start=next(r for r in reversed(c[:c.index(end)]) if r["event"]=="confirm_view_start" and r["frame"]==end["frame"] and r["instance"]==end["instance"])
fid=end["frame"]; node=next(r for r in hist["final_memory"]["instances"] if r["instance_id"]==end["instance"])
o,=[o for o in hist["observations"] if o["observation_id"] in node["observation_ids"] and o["frame_id"]==fid and o["step"]<=start["step"]]
keyframe,=[r for r in a if r.get("mapping",{}).get("is_keyframe") and r.get("mapping_frame_id")==fid]
proposal,=[r for r in rows(d/"candidate_evidence.jsonl") if r.get("event")=="proposal" and r.get("candidate_id")==o["candidate_id"]]
assert proposal["frame_id"]==fid and np.max(np.abs(np.asarray(o["bbox"])-proposal["bbox"]))<1e-4
assert len(start["planned_actions"])==end["step"]-start["step"]==3
assert start["planned_actions"]==[x["action"] for x in times[start["step"]:end["step"]]]
pixel,norm=np.asarray(o["pixel"],dtype=float),np.asarray(proposal["pixel"],dtype=float)
assert np.min(pixel)>0 and np.min(norm)>0 and np.max(norm)<=1000
# Pixel and normalized click from the same mapper receipt fix the coordinate
# scale across the raster. Vary the quantized input by 1 pixel to bound rounding.
# In the paper export "pixel" is already normalized to 1,000, independent of bbox.
# The official mapping server registers its mask bbox on the VGGT pixel grid.
from PIL import Image
import ast
server=P/"private_sources/MOST-Nav/mapping/server.py"
tree=ast.parse(server.read_text())
grid_fn,=[x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=="_vggt_frame_grid"]
grid_fn.decorator_list=[]
fncode=compile(ast.fix_missing_locations(ast.Module(body=[grid_fn],type_ignores=[])),str(server),"exec")
scope={"np":np,"Image":Image}
exec(fncode,scope)
assert np.allclose(norm,pixel)
dims=np.array(scope["_vggt_frame_grid"](np.zeros((1280,1280,3),np.uint8)).shape[1::-1])
bounds=np.asarray(o["bbox"],dtype=float)/np.tile(dims,2)
assert np.all(norm/1000>=bounds[:2]-.025) and np.all(norm/1000<=bounds[2:]+.025)
assert (bounds>=-.02).all() and (bounds<=1.02).all() and bounds[0]<bounds[2] and bounds[1]<bounds[3]
env,item,data=semantic_scene(doc)
try:
    readings={}; audit_steps={}
    sys.path.insert(0,str(P/"jepa_reid_v21/detector"))
    from proposal_worker import ProposalWorker
    from appearance import build_reference,match_reference
    from report_evidence import eligible_boxes,current_support
    import cv2
    for name,t in (("registered",keyframe["step"]),("start",start["step"]),("finish",end["step"])):
        frame=replay_until(env,item,data,times,t)
        assert frame["rgb"].shape[:2]==(1280,1280)
        mask=np.asarray(frame["semantic"]).squeeze(); ids=list(map(int,item.target_object_ids))
        assert tuple(mask.shape)==(320,320)
        visible={i:int((mask==i).sum()) for i in ids}
        near=set(map(int,_near_target_ids(tuple(state(env)[0]),item.target_object_ids,item.target_positions,
                          threshold=.25,pathfinder=env.sim.pathfinder,target_view_points=item.view_points)))
        audit_steps[name]=(max(visible,key=visible.get),near)
        if name=="registered":
            # Reproduce the RGB geometry of the original mapper's get_frame_image.
            mapped=scope["_vggt_frame_grid"](np.asarray(frame["rgb"])[...,:3])
            ok,enc=cv2.imencode(".jpg",cv2.cvtColor(mapped,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,85])
            ref=build_reference(enc.tobytes(),o["bbox"],point_1000=norm)
            assert ok and ref is not None
            bb=np.clip(np.round(bounds*320).astype(int),0,320)
            x,y,u,v=bb
            ix,iy=np.minimum(319,np.maximum(0,np.round(norm*.32).astype(int)))
            hit={i:int((mask[y:v,x:u]==i).sum()) for i in ids}
            conservative_hit={i:int((mask[min(y+2,319):max(y+2,v-2),min(x+2,319):max(x+2,u-2)]==i).sum()) for i in ids}
            expanded_hit={i:int((mask[max(0,y-2):min(320,v+2),max(0,x-2):min(320,u+2)]==i).sum()) for i in ids}
            best=max(hit,key=hit.get)
            correct=hit[best]>0
            source_id=best if correct else None
            audit=dict(any_task_target_pixels_in_box=sum(hit.values()),most_task_instance_pixels_in_box=hit[best],
                       box_area=int((u-x)*(v-y)),click_on_task_instance=int(mask[iy,ix]) in ids,
                       max_instance_pixels_in_inset=conservative_hit[best],all_task_pixels_in_expanded_box=sum(expanded_hit.values()),
                       box_dominant_target_same_as_click=bool(correct and int(mask[iy,ix])==best),
                       scene_target_instances_overlapping=sum(v>0 for v in hit.values()),max_task_pixels_in_source=max(visible.values()),
                       unique_image_and_registration=True,source_age_when_rechecked=end["step"]-t)
        if name=="finish":
            image=np.asarray(frame["rgb"])[...,:3].copy()
            det=ProposalWorker(P/"b_verification/env/bin/python",P,"cuda")
            try: boxes=eligible_boxes(det.detect(image,"distinct clock objects"),image.shape)
            finally: det.close()
            target=mask==source_id
            fy,fx=np.where(target)
            rect=(fx.min(),fy.min(),fx.max()+1,fy.max()+1) if len(fy) else None
            from report_evidence import overlap
            fracs=[]; cover=[]; ious=[]
            for x,y,u,v in boxes:
                n=int(target[y:v,x:u].sum())
                fracs.append(n/max(1,(v-y)*(u-x))); cover.append(n/max(1,len(fy)))
                ious.append(overlap((x,y,u,v),rect) if rect is not None else 0)
            patch=match_reference(ref,image)
            accepted,reason=current_support(ref,patch,image,boxes)
            live=dict(eligible_boxes=len(boxes),max_instance_pixel_coverage=round(max(cover,default=0),4),
                      max_box_target_pixel_fraction=round(max(fracs,default=0),4),
                      max_iou_with_semantic_bounds=round(float(max(ious,default=0)),4),
                      original_template_match=bool(patch.matched),original_template_reason=patch.reason,
                      replayed_report_support=accepted is not None,replayed_support_reason=reason)
        readings[name]=dict(task_instances_visible_1pix=sum(v>0 for v in visible.values()),task_instances_visible_103px=sum(v>=103 for v in visible.values()),
                            max_task_target_pixels=max(visible.values()),task_instances_in_report_range=len(near),
                            source_instance_pixels=visible[source_id] if source_id is not None else None,
                            source_instance_in_report_range=source_id in near if source_id is not None else None)
    result=dict(offline_frozen_detector=live,same_most_visible_instance_start_and_finish=audit_steps["start"][0]==audit_steps["finish"][0],same_near_instance_start_and_finish=audit_steps["start"][1]==audit_steps["finish"][1],mapper_evidence_grid=list(map(int,dims)),scope="First accepted clock recheck ending no_current_instance_match",paid_requests=0,actions=len(start["planned_actions"]),
                forward_actions=sum(x==1 for x in start["planned_actions"]),view_strategy=start["view_source"],geometry_receipt=audit,states=readings)
finally:
    env.close()
out=P/"jepa_reid_v24/verification/closed_clock_confirm_audit.json"; out.write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result))
