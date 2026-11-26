"""Audit whether a semantic box measures the object visible in the RGB."""
from pathlib import Path
import json, os
import numpy as np
import cv2
from PIL import Image, ImageDraw
from diagnose_images import rows
from capture_panel import read
from check_routes import semantic_scene
from evaluation.main.evaluator import _reset_environment
from historical_trial import frozen
from verify_habitat import replay_until


def summarize(project,private):
    panel=rows(private/"manifest.jsonl"); gt=rows(private/"gt.jsonl")
    current=rows(project/"jepa_box_v26/compare/private/habitat_dino.jsonl")
    report={}
    grid=Image.new("RGB",(3*420,2*420),(255,255,255))
    for j,(fid,f) in enumerate((k,r) for k,r in panel.items() if r["category"]=="clock" and r["group"]=="archived"):
        rgb=np.asarray(Image.open(private/f["rgb"]).convert("RGB"))
        t=gt[fid]; tsi=max(t["boxes"],key=lambda q:q["pixels"])["instance"]
        m=(np.asarray(Image.open(private/t["mask"]))==tsi).astype("uint8")
        n,labels,stats,centroids=cv2.connectedComponentsWithStats(m,connectivity=8)
        box=np.array(current[fid]["boxes"][0]["bbox"])*320/518
        x,y,u,v=np.clip(np.rint(box),0,320).astype(int)
        inside=set(np.unique(labels[y:v,x:u]))-{0}
        inside_cc_pixels=int(sum(stats[i,-1] for i in inside))
        sizes=sorted([int(s[-1]) for s in stats[1:]],reverse=True)
        rgb_overlay=rgb.astype(float)
        mask=cv2.resize(m,rgb.shape[1::-1],interpolation=cv2.INTER_NEAREST).astype(bool)
        rgb_overlay[mask]=.43*rgb_overlay[mask]+.57*np.array([20,255,60])
        rgb_overlay=Image.fromarray(rgb_overlay.astype("uint8"))
        grid.paste(Image.fromarray(rgb).resize((420,420)),(j*420,0))
        grid.paste(rgb_overlay.resize((420,420)),(j*420,420))
        report[fid.removeprefix("clock_archived_")]=dict(num_components=n-1,
            total_target_pixels=int(m.sum()),pixels_in_top_box=int(m[y:v,x:u].sum()),
            pixels_in_all_components_touching_top_box=inside_cc_pixels,
            pixels_in_components_disconnected_from_box=int(m.sum()-inside_cc_pixels),
            largest_component_pixels=sizes[:5])
    grid.save(private/"clock_pixel_audit.png")
    pilot=project/"mapping_runtime/pilots/jepa_reid_v24_first2"
    eid=read(pilot/"panel.json")["rows"][0]["episode_id"]
    doc=read(Path(read(pilot/"comparison/accepted/second"/(eid+".json"))["result"]))
    env,item,data=semantic_scene(doc)
    try:
        _reset_environment(env,item,data)
        ss=env.sim.habitat_config.agents.main_agent.sim_sensors
        a,b=ss.rgb,ss.semantic
        pose=env.sim.get_agent_state().sensor_states
        report["camera_geometry"]=dict(rgb_hfov=float(a.hfov),semantic_hfov=float(b.hfov),
            rgb_size=[int(a.width),int(a.height)],semantic_size=[int(b.width),int(b.height)],
            orientation_config_equal=bool(np.allclose(a.orientation,b.orientation)),
            sensor_position_difference_m=float(np.linalg.norm(pose["rgb"].position-pose["semantic"].position)),
            quaternion_difference=float(min(abs(pose["rgb"].rotation-pose["semantic"].rotation),abs(pose["rgb"].rotation+pose["semantic"].rotation))))
        _,steps,start,_,_,_=frozen()
        frame=replay_until(env,item,data,steps,start["step"]+len(start["planned_actions"]))
        raw=np.asarray(frame["semantic"]).squeeze()
        rawids=list(map(int,item.target_object_ids))
        main_id=max(rawids,key=lambda q:(raw==q).sum())
        scene=env.sim.semantic_scene
        object_records=[]
        for ob in scene.objects:
            if ob is None: continue
            key=int(ob.semantic_id) if hasattr(ob,"semantic_id") else int(str(ob.id).split("_")[-1])
            if key in rawids:
                object_records.append({"is_closeup_instance":key==main_id,"category":ob.category.name()})
        report["semantic_identity"]=dict(ids_positive=all(k>0 for k in rawids),
            episode_instances=len(rawids),matched_semantic_scene_records=object_records,
            mask_equal_to_selected_raw_id=bool(np.array_equal(raw==main_id,
                np.asarray(Image.open(private/gt["clock_archived_after_approach"]["mask"]))==rawids.index(main_id)+1)))
    finally:
        env.close()
    (project/"jepa_box_v26/habitat/alignment_audit.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    p=Path(os.environ["NAV_ROOT"]);summarize(p,p/"jepa_box_v26/habitat/private")
