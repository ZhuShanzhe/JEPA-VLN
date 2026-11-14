"""Run from the Habitat env with a headless EGL renderer; output private observations."""
import hashlib
import json
from pathlib import Path

import habitat_sim as hs
import numpy as np
import magnum as mn
from PIL import Image

here = Path(__file__).resolve().parents[1]
room = here/"room"
audit = here/"acceptance"
audit.mkdir(exist_ok=True)
cfg = hs.SimulatorConfiguration()
cfg.scene_id = str(room/"room.stage_config.json")
cfg.enable_physics = True
cfg.override_scene_light_defaults = True
cfg.scene_light_setup = hs.gfx.DEFAULT_LIGHTING_KEY
a = hs.agent.AgentConfiguration(height=1.45, radius=.12)
a.sensor_specifications = []
for name, kind in (("rgb",hs.SensorType.COLOR),("depth",hs.SensorType.DEPTH),("semantic",hs.SensorType.SEMANTIC)):
    cam = hs.CameraSensorSpec()
    cam.uuid,cam.sensor_type = name,kind
    cam.position,cam.resolution,cam.hfov = mn.Vector3(0,1.15,0),[256,256],68
    a.sensor_specifications.append(cam)
a.action_space = {key:hs.agent.ActionSpec(key,hs.agent.ActuationSpec(amount=amount))
                  for key,amount in (("move_forward",.25),("turn_left",15.),("turn_right",15.))}
obs = {}
with hs.Simulator(hs.Configuration(cfg,[a])) as sim:
    tm = sim.get_object_template_manager()
    tids = tm.load_configs(str(room/"chair.object_config.json"))
    assert len(tids)==1
    om = sim.get_rigid_object_manager()
    for obj_id,x in ((101,-2.25),(102,2.25)):
        obj=om.add_object_by_template_id(tids[0])
        obj.translation=mn.Vector3(x,0,-1.65)
        obj.motion_type=hs.physics.MotionType.STATIC
        obj.semantic_id=obj_id
    nav=hs.NavMeshSettings()
    nav.set_defaults()
    nav.agent_height,nav.agent_radius,nav.include_static_objects=1.45,.12,True
    assert sim.recompute_navmesh(sim.pathfinder,nav)
    assert sim.pathfinder.save_nav_mesh(str(room/"room.navmesh"))
    start=np.array(sim.pathfinder.snap_point(np.array([0.,0.,3.75])))
    assert sim.pathfinder.is_navigable(start)
    labels={}
    witnesses={}; detections={}; poses={}; histories={}; candidates={}
    for name,other,expected in (("left","right",101),("right","left",102)):
        state=hs.AgentState()
        state.position=start
        sim.get_agent(0).set_state(state)
        seq=["turn_"+name]*3+["move_forward"]*10+["turn_"+other]*3+["move_forward"]*3
        lateral=10*.25/np.sqrt(2)*(-1 if name=="left" else 1)
        candidates[name]=dict(planned_actions=seq, target_relative_xz_m=[float(lateral),float(-abs(lateral)-.75)])
        seen_counts=[]
        frames=[]; depths=[]; masks=[]; states=[]
        for i,act in enumerate([None]+seq):
            frame=sim.step(act) if act else sim.get_sensor_observations()
            assert not frame.get("collided",False)
            gt,rgb,depth=frame["semantic"],frame["rgb"][:,:,:3],frame["depth"]
            assert np.isfinite(depth).all() and rgb.std()>4 and depth.max()>depth.min()
            states.append(np.asarray(sim.get_agent(0).state.position).tolist())
            frames.append(rgb.copy());depths.append(depth.astype(np.float16));masks.append(gt.astype(np.uint16))
            seen_counts.append([int((gt==k).sum()) for k in (101,102)])
        file=audit/(name+"_trajectory.npz")
        np.savez_compressed(file,rgb=np.stack(frames),depth=np.stack(depths),judge_instance_ids=np.stack(masks),positions=np.array(states),actions=np.array(seq))
        assert np.allclose(states[0],start)
        if name=="left": initial=frames[0].copy()
        else: assert np.array_equal(initial,frames[0])
        Image.fromarray(frames[-1]).save(audit/(name+"_final.png"))
        counts=np.asarray(seen_counts)
        detections[name]=counts.max(axis=0).tolist()
        assert max(counts[0])<64,counts[0]
        assert counts[:,expected-101].max()>200
        assert counts[:,1-(expected-101)].max()<64,(name,counts.max(axis=0))
        assert np.linalg.norm(np.array(states[-1])-start)>2
        witnesses[name]=hashlib.sha256(file.read_bytes()).hexdigest()
        poses[name]=states[-1]
        j=int(counts[:,expected-101].argmax())
        yy,xx=np.where(masks[j]==expected)
        report_img=f"{name}_report.png"
        Image.fromarray(frames[j]).save(audit/report_img)
        inverse={"turn_left":"turn_right","turn_right":"turn_left","move_forward":"move_forward"}
        back=["turn_left"]*12 + [inverse[act] for act in reversed(seq)] + ["turn_left"]*12
        for act in back:
            assert not sim.step(act).get("collided",False)
        returned=sim.get_agent(0).state
        err=float(np.linalg.norm(np.asarray(returned.position)-start))
        assert err<.02 and abs(returned.rotation.w)>.999
        histories[name]=dict(report_rgb=report_img, report_bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                             report_step=j, report_observed_pose=states[j], return_actions=back,
                             return_position_error_m=err, evidence_pixels=int(counts[j,expected-101]))
    Image.fromarray(initial).save(audit/"start.png")
    for prior in (101,102):
        labels[str(prior)]={k:int(any(i!=prior and v>=64 for i,v in zip((101,102),vv))) for k,vv in detections.items()}
    assert labels=={"101":{"left":0,"right":1},"102":{"left":1,"right":0}}
    histories["left"]["candidate_labels"]=labels["101"]
    histories["right"]["candidate_labels"]=labels["102"]
    receipt=dict(status="pass",habitat_sim=hs.__version__,navigable_area=float(sim.pathfinder.navigable_area),
                 start=start.tolist(),endpoints=poses,actions_per_branch=len(seq),rgb_frames_per_branch=len(seq)+1,
                 max_judge_instance_pixels=detections,synthetic_prior_report_whatif=labels,
                 report_histories=histories,candidates=candidates,
                 scope="Actual out-and-back observation/evidence, constructed report events, engineering fixture only. Semantic IDs/bboxes are offline judge information. Agent sees report RGB, observed report pose, and planned actions.",
                 blender_generated_assets=True,scene_sha256=hashlib.sha256((room/"room.glb").read_bytes()).hexdigest(),
                 trajectory_sha256=witnesses)
    (audit/"scene_smoke.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))
