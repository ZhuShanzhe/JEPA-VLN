"""Cross two queries with two material permutations, fixed camera motion and identical geometry."""
import json
from pathlib import Path
import habitat_sim as hs
import magnum as mn
import numpy as np
from PIL import Image

root=Path(__file__).resolve().parents[1]
out=root/"attribute_fixture"
clips=out/"clips"
reports=out/"reports"
clips.mkdir(exist_ok=True)
reports.mkdir(exist_ok=True)
assets=out/"assets"
for mesh in sorted(assets.glob("*.glb")):
    mesh.with_suffix(".object_config.json").write_text(json.dumps(dict(
        render_asset=mesh.name,collision_asset=mesh.name,mass=1.,COM=[0,0,0],use_bounding_box_for_collision=False)))
base=json.loads((root/"acceptance/scene_smoke.json").read_text())
cfg=hs.SimulatorConfiguration()
cfg.scene_id=str(root/"room/room.stage_config.json")
cfg.enable_physics=True
cfg.override_scene_light_defaults=True
cfg.scene_light_setup=hs.gfx.DEFAULT_LIGHTING_KEY
agent=hs.agent.AgentConfiguration(height=1.45,radius=.12)
agent.sensor_specifications=[]
for key,kind in (("rgb",hs.SensorType.COLOR),("depth",hs.SensorType.DEPTH),("judge",hs.SensorType.SEMANTIC)):
    c=hs.CameraSensorSpec()
    c.uuid,c.sensor_type,c.position,c.resolution,c.hfov=key,kind,mn.Vector3(0,1.15,0),[256,256],68
    agent.sensor_specifications.append(c)
agent.action_space={k:hs.agent.ActionSpec(k,hs.agent.ActuationSpec(amount=v)) for k,v in
                       (("move_forward",.25),("turn_left",15.),("turn_right",15.))}

def colors(rgb):
    r,g,b=np.moveaxis(rgb.astype(np.float32),-1,0)
    return {"red":(r>1.4*b)&(r>1.35*g)&(r>50), "blue":(b>1.4*r)&(b>1.35*g)&(b>50)}

seen={}
rows=[]
permutations=(("red_left",("red","blue")),("blue_left",("blue","red")))
for layout,palette in permutations:
    with hs.Simulator(hs.Configuration(cfg,[agent])) as sim:
        tm=sim.get_object_template_manager()
        om=sim.get_rigid_object_manager()
        first=layout==permutations[0][0]
        for idx,sgn in enumerate((-1,1)):
            for name,xyz,instance in (
                ("chair",(sgn*2.25,0,-1.65),101+idx),("marker",(sgn*.80,1.26,1.76),201+idx)):
                key=name+"_"+palette[idx]+".object_config.json"
                ids=tm.load_configs(str(assets/key))
                obj=om.add_object_by_template_id(ids[0])
                obj.translation=mn.Vector3(*xyz)
                obj.motion_type=hs.physics.MotionType.STATIC
                obj.semantic_id=instance
        nav=hs.NavMeshSettings()
        nav.set_defaults()
        nav.agent_height,nav.agent_radius,nav.include_static_objects=1.45,.12,True
        assert sim.recompute_navmesh(sim.pathfinder,nav)
        if first: nav_area=float(sim.pathfinder.navigable_area)
        else: assert abs(sim.pathfinder.navigable_area-nav_area)<1e-5
        start=np.asarray(base["start"])
        assert sim.pathfinder.is_navigable(start)
        judged={}
        reachable={}
        for side in ("left","right"):
            st=hs.AgentState()
            st.position=start
            sim.get_agent(0).set_state(st)
            seq=base["candidates"][side]["planned_actions"]
            rgb=[]; depth=[]; masks=[]; poses=[]
            for act in [None]+seq:
                f=sim.step(act) if act else sim.get_sensor_observations()
                assert not f.get("collided",False) and np.isfinite(f["depth"]).all()
                rgb.append(f["rgb"][:,:,:3].copy())
                depth.append(f["depth"].copy())
                masks.append(f["judge"].copy())
                pose=sim.get_agent(0).state
                poses.append([*list(pose.position),pose.rotation.w,pose.rotation.x,pose.rotation.y,pose.rotation.z])
            rgb,depth,masks,poses=[np.stack(a) for a in (rgb,depth,masks,poses)]
            assert np.count_nonzero(np.isin(masks[0],[101,102]))==0
            assert (masks[0]==201).sum()>100 and (masks[0]==202).sum()>100
            if first:
                seen[side]=(depth.copy(),masks.copy(),poses.copy())
            else:
                d0,m0,p0=seen[side]
                assert np.array_equal(depth,d0) and np.array_equal(masks,m0) and np.allclose(poses,p0,atol=1e-6)
            if side=="left":
                start_rgb=rgb[0].copy()
                Image.fromarray(start_rgb).save(clips/(layout+"_start.png"))
            else:
                assert np.array_equal(start_rgb,rgb[0])
            actual_mask=np.isin(masks[-1],[101,102])
            assert actual_mask.sum()>300 and int(np.isin(masks[-1],[102 if side=="left" else 101]).sum())==0
            best={k:int((v&actual_mask).sum()) for k,v in colors(rgb[-1]).items()}
            assert max(best.values())>actual_mask.sum()*.3
            judged[side]=max(best,key=best.get)
            reachable[side]=np.linalg.norm(poses[-1,:3]-poses[0,:3])
            np.savez_compressed(clips/(layout+"_"+side+".npz"),rgb=rgb,depth=depth.astype(np.float16),
                offline_judge=masks.astype(np.uint16),positions_quat_wxyz=poses,planned_actions=np.array(seq))
        assert abs(reachable["left"]-reachable["right"])<1e-5
        for q in ("red","blue"):
            # This deterministic diagnostic sees only current RGB and query, not ID masks or future images.
            pixels=colors(start_rgb)[q]
            pixel_votes=[int(pixels[:,:128].sum()),int(pixels[:,128:].sum())]
            assert max(pixel_votes)>min(pixel_votes)+100
            chosen=("left","right")[int(np.argmax(pixel_votes))]
            correct=next(s for s,c in judged.items() if c==q)
            rows.append(dict(layout=layout,query=q,branch_target_color=judged.copy(),
                        supervision={s:int(c==q) for s,c in judged.items()}, correct_direction=correct,
                        fixed_distance_tie_break="left", RGB_marker_choice=chosen,
                        per_half_query_colored_pixels=pixel_votes))
assert all(r["correct_direction"]==r["RGB_marker_choice"] for r in rows)
assert sum(r["correct_direction"]=="left" for r in rows)==2
receipt=dict(scope="Synthetic counterbalanced interface test only. Markers deliberately encode hidden chair colour. No model training or generalization claim.",
             layouts=2,queries_per_layout=2,actions_per_branch=19,physical_rollouts=4,judge="offline chair-mask RGB colour",
             cross_layout_identical_depth_all_frames=True,cross_layout_identical_semantic_ID_masks_all_frames=True,
             cross_layout_identical_action_pose_traces=True,initial_chair_pixels=0,
             target_color_observed_only_after_movement=True,
             distance_only_fixed_choice_correct=2, RGB_marker_rule_correct=4,
             uniform_random_expected_accuracy=.5, rows=rows)
(reports/"counterbalance.json").write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
