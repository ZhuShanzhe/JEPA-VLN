"""Physically replay a mined HSSD training pair and produce decision-only report examples."""
import argparse
import json
import math
from pathlib import Path
import tempfile

import habitat_sim as hs
from habitat_sim.utils.common import quat_from_coeffs
import magnum as mn
import numpy as np
from PIL import Image

from mine_hssd_history import ROOT, BASE, sha

ACTIONS=("move_forward","turn_left","turn_right")


def simulation(scene, raw):
    cfg=hs.SimulatorConfiguration()
    cfg.scene_dataset_config_file=str(raw/"hssd-hab.scene_dataset_config.json")
    cfg.scene_id=scene
    cfg.enable_physics=True
    cfg.allow_sliding=False
    agent=hs.agent.AgentConfiguration(height=1.4,radius=.18)
    for key,kind in (("rgb",hs.SensorType.COLOR),("depth",hs.SensorType.DEPTH),("semantic",hs.SensorType.SEMANTIC)):
        c=hs.CameraSensorSpec()
        c.uuid=key
        c.sensor_type=kind
        c.resolution=[256,256]
        c.position=mn.Vector3(0,1.25,0)
        c.hfov=mn.Deg(90)
        agent.sensor_specifications.append(c)
    agent.action_space={a:hs.agent.ActionSpec(a,hs.agent.ActuationSpec(amount=x)) for a,x in zip(ACTIONS,(.25,30.,30.))}
    return hs.Simulator(hs.Configuration(cfg,[agent]))


def dead_reckon(seq):
    x=y=yaw=0.
    for code in seq:
        if code==0:
            x+=.25*math.cos(yaw)
            y+=.25*math.sin(yaw)
        else:
            yaw+=math.pi/6 if code==1 else -math.pi/6
    return [x,y,yaw]


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2))


def build(pair,stage,scene_root):
    splitfile=ROOT/"simulation/train_dev_v1.json"
    split=json.loads(splitfile.read_text())
    scene=pair["scene"]
    assert scene in split["train"] and scene not in split["dev"]
    src=BASE/pair["source"]
    m=json.loads((src/"manifest.json").read_text())
    assert m["split"]=="train" and m["split_sha256"]==sha(splitfile)
    judge=json.loads((src/"teacher/labels.json").read_text())
    target_ids=sorted(map(int,judge["instance_map"]))
    n=m["frontier_count"]
    common=stage/"decision"
    private=stage/"judge"
    common.mkdir(); private.mkdir()
    frozen=m["frozen_pose_xyz_xyzw"]
    with np.load(src/"context.npz") as ctx:
        xy,plan=ctx["frontiers_xy"],ctx["planned_actions"]
    cands=[dict(id=f"c{i}",relative_forward_left_m=xy[i].tolist(),planned_actions=plan[i].tolist()) for i in range(n)]
    witness={}
    histories=[]
    with simulation(scene,scene_root) as sim:
        objects=sim.get_rigid_object_manager().get_objects_by_handle_substring("")
        for sid,(handle,obj) in enumerate(sorted(objects.items()),1):
            obj.semantic_id=sid
            if sid in target_ids:
                assert judge["instance_map"][str(sid)]["handle"]==handle
        del obj
        nav=hs.NavMeshSettings();nav.set_defaults()
        nav.agent_height=1.4;nav.agent_radius=.18;nav.include_static_objects=True
        assert sim.recompute_navmesh(sim.pathfinder,nav)
        sim.seed(m["seed"])
        for j,w in enumerate(pair["witnesses"]):
            name=f"h{j}"
            i=w["branch"]
            expected=w["instance"]
            k=w["witness_step"]
            with np.load(src/f"branch{i:03d}.npz") as record:
                seq=record["actions"][:k].tolist()
                assert not record["collided"][:k].any()
            agent=sim.get_agent(0)
            initial=hs.AgentState()
            initial.position=np.array(frozen[:3],dtype=np.float32)
            initial.rotation=quat_from_coeffs(np.array(frozen[3:],dtype=np.float32))
            agent.set_state(initial)
            for code in seq:
                f=sim.step(ACTIONS[code])
                assert not f.get("collided",False), "outbound_collision"
            pixels=int((f["semantic"]==expected).sum())
            assert pixels==w["peak_pixels"],("offline replay changed semantic target count",pixels,w["peak_pixels"])
            report_image=np.asarray(f["rgb"][:,:,:3])
            assert report_image.std()>2
            Image.fromarray(report_image).save(common/f"{name}_report.png")
            estimate=dead_reckon(seq)
            pos=agent.get_state().position-np.array(initial.position)
            judge_report_error=float(np.linalg.norm(np.array(estimate[:2])-np.array([-pos[2],-pos[0]])))
            assert judge_report_error < .04, ("odometry",judge_report_error)
            inverse={0:0,1:2,2:1}
            back=[1]*6+[inverse[c] for c in seq[::-1]]+[1]*6
            for code in back:
                f=sim.step(ACTIONS[code])
                assert not f.get("collided",False),"return collision"
            state=agent.get_state()
            displacement=float(np.linalg.norm(state.position-initial.position))
            agreement=float(abs((initial.rotation.inverse()*state.rotation).w))
            assert displacement<.04 and agreement>.999
            # Observe again at the decision point using a real twelve-view turn,
            # after the reported branch and a physically executed return.
            frames=[]
            for step in range(12):
                img=f["rgb"][:,:,:3]
                name_img=f"{name}_start{step:02d}.png"
                Image.fromarray(img).save(common/name_img)
                frames.append(name_img)
                f=sim.step("turn_left")
            decision={"target_synset":pair["target"],"observed_rgb":frames,"candidates":cands,
                      "reports":[{"observed_rgb":f"{name}_report.png","odometry_forward_left_yaw":estimate}]}
            save_json(common/f"{name}.json",decision)
            labels=[]
            for b in range(n):
                with np.load(src/"teacher"/f"branch{b:03d}.npz") as gt:
                    ids=gt["instance_ids"]
                    visible=set(map(int,ids[gt["target_pixel_counts"].max(0)>=64]))
                labels.append(int(bool(visible-{expected})))
            histories.append(labels)
            witness[name]={"used_candidate":f"c{i}","credited_instance":expected,"measured_pixels":pixels,
                           "report_step":k,"physical_prior":seq,"return_actions":back,
                           "revisit_position_error_m":displacement,"revisit_rotation_agreement":agreement,
                           "report_dead_reckoning_error_m":judge_report_error,"utility":labels}
    b0,b1=[w["branch"] for w in pair["witnesses"]]
    assert [histories[0][b0],histories[0][b1],histories[1][b0],histories[1][b1]]==[0,1,1,0]
    save_json(private/"witnesses_and_labels.json",witness)
    return dict(status="pass",scene=scene,query=pair["target"],train_only=True,source=pair["source"],
                start_count=1,histories=2,frontier_count=n,paired_branches=[b0,b1],
                policy_fields=["current post-return RGB turn history","target synset","nominal frontier + actions",
                               "prior report RGB + action dead-reckoned forward/left/yaw"],
                semantic_input=False,simulator_true_pose_input=False,true_report_credit=False,
                offline_label="constructed prior credit for one matching instance; any other matching ID visible on branch at 64 pixels",
                previous_and_new_record_scope="new physical report-and-return observations; existing same-start future branch outcomes",
                split_sha256=sha(splitfile),returned_position_error_m=[v["revisit_position_error_m"] for v in witness.values()],
                decision_input_sha256={f.name:sha(f) for f in sorted(common.iterdir())})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mined",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--scene-root",type=Path,default=ROOT/"simulation"/"hssd",
                   help="HSSD Habitat asset directory with hssd-hab.scene_dataset_config.json")
    p.add_argument("--pair-index",type=int,default=0,help="Index in the predetermined training candidate list")
    args=p.parse_args()
    if args.output.exists():p.error("Output must be new")
    catalog=json.loads(args.mined.read_text())
    assert catalog["split_sha256"]==sha(ROOT/"simulation/train_dev_v1.json")
    selected=catalog["pairs"][args.pair_index]
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        stage=Path(temp)/"record"
        stage.mkdir()
        receipt=build(selected,stage,args.scene_root)
        receipt.update(pair_index=args.pair_index,miner_receipt_sha256=sha(args.mined))
        receipt["collector_sha256"]=sha(Path(__file__).resolve())
        receipt["hssd_scene_config_sha256"]=sha(args.scene_root/"hssd-hab.scene_dataset_config.json")
        save_json(stage/"receipt.json",receipt)
        stage.rename(args.output)
    print(json.dumps({k:v for k,v in receipt.items() if k!="decision_input_sha256"}))


if __name__=="__main__":
    main()
