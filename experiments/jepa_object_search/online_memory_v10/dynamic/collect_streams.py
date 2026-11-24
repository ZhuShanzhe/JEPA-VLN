"""Generate causal Habitat histories with actual rigid-object relocations inside a running world."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import habitat_sim as hs
from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector
import magnum as mn
import numpy as np
from PIL import Image


ACTIONS = ("move_forward", "turn_left", "turn_right")
PLANS = tuple([s] * 2 + [0] * 9 + [3 - s] * 2 + [0] * 11 for s in (1, 2))
QUERIES = ("cup", "sofa")
TRAIN = ("open_table", "closed_left", "closed_right", "gallery")
TEST = ("offset_gallery", "split_rear")
CENTRES = [[-2.03, 0., -1.65], [2.03, 0., -1.65]]


def write(path, data):
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if path.exists():
        if path.read_text() != payload:
            raise RuntimeError(f"Frozen file differs: {path}")
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(payload)
        tmp.rename(path)


def configure(stage):
    sim = hs.SimulatorConfiguration()
    sim.scene_id = str(stage)
    sim.gpu_device_id, sim.enable_physics, sim.allow_sliding = -1, True, False
    sim.override_scene_light_defaults = True
    sim.scene_light_setup = hs.gfx.DEFAULT_LIGHTING_KEY
    agent = hs.agent.AgentConfiguration(height=1.45, radius=.12)
    for name, kind in (("rgb",hs.SensorType.COLOR), ("instance",hs.SensorType.SEMANTIC)):
        cam = hs.CameraSensorSpec()
        cam.uuid, cam.sensor_type, cam.position = name, kind, mn.Vector3(0, 1.25, 0)
        cam.resolution, cam.hfov = [256,256], 90
        agent.sensor_specifications.append(cam)
    agent.action_space = {k: hs.agent.ActionSpec(k, hs.agent.ActuationSpec(amount=v))
                          for k,v in zip(ACTIONS, (.25,30.,30.))}
    return hs.Configuration(sim,[agent])


def sequence(house, cue, left, move, assets, dst, audit_visibility=False, stages_dir=None, frozen_test=False, first_revisit=0):
    # Persist actor inputs and scoring truth in different files; actor never receives condition names.
    key = f"{house}.{cue}.{left}.{int(move)}"
    recipient = dst / ("train" if house in TRAIN else "test")
    recipient.mkdir(exist_ok=True, parents=True)
    prefix = recipient / (key+".visibility.json" if audit_visibility else key+".truth.json")
    if prefix.exists():
        return
    images = dst / "rgb"
    images.mkdir(exist_ok=True)
    history, decisions, labels = [], [], []
    visible_counts = []
    with hs.Simulator(configure((stages_dir or assets) / (house+".stage_config.json"))) as sim:
        templates, objects = sim.get_object_template_manager(), sim.get_rigid_object_manager()

        def add(kind, xyz, sid):
            template = templates.load_configs(str(assets / (kind+".object_config.json")))[0]
            obj = objects.add_object_by_template_id(template)
            obj.translation = mn.Vector3(*xyz)
            obj.motion_type = hs.physics.MotionType.STATIC
            obj.semantic_id = sid
            return obj

        entities = [add(kind, CENTRES[i ^ left], 101+i) for i,kind in enumerate(QUERIES)]
        if cue != "hidden":
            for side in range(2):
                associate = side ^ left ^ int(cue == "conflicting")
                add(("coffee_station","lamp")[associate], [(-.78,.78)[side],0,2.10], 301+side)
        settings = hs.NavMeshSettings()
        settings.set_defaults()
        settings.agent_radius, settings.agent_height, settings.include_static_objects = .12, 1.45, True
        if not sim.recompute_navmesh(sim.pathfinder,settings):
            raise RuntimeError("Initial navmesh")
        actor = sim.get_agent(0)
        state = hs.AgentState()
        state.position = sim.pathfinder.snap_point(np.asarray([0,0,3.75],np.float32))
        state.rotation = quat_from_angle_axis(0.,np.asarray([0.,1.,0.]))
        actor.set_state(state)
        origin = actor.get_state().position.copy()
        step = 0
        known = set()

        def current(rgb=True):
            nonlocal step
            st = actor.get_state()
            x,_,z = st.position
            hx,_,hz = quat_rotate_vector(st.rotation,np.array([0.,0.,-1.]))
            hp = [-float(hz),-float(hx)]
            coord = [-float(z),-float(x)]
            obs = sim.get_sensor_observations()
            frame = np.ascontiguousarray(obs["rgb"][:,:,:3])
            if rgb:
                digest = hashlib.sha256(frame).hexdigest()
                if digest not in known and not (images/(digest+".png")).exists():
                    Image.fromarray(frame).save(images/(digest+".png"),compress_level=1)
                known.add(digest)
            else:
                digest = None
            return digest, coord, hp, obs["instance"], st.position.copy()

        def emit(action, collision):
            digest,coord,hp,seg,_ = current()
            history.append([step,action,coord,hp,bool(collision),digest])
            if audit_visibility:
                visible_counts.append([int(np.count_nonzero(seg==101+j)) for j in range(2)])

        def advance(action):
            nonlocal step
            collision = actor.act(ACTIONS[action])
            step += 1
            emit(action,collision)

        def check_origin():
            # Every return to the decision point is made using actions, never by teleport.
            _,_,h,_,p = current(False)
            if np.linalg.norm(p-origin)>.031 or np.linalg.norm(np.array(h)-[1,0])>.001:
                raise RuntimeError(f"Return drift {house} {p} {h}")

        def score_routes():
            saved = actor.get_state()
            gains,travel = [],[]
            for plan in PLANS:
                actor.set_state(saved)
                gained = [False,False]
                length = 0.
                probe = []
                for action in plan:
                    before = actor.get_state().position.copy()
                    actor.act(ACTIONS[action])
                    _,_,_,seg,pos = current(False)
                    length += float(np.linalg.norm(pos-before))
                    for j,obj in enumerate(entities):
                        delta = np.asarray(obj.translation)-pos
                        distance = np.linalg.norm(delta[[0,2]])
                        pixels = np.count_nonzero(seg==101+j)
                        if len(probe)<48:
                            probe.append([j,round(float(distance),3),int(pixels)])
                        if distance<=2. and pixels>=64:
                            gained[j]=True
                if not any(gained):
                    print("unseen oracle", key, probe[-2:], "pose",pos,"ids",np.unique(seg),"target", [list(o.translation) for o in entities],flush=True)
                gains.append(gained)
                travel.append(round(length,4))
            actor.set_state(saved)
            return np.transpose(gains).astype(int).tolist(),travel

        def mark(phase):
            check_origin()
            # The scoring fork is restored to the precise saved state; its frames never enter history.
            before = current()[0]
            gt,dist = score_routes()
            if current()[0]!=before or any(sum(row)!=1 for row in gt) or min(dist)<4.95:
                raise RuntimeError(f"Unusable scoring fork: {key} {phase} {gt} {dist}")
            decisions.append(dict(t=step))
            labels.append(dict(phase=phase,t=step,query_routes=gt,travel_m=dist))

        def visit(side):
            for a in PLANS[side]:
                advance(a)
            # Reverse the outward path with forward-only motion and two half turns.
            for a in [1]*6 + [{0:0,1:2,2:1}[u] for u in reversed(PLANS[side])] + [1]*6:
                advance(a)
            check_origin()

        emit(-1,False)
        for _ in range(12):
            advance(1)
        mark("unseen")
        visit(0)
        mark("left_observed")
        visit(1)
        mark("both_observed")
        before = current()[0]
        if move:
            p0,p1 = entities[0].translation,entities[1].translation
            for obj,dest in zip(entities,(p1,p0)):
                obj.motion_type = hs.physics.MotionType.KINEMATIC
                obj.translation = dest
                obj.motion_type = hs.physics.MotionType.STATIC
            if not sim.recompute_navmesh(sim.pathfinder,settings):
                raise RuntimeError("Moved navmesh")
        after = current()[0]
        if before!=after and (house in TEST or frozen_test):
            raise RuntimeError(f"Event visible from decision point: {key}")
        mark("unobserved_event")
        for _ in range(48):
            advance(1)
        mark("old_evidence")
        visit(first_revisit)
        mark("revisited_left" if first_revisit==0 else "revisited_right")
        visit(1-first_revisit)
        mark("revisited_both")
    write(recipient/(key+".actor.json"),dict(schema=["step","action","xz","heading_xz","collided","rgb_id"],
        history=history,checkpoints=decisions,actions=PLANS))
    write(recipient/(key+".truth.json"),dict(house=house,cue=cue,initial_left=left,relocated=move,
        same_origin_rgb_at_event=before==after,queries=QUERIES,labels=labels))
    if audit_visibility:
        write(prefix,visible_counts)
    print("recorded",key,len(history),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("assets",type=Path)
    ap.add_argument("output",type=Path)
    ap.add_argument("house",choices=TRAIN+TEST)
    ap.add_argument("--audit-visibility",action="store_true")
    a = ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    for cue in ("hidden","congruent","conflicting"):
        for left in range(2):
            for move in (False,True):
                sequence(a.house,cue,left,move,a.assets,a.output,a.audit_visibility)


if __name__=="__main__":
    main()
