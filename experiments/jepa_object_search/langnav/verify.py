"""HSSD simulation helpers used by the branch-observation check."""
from pathlib import Path

import habitat_sim as hs
import magnum as mn
import numpy as np
from habitat_sim.utils.common import quat_from_coeffs

SIM = Path(__file__).resolve().parents[1] / 'simulation'


def scene_sim(scene):
    cfg=hs.SimulatorConfiguration()
    cfg.scene_dataset_config_file=str(SIM/'hssd-render/hssd-hab.scene_dataset_config.json')
    cfg.scene_id=scene
    cfg.enable_physics=True
    cfg.scene_light_setup=hs.gfx.DEFAULT_LIGHTING_KEY
    cfg.override_scene_light_defaults=True
    if hasattr(cfg,'pbr_image_based_lighting'):
        cfg.pbr_image_based_lighting=True
    nav=hs.nav.NavMeshSettings()
    nav.set_defaults()
    nav.agent_radius=.1
    nav.agent_height=1.5
    nav.include_static_objects=True
    cfg.navmesh_settings=nav
    a=hs.agent.AgentConfiguration(height=1.5,radius=.1)
    for name,kind in [('rgb',hs.SensorType.COLOR),('depth',hs.SensorType.DEPTH),('semantic',hs.SensorType.SEMANTIC)]:
        spec=hs.CameraSensorSpec()
        spec.uuid=name
        spec.sensor_type=kind
        spec.resolution=[640,640]
        spec.hfov=90
        spec.position=mn.Vector3(0,1.5,0)
        a.sensor_specifications.append(spec)
    a.action_space={k:hs.agent.ActionSpec(k,hs.agent.ActuationSpec(amount=v)) for k,v in [('move_forward',.25),('turn_left',30.),('turn_right',30.)]}
    return hs.Simulator(hs.Configuration(cfg,[a]))


def setpose(sim,pos,q):
    st=hs.AgentState()
    st.position=np.asarray(pos,dtype=np.float32)
    st.rotation=quat_from_coeffs(np.asarray(q,dtype=np.float32))
    sim.get_agent(0).set_state(st)
    return sim.get_sensor_observations()
