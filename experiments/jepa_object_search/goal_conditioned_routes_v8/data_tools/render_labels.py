"""HSSD RGB, depth and instance sensors for the physically replayed route teacher."""
import os
from pathlib import Path

import habitat_sim as hs
import magnum as mn


def simulation(house):
    dataset = Path(os.environ["HSSD_SCENE_DATASET_CONFIG"])
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    config = hs.SimulatorConfiguration()
    config.scene_dataset_config_file = str(dataset)
    config.scene_id = house
    config.enable_physics = True
    config.allow_sliding = False
    agent = hs.agent.AgentConfiguration(height=1.4, radius=.18)
    for key, kind in (("rgb", hs.SensorType.COLOR), ("depth", hs.SensorType.DEPTH), ("semantic", hs.SensorType.SEMANTIC)):
        camera = hs.CameraSensorSpec()
        camera.uuid, camera.sensor_type, camera.resolution = key, kind, [256, 256]
        camera.position, camera.hfov = mn.Vector3(0, 1.25, 0), mn.Deg(90)
        agent.sensor_specifications.append(camera)
    commands = dict(move_forward=.25, turn_left=30., turn_right=30.)
    agent.action_space = {name: hs.agent.ActionSpec(name, hs.agent.ActuationSpec(amount=amount))
                          for name, amount in commands.items()}
    sim = hs.Simulator(hs.Configuration(config, [agent]))
    nav = hs.NavMeshSettings()
    nav.set_defaults()
    nav.agent_height, nav.agent_radius, nav.include_static_objects = 1.4, .18, True
    if not sim.recompute_navmesh(sim.pathfinder, nav):
        sim.close()
        raise ValueError("The scene did not yield a navigable mesh")
    return sim
