"""Render the previously frozen navigation actions once, keeping evaluation labels separate."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'reactive_view_20261129'
CAMERAS = HERE.parent / 'new_view_20261129'
sys.path.insert(0, str(CAMERAS))
from sample_views import ACTION, ASSETS, SOURCE_H, camera, inventory, simulator


def render(index):
    os.umask(0o077)
    houses = json.loads((CAMERAS/'view_design.json').read_text())['cohorts']['blind']
    house = houses[index]
    sensor_out = HERE/'private_inputs'/house
    truth_out = HERE/'private_scoring'/house
    sensor_out.mkdir(parents=True, exist_ok=True)
    truth_out.mkdir(parents=True, exist_ok=True)
    sim = simulator(house)
    n = 0
    try:
        inventory(sim, ASSETS)
        agent = sim.get_agent(0)
        for path in sorted((ORIGINAL/'private_roll'/house).glob('*.json')):
            row = json.loads(path.read_text())
            if not row['focus']:
                continue
            n += 1
            src = row['source']
            with np.load(CAMERAS/'private_render_blind60'/('blind_'+house)/f'{src}_sensor.npz') as f:
                pos, yaw, oid, cue = f['position'], float(f['yaw']), int(f['oid']), f['box']
            obs = camera(sim, pos, yaw, SOURCE_H, 0.)
            rgb, depth, odom, motion, masks = [], [], [], [], []
            for step, event in enumerate(row['events']):
                if step:
                    prev = agent.state.position.copy()
                    contact = agent.act(ACTION[event['action']])
                    moved = np.linalg.norm((agent.state.position-prev)[[0,2]])
                    assert event['bump'] == bool(event['action']==1 and (contact or moved<.23))
                    obs = sim.get_sensor_observations()
                diff = agent.state.position-pos
                angle = 2*math.atan2(agent.state.rotation.y, agent.state.rotation.w)-yaw
                pose = (math.cos(yaw)*diff[0]-math.sin(yaw)*diff[2],
                        -math.sin(yaw)*diff[0]-math.cos(yaw)*diff[2], math.atan2(math.sin(angle),math.cos(angle)))
                assert np.allclose(pose,event['pose'],atol=1e-5)
                mask = obs['ids'] == oid
                assert int(mask.sum()) == event['px']
                rgb.append(obs['rgb'][...,:3].copy())
                depth.append(obs['depth'].squeeze().copy())
                odom.append(pose)
                motion.append(event['action'])
                masks.append(np.packbits(mask.reshape(-1)))
            assert len(rgb)-1 <= row['budget']
            np.savez(sensor_out/f'{src}.npz', rgb=rgb, depth=depth, odom=odom, cue=cue, action=motion,
                     budget=row['budget'])
            np.savez(truth_out/f'{src}.npz', packed=masks)
    finally:
        sim.close()
    print(f'{house}  {n} motions replayed',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('house_index',type=int)
    render(p.parse_args().house_index)
