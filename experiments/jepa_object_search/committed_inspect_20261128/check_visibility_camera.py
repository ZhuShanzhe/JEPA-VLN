"""Compare RGB from the offline instance camera setup against the unchanged physical trial camera."""
import json
import gzip
from pathlib import Path
import sys
import numpy as np
from visibility_review import HERE, ROOT, DATA, configured_sim, lines, hs, quat_from_coeffs

sys.path.insert(0,str(ROOT/"approach_search_20261128/eval4"))
from physical_panel import sensor_sim
from stopwatch.data.semantics import instance_lut


def main():
    panel = json.loads((ROOT/"local_target_20261128/eval_family/panel.json").read_text())
    comparisons = 0
    max_channel_error = 0
    largest_fraction_different = 0.
    for i, spec in enumerate(panel):
        events = lines(HERE/f"private_events/{i}.jsonl")
        steps = set()
        for e in events:
            if e["event"] == "acquired":
                steps.update(e["step"]-d for d in (4,3,1,0))
            if e["event"] == "arrived":
                steps.add(e["step"])
        if not steps:
            continue
        with gzip.open(DATA/"eval/hssdov/content"/(spec["scene"]+".json.gz")) as f:
            ep = next(r for r in json.load(f)["episodes"] if str(r["episode_id"]) == str(spec["id"]))
        poses = lines(HERE/f"private_events/{i}.poses.jsonl")
        def render(sim, semantic):
            try:
                if semantic:
                    instance_lut(sim,"hssd",DATA/"vocab")
                frames = []
                for s in sorted(steps):
                    cam = hs.AgentState()
                    cam.position = np.float32(poses[s]["xyz"])
                    cam.rotation = quat_from_coeffs(np.asarray(poses[s]["quat_xyzw"]))
                    sim.get_agent(0).set_state(cam)
                    frames.append(sim.get_sensor_observations()["rgb"][:,:,:3].copy())
                return np.asarray(frames)
            finally:
                sim.close()
        original = render(sensor_sim(ep),False)
        offline = render(configured_sim(ep,check_rgb=True),True)
        max_channel_error = max(max_channel_error,int(np.abs(original.astype(int)-offline).max()))
        largest_fraction_different = max(largest_fraction_different, float(np.any(original!=offline,axis=-1).mean(axis=(1,2)).max()))
        comparisons += len(steps)
    result = dict(rgb_frame_pairs=comparisons,max_rgb_channel_difference=max_channel_error,
                  largest_fraction_of_different_pixels=largest_fraction_different)
    print(result)
    assert comparisons and max_channel_error == 0
    (HERE/"camera_check.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
