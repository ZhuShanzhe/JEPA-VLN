"""Run the combined controller against live Habitat sensor packets and audit its first report."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from panel import AREA, BASE, GRID, HERE, ORIGINAL, ROOT, ROUTE, SOURCE, houses, panel, read

sys.path.insert(0, str(ROUTE))
from verify_inspection_session import receive


def load():
    if GRID:
        from inspection import load_layout_inspection
        return load_layout_inspection()
    from refined_inspection import load_inspection
    from area_verifier import AreaMemory
    from layout_policy import LayoutPlanner
    inspection = load_inspection()
    inspection.reroute = LayoutPlanner(inspection.reroute)
    inspection.verifier = AreaMemory
    return inspection


def run(index):
    import torch
    from threadpoolctl import threadpool_limits
    from PIL import Image
    torch.set_num_threads(2)
    house = houses()[index]
    candidates = list(panel(house))
    # Fixed equally spaced task IDs, selected without reading new results.
    tasks = [candidates[int(k)] for k in np.linspace(0,len(candidates)-1,5)]
    directory = HERE/"private_live"/house
    directory.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet")
    env.pop("LD_LIBRARY_PATH",None)
    env.pop("PYTHONPATH",None)
    py = env.get("JEPA_HABITAT_PYTHON",str(ROOT/"simulation/habitat-env/bin/python"))
    counts = Counter()
    with tempfile.TemporaryDirectory(dir=directory) as where, threadpool_limits(limits=2):
        packet = Path(where)/"obs.npz"
        proc = subprocess.Popen([py,"-u",str(ORIGINAL/"verify_inspection_session.py"),"--house",house,"--packet",str(packet)],
                                stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
        try:
            inspection = load()
            for src, original in tasks:
                motions = read(ROUTE/"private_motion"/house/(src+".json"))["events"]
                expected_dir = HERE/"private_decisions/new_new"/house/src
                expected = read(expected_dir/"decision.json")
                i = 0
                def observe(cmd, action):
                    nonlocal i
                    assert motions[i]["action"]==action, (src,i,action,motions[i]["action"])
                    obs = receive(proc,packet,cmd)
                    pose = [obs[k] for k in ("x","z","yaw")]
                    assert np.allclose(motions[i]["pose"],pose,atol=1e-5) and obs["bump"]==motions[i]["bump"]
                    i += 1
                    return obs
                source = observe("source "+src,0)
                with np.load(SOURCE/"private_render_blind60"/("blind_"+house)/(src+"_sensor.npz")) as s:
                    box = s["box"].tolist()
                dest = directory/src
                result = inspection.run(source,box,original["budget"]-18,original["budget"],
                        lambda a:observe("act "+str(a),a),dest)
                assert i-1 == result["actions_used"] == expected["actions_used"]
                assert (result["status"]=="confirmed") == (expected["step"] is not None)
                if expected["step"] is not None:
                    record = read(dest/"confirmed/evidence.json")
                    assert expected["frame_sha256"] == record["rgb_sha256"] and expected["box"] == record["box"]
                    with Image.open(dest/"confirmed/crop.png") as a, Image.open(expected_dir/"confirmed/crop.png") as b:
                        assert np.array_equal(np.asarray(a),np.asarray(b))
                    counts["matching_submissions"] += 1
                counts["groups"] += 1
                counts["actions"] += i-1
        finally:
            proc.stdin.close()
            if proc.wait(timeout=60) != 0:
                raise RuntimeError("Simulator did not exit cleanly")
    (directory/"totals.json").write_text(json.dumps(counts))


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser()
    p.add_argument("--house",type=int,choices=(0,1,2))
    index = p.parse_args().house
    if index is None:
        counts = sum((Counter(read(HERE/"private_live"/h/"totals.json")) for h in houses()),Counter())
        assert counts["groups"]==15
        (HERE/"live_acceptance.json").write_text(json.dumps(dict(houses=3,**counts),indent=2)+"\n")
    else:
        run(index)
