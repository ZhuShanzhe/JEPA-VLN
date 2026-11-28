"""Run the deployable controller in Habitat, then compare first submissions to the paired evaluation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "reactive_view_20261129"
PRIOR = HERE.parent / "new_view_20261129"
sys.path.insert(0, str(PREVIOUS))
from verify_inspection_session import receive


def houses():
    return json.loads((PRIOR/"view_design.json").read_text())["cohorts"]["blind"]


def live(house):
    import torch
    from threadpoolctl import threadpool_limits
    from refined_inspection import load_inspection

    torch.set_num_threads(2)
    inspection = load_inspection()
    previous_dir = PREVIOUS/"private_roll"/house
    current_dir = HERE/"private_live_confirm"/house
    current_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    env = dict(os.environ, MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("LD_LIBRARY_PATH",None)
    env.pop("PYTHONPATH",None)
    py = env.get("JEPA_HABITAT_PYTHON",str(HERE.parent/"simulation/habitat-env/bin/python"))
    helper = PREVIOUS/"verify_inspection_session.py"
    with tempfile.TemporaryDirectory(dir=current_dir) as tmp, threadpool_limits(limits=2):
        packet = Path(tmp)/"frame.npz"
        proc = subprocess.Popen([py,"-u",str(helper),"--house",house,"--packet",str(packet)], stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
        try:
            for path in sorted(previous_dir.glob("*.json"),key=lambda p:int(p.stem)):
                prior = json.loads(path.read_text())
                if not prior["focus"]:
                    continue
                out = current_dir/path.stem
                out.mkdir(mode=0o700,exist_ok=True)
                logs = prior["events"]
                actions = []
                def sensor(command,action):
                    t = len(actions)
                    # Only the action from the live policy is sent to Habitat.
                    obs = receive(proc,packet,command)
                    assert logs[t]["action"] == action and logs[t]["bump"] == obs["bump"]
                    assert np.allclose(logs[t]["pose"],[obs[k] for k in ("x","z","yaw")],atol=1e-5)
                    actions.append(int(action))
                    return obs
                src = sensor("source "+path.stem,0)
                original = PRIOR/"private_render_blind60"/("blind_"+house)/(path.stem+"_sensor.npz")
                with np.load(original) as data:
                    box = data["box"].tolist()
                result = inspection.run(src,box,prior["budget"]-18,prior["budget"],lambda a:sensor("act "+str(a),a),out)
                assert len(actions)-1 == result["actions_used"] <= prior["budget"]
                receipt = dict(status=result["status"],actions_used=result["actions_used"])
                (out/"decision.json").write_text(json.dumps(receipt)+"\n")
        finally:
            proc.stdin.close()
            if proc.wait(timeout=60) != 0:
                raise RuntimeError("Simulation worker exited with an error")


def verify():
    counts = Counter()
    for house in houses():
        for path in (HERE/"private_live_confirm"/house).glob("*/decision.json"):
            reference = HERE/"private_paired"/"full"/house/path.parent.name
            a = json.loads(path.read_text())
            b = json.loads((reference/"decision.json").read_text())
            assert a["actions_used"] == b["actions_used"]
            assert (a["status"]=="confirmed") == (b["step"] is not None)
            if b["step"] is not None:
                saved = path.parent/"confirmed"
                r = json.loads((saved/"evidence.json").read_text())
                assert r["step"] == b["step"] and list(r["box"]) == list(b["box"])
                for filename in ("frame.png","crop.png"):
                    with Image.open(saved/filename) as live_image, Image.open(reference/filename) as paired_image:
                        assert np.array_equal(np.asarray(live_image),np.asarray(paired_image))
                with Image.open(saved/"frame.png") as im:
                    assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r["rgb_sha256"]
                counts["exact_submission_frame_and_crop_pairs"]+=1
            else:
                assert not (path.parent/"confirmed").exists()
            counts["groups"]+=1
            counts["matched_executed_actions"]+=a["actions_used"]
    assert counts["groups"]==94
    report = {"status":"passed","houses":3,**dict(counts),
              "checks":"Frozen route head generates actions in live Habitat. All observed contacts and odometry match the prior prefix, and first confirmed step, box, full frame, crop and actual cost match the separate paired verifier run in all 94 tasks."}
    (HERE/"live_acceptance.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)


if __name__=="__main__":
    os.umask(0o077)
    p=argparse.ArgumentParser()
    p.add_argument("--house-index",type=int,choices=(0,1,2))
    args=p.parse_args()
    if args.house_index is None:
        verify()
    else:
        live(houses()[args.house_index])
