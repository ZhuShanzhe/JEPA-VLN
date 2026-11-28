"""Offline authority check: force exploration in a development scene and compare every physical pose."""
import gzip
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"local_target_20261128/eval"))
import run_local as phys
import physical_panel as former
from trigger_policy import InspectOnlyPolicy


class Explore:
    def choose(self, views, goal, regions, patches, memory):
        return dict(mode="explore",region_id=regions[-1]["id"],patch_id=None,cue="none",target_evidence="none")


def collect_poses(owner, run):
    original=owner.policy_pose
    poses=[]
    def record(sim):
        p=original(sim)
        poses.append(p)
        return p
    with patch.object(owner,"policy_pose",record):
        run()
    return np.asarray(poses)


def main():
    spec=json.loads((ROOT/"approach_search_20261128/eval3/panel.json").read_text())[0]
    spec["budget"]=65
    ep=next(e for e in json.load(gzip.open(phys.DATA/"eval/hssdov/content"/(spec["scene"]+".json.gz")))["episodes"]
            if str(e["episode_id"])==str(spec["id"]))
    sim=phys.sensor_sim(ep)
    worker=former.Worker()
    try:
        old=collect_poses(former,lambda:former.rollout(sim,ep,spec,worker,"new"))
    finally:
        worker.close()
        sim.close()
    sim=phys.sensor_sim(ep)
    channel,process=phys.start_worker(ROOT,HERE/"development_worker.log")
    try:
        log=io.StringIO()
        with patch.object(phys,"pick",InspectOnlyPolicy(Explore()).pick):
            current=collect_poses(phys,lambda:phys.episode_run(sim,ep,spec,channel,log))
    finally:
        channel.close()
        process.wait(timeout=60)
        sim.close()
    assert old.shape==current.shape==(66,3)
    assert np.allclose(current,old,rtol=0,atol=1e-6),np.max(np.abs(current-old))
    events=[json.loads(l) for l in log.getvalue().splitlines()]
    checked=[e for e in events if e.get("gpt_called")]
    assert checked and all(e["old_selected_region_id"]==e["executed_region_id"] for e in checked)
    different=sum(e["gpt_region"]!=e["executed_region_id"] for e in checked)
    assert different>0
    summary=dict(development_house_only=True,compared_poses=len(old),physical_steps=spec["budget"],
                 max_absolute_pose_difference=float(np.max(np.abs(current-old))),
                 compared_selections=len(checked),adversarial_region_choices_overridden=different)
    (HERE/"region_contract.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
