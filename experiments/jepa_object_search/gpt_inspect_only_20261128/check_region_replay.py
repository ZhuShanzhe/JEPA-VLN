"""Negative control: disable all close inspections in the identical physical driver, no API."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
STAGE=ROOT/"local_target_20261128/eval_family"
sys.path.insert(0,str(ROOT/"local_target_20261128/eval"))
import run_local as physical
from trigger_policy import InspectOnlyPolicy


class ExploreOnly:
    def choose(self,views,goal,regions,patches,memory):
        # Even a different advisory ID must never override the frozen regional ranking.
        return dict(region_id=regions[-1]["id"],mode="explore",patch_id=None,
                    target_evidence="none",cue="none")


def main():
    assert hashlib.sha256((STAGE/"panel.json").read_bytes()).hexdigest()=="9d3167d378f2f31fcd7df78d95e222fe150028bcfff739fea0f9334015e69674"
    panel=json.loads((STAGE/"panel.json").read_text())
    old=json.loads((STAGE/"paired_public.json").read_text())["by_house"]
    private=HERE/"private_control"
    private.mkdir(mode=0o700,exist_ok=True)
    channel,proc=physical.start_worker(ROOT,private/"worker.log")
    rows=[]
    try:
        for i,(spec,ref) in enumerate(zip(panel,old)):
            assert spec["scene"]==ref["scene"]
            shard=physical.DATA/"eval/hssdov/content"/(spec["scene"]+".json.gz")
            assert hashlib.sha256(shard.read_bytes()).hexdigest()==spec["shard_sha256"]
            with gzip.open(shard) as f:
                ep=next(e for e in json.load(f)["episodes"] if str(e["episode_id"])==str(spec["id"]))
            sim=physical.sensor_sim(ep)
            try:
                with (private/f"{i}.jsonl").open("w") as log, patch.object(physical,"pick",InspectOnlyPolicy(ExploreOnly()).pick):
                    row=physical.episode_run(sim,ep,spec,channel,log)
                assert row["steps"]==spec["budget"]==300 and row["stats"].get("local_steps",0)==0
                events=[json.loads(l) for l in (private/f"{i}.jsonl").read_text().splitlines()]
                gated=[d for d in events if d.get("gpt_called")]
                assert gated and all(d["old_selected_region_id"]==d["executed_region_id"] and
                                     d["region_key"]==d["old_selected_region_key"] for d in gated)
                values={k:row[k] for k in ("distinct_approached","quota_met","approached_half_m","approached_one_m","first_steps")}
                diff=[k for k in values if values[k]!=ref["previous"][k]]
                rows.append(dict(goal=spec["goal"],scene=spec["scene"],steps=row["steps"],
                                 region_checks=len(gated),differences_from_earlier_region=diff,**values))
                print("completed control",i,flush=True)
            finally:sim.close()
    finally:
        channel.close()
        proc.wait(timeout=60)
    output=dict(conditions="same driver and region policy, close inspection disabled",paid_requests=0,
                houses=rows,houses_identical_on_listed_metrics=sum(not r["differences_from_earlier_region"] for r in rows),
                total_distinct=sum(r["distinct_approached"] for r in rows),quota_met=sum(r["quota_met"] for r in rows))
    dst=HERE/"parity.json"
    tmp=dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(output,indent=2)+"\n")
    tmp.replace(dst)


if __name__=="__main__":
    main()
