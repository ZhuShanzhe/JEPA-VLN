"""Exercise the policies on a previously used development house before the paired test."""
import gzip
import json
import os
import sys
from run_local import HERE, ROOT, DATA, episode_run, sensor_sim, start_worker


def main():
    spec=json.loads((ROOT/"approach_search_20261128/eval3/panel.json").read_text())[0]
    ep=next(e for e in json.load(gzip.open(DATA/"eval/hssdov/content"/(spec["scene"]+".json.gz")))["episodes"]
            if str(e["episode_id"])==str(spec["id"]))
    spec["budget"]=40
    sim=sensor_sim(ep)
    channel,process=start_worker(ROOT,HERE/"development_worker.log")
    try:
        with open(os.devnull,"w") as log:
            row=episode_run(sim,ep,spec,channel,log)
        print(json.dumps(dict(steps=row["steps"],stats=row["stats"],counts=row["counts"])))
        assert row["steps"]==40
    finally:
        channel.close()
        process.wait(timeout=30)
        sim.close()


if __name__=="__main__":
    main()
