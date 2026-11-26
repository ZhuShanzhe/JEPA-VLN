"""Collect clock anchors from development houses absent from the older clock crop collection."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def main(project,assets,out,split):
    if split not in ("train","val"):
        raise ValueError("development split only")
    old=project/"jepa_reid_v20"
    sys.path[:0]=[str(old/"collect"),str(project/"jepa_reid_v21/data")]
    import collect_views as cv
    from build_frames import CLOCKS,read_rows
    cv.WORDS.update(CLOCKS)
    houses={s["house"] for s in read_rows(old/"data/manifest.jsonl") if s["split"]==split}
    rows=[]
    count=0
    for house in sorted(houses,key=lambda h:hashlib.sha256(("stage3-clock"+h).encode()).digest()):
        sim=cv.open_sim(assets,house)
        try:
            objects=cv.inventory(sim,assets).get("wall_clock",[])
            rng=np.random.default_rng(int(hashlib.sha256(house.encode()).hexdigest()[:8],16))
            found=False
            for sid,asset,center in objects[:4]:
                r=cv.sample_object(sim,house,split,sid,asset,center,"clock",out/(split+"_clock_anchor"),rng)
                rows.extend(r)
                found=found or len(r)>0
            count+=int(found)
        finally:
            sim.close()
        if count>=5:
            break
    (out/(split+"_clock_seeds.jsonl")).write_text("".join(json.dumps(r)+"\n" for r in rows))
    print("eligible clock houses",count,"anchors",len(rows))


if __name__=="__main__":
    main(*map(Path,sys.argv[1:4]),sys.argv[4])
