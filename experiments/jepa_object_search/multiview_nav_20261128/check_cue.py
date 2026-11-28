"""Replay all four RGB-only requests with the live detector; prohibit any paid call."""
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from cue_service import CAMERAS, CueSelector, HERE, OLD


def main():
    catalog=json.loads((OLD.parent/"inputs.json").read_text())
    groups={}
    for row in catalog:
        groups.setdefault(row["group"],{})[row["view"]]=row
    assert len(groups)==4
    evidence=[]
    with tempfile.TemporaryDirectory(dir=HERE) as d:
        svc=CueSelector(d,allow_new=False)
        try:
            assert len(svc.old)==4
            for index,records in enumerate(groups.values()):
                frames=[]
                for name in CAMERAS:
                    with Image.open(OLD.parent/(records[name]["id"]+".png")) as im:
                        frames.append(np.asarray(im.convert("RGB")))
                result=svc.select(frames,records["center"]["goal"])
                receipt=json.loads((OLD/f"{index}.response.json").read_text())
                assert svc.last_receipt["event"]=="prior_cache" and svc.last_receipt["request_hash"]==receipt["request_hash"]
                assert (None if result is None else result["candidate_id"])==receipt["choice"]
                if result is not None:
                    assert result["candidate_id"][0]=="RCL"[result["view_index"]]
                evidence.append({"event":index+1,"choice":receipt["choice"],
                                 "view_index":None if result is None else result["view_index"],
                                 "candidates":svc.last_receipt["candidates"],"byte_exact_request":True})
            assert not (Path(d)/"cue_calls.jsonl").exists()
            # Isolate the geometry contract at common native aspect ratio, injecting a fixed
            # detector output and a selection; the 12 HSSD views above use the real detector.
            svc._propose=lambda paths,goal: [[{"xyxy":[64.,48.,320.,240.]}],[],[]]
            svc._choose=lambda h,wire,offered: ("fixture","R1")
            native=svc.select([np.zeros((480,640,3),dtype=np.uint8)]*3,"clock")
            assert native["bbox_norm"]==[.1,.1,.5,.5] and native["view_index"]==0
        finally:
            svc.close()
    report=dict(fresh_dino_views=12, prior_cache_hits=4, new_paid_calls=0,
                separate_geometry_fixture_640x480="passed", cases=evidence)
    (HERE/"cue_preflight.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
