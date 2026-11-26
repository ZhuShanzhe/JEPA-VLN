"""Run official automatic mask proposals on frozen Habitat RGB; resume from private cache."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from native_regions import NativeRegions


def run(args):
    rows = [json.loads(r) for r in args.manifest.read_bytes().splitlines()]
    if args.split:
        rows = [r for r in rows if r["split"] in args.split]
    assert len({r["frame_id"] for r in rows}) == len(rows)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    cache = args.output.with_suffix(".cache")
    cache.mkdir(exist_ok=True)
    model = NativeRegions(args.project)
    begin = time.monotonic()
    temp = args.output.with_suffix(".partial")
    with temp.open("w") as stream:
        for i,entry in enumerate(rows,1):
            file = args.images / entry["rgb"]
            key = hashlib.sha256(file.read_bytes()).hexdigest()
            checkpoint = cache / (key+".json")
            if checkpoint.is_file():
                result = json.loads(checkpoint.read_bytes())
            else:
                image = np.asarray(Image.open(file).convert("RGB"))
                h,w = image.shape[:2]
                result = dict(width=w,height=h,rgb_sha256=key,boxes=model.propose(image))
                result_temp = checkpoint.with_suffix(".tmp")
                result_temp.write_text(json.dumps(result))
                result_temp.replace(checkpoint)
            line = dict(frame_id=entry["frame_id"],**result)
            stream.write(json.dumps(line)+"\n")
            stream.flush()
            if i==1 or i%20==0:
                print(i,len(rows),round(time.monotonic()-begin,2),flush=True)
    temp.replace(args.output)
    print("complete",len(rows),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("project","manifest","images","output"):
        p.add_argument(k,type=Path)
    p.add_argument("--split",nargs="+")
    run(p.parse_args())
