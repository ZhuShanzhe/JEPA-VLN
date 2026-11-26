"""Apply a fixed RGB box refiner to another frozen frame/candidate manifest."""
import argparse
import json
from pathlib import Path
import cv2
from box_refiner import BoxRefiner


def apply(frames_file, proposals_file, checkpoint, output):
    model=BoxRefiner(checkpoint)
    frames=[json.loads(s) for s in frames_file.read_bytes().splitlines()]
    rows=[json.loads(s) for s in proposals_file.read_bytes().splitlines()]
    proposals={r["frame_id"]:r["boxes"] for r in rows}
    assert len(proposals)==len(frames)==len(rows)
    assert set(proposals)=={f["frame_id"] for f in frames}
    tmp=output.with_suffix(".tmp")
    total=0
    with tmp.open("w",encoding="utf8") as fout:
        for f in frames:
            path=frames_file.parent/f["rgb"]
            image=cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            data=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            before=proposals[f["frame_id"]]
            after=model.refine(data,before, f.get("category"))
            assert [b["score"] for b in before]==[b["score"] for b in after]
            assert all(b["bbox"]==b["xyxy"] for b in after)
            total+=len(after)
            fout.write(json.dumps(dict(frame_id=f["frame_id"],boxes=after),allow_nan=False)+"\n")
    tmp.replace(output)
    print(f"{len(frames)} aligned RGB frames, {total} proposals refined")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for arg in ("frames","proposals","checkpoint","output"):
        parser.add_argument(arg,type=Path)
    args=parser.parse_args();apply(args.frames,args.proposals,args.checkpoint,args.output)
