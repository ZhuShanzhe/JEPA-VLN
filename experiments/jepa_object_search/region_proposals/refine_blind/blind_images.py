"""Prepare anonymous RGB-only visual inspection panels before opening predictions."""
import json
from pathlib import Path
import sys
import random
from PIL import Image,ImageDraw


def run(src,dest):
    dest.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in (src/"manifest.jsonl").read_bytes().splitlines() if json.loads(x)["category"]=="clock"]
    random.Random(45181).shuffle(rows)
    lookup={}
    for index,row in enumerate(rows):
        code=f"V{index+1:02d}"
        lookup[code]=row["frame_id"]
        image=Image.open(src/row["rgb"]).convert("RGB")
        assert image.size==(518,518)
        image.save(dest/(code+".png"))
        if index%2==0:
            panel=Image.new("RGB",(1036,540),"white")
        off=index%2*518
        panel.paste(image,(off,22))
        ImageDraw.Draw(panel).text((off+2,2),code,fill="black")
        if index%2 or index+1==len(rows):
            panel.save(dest/f"panel{index//2+1:02d}.jpg",quality=94)
    (dest/"lookup.json").write_text(json.dumps(lookup))
    print(f"{len(rows)} RGB frames")

if __name__=="__main__":
    run(*map(Path,sys.argv[1:]))
