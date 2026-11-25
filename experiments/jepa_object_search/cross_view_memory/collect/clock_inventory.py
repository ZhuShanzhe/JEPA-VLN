"""Check whether native HSSD houses contain distinct clock instances for association testing."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re


def inventory(assets, data):
    with (assets/"metadata/fpmodels-with-decomposed.csv").open() as f:
        clocks = {}
        names = Counter()
        for row in csv.DictReader(f):
            exact_synset = row["wnsynsetkey"]
            key = row["main_wnsynsetkey"] or exact_synset
            if "clock" in key.split(".")[0]:
                clocks[row["id"]]=key
                names[key]+=1
    result={}
    cohorts=json.loads((data/"sampling_protocol.json").read_text())["houses"]
    for split, houses in cohorts.items():
        counts=[]
        for house in houses:
            scene=json.loads((assets/"scenes"/f"{house}.scene_instance.json").read_text())
            count=0
            for obj in scene["object_instances"]:
                handles=re.findall("[a-fA-F0-9]{40}",obj["template_name"])
                count+=any(h in clocks for h in handles)
            counts.append(count)
        result[split]=dict(houses_with_any=sum(n>0 for n in counts),
                           houses_with_multiple=sum(n>=2 for n in counts),instance_count=sum(counts))
    out=dict(clock_asset_synsets=names,cohort=result)
    (data/"clock_inventory.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("assets",type=Path)
    p.add_argument("data",type=Path)
    a=p.parse_args()
    inventory(a.assets,a.data)
