"""Offline probes for the arrival-conditioned image association."""
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from local_match import choose_arrival_box, match_prepared, prepare_reference
from test_visual_witness import BOX, TestAgent, scene, step_view


class ArrivalTests(unittest.TestCase):
    def test_later_view_of_registered_instance(self):
        image=scene(1)
        obs,boxes=step_view(image,12,14)
        self.assertEqual(choose_arrival_box(image,BOX,obs.rgb,boxes),0)

    def test_different_object_same_scene(self):
        obs,boxes=step_view(scene(345),12,14)
        self.assertIsNone(choose_arrival_box(scene(1),BOX,obs.rgb,boxes))

    def test_same_object_unrelated_background(self):
        obs,boxes=step_view(scene(1,background=345),12,14)
        self.assertIsNone(choose_arrival_box(scene(1),BOX,obs.rgb,boxes))

    def test_ambiguous_or_missing(self):
        img=scene(1)
        self.assertIsNone(choose_arrival_box(img,BOX,img,[BOX,BOX]))
        self.assertIsNone(choose_arrival_box(img,BOX,img,[]))


def validate(folder):
    def rows(name):
        return [json.loads(line) for line in (folder/name).read_text().splitlines()]
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    detections={row["name"]:row["boxes"] for row in rows("dino.jsonl")}
    groups=defaultdict(list)
    for case in rows("cases.jsonl"):
        groups[(case["source"],case["kind"])].append(case)
    @lru_cache(maxsize=64)
    def rgb(name):
        return cv2.cvtColor(cv2.imread(str(folder/"observations"/(name+".png"))),cv2.COLOR_BGR2RGB)
    def label(name,box):
        x,y=np.floor((np.asarray(box[:2])+box[2:])/2).astype(int)
        with np.load(folder/"observations"/(name+".npz"),allow_pickle=False) as arr:
            return int(arr["semantic"][y,x])
    counts=defaultdict(Counter)
    by_category=defaultdict(lambda: defaultdict(Counter))
    houses=set()
    for (source,kind),cases in sorted(groups.items()):
        queries=list(dict.fromkeys(c["query"] for c in cases))[:2]
        count=counts[kind]
        count["candidate_groups"]+=1
        if len(queries)<2 or not detections[source]:
            count["excluded_incomplete"]+=1
            continue
        refimg=rgb(source)
        srcbox=min(detections[source],key=lambda b:np.linalg.norm(
            (np.asarray(b[:2])+b[2:])/2-np.array(refimg.shape[1::-1])/2))
        prepared=prepare_reference(refimg,srcbox)
        legacy=TestAgent(refimg,srcbox)
        for step,q in enumerate(queries):
            curr=rgb(q)
            boxes=detections[q]
            scaled=(np.asarray(boxes)*320/curr.shape[1]).tolist()
            prior,_=legacy._support(1,SimpleNamespace(roi=(.4,.4,.6,.6)),
                                    SimpleNamespace(step_count=step,rgb=curr),scaled)
        i=match_prepared(prepared,curr,boxes)
        answer=boxes[i] if i is not None else None
        union=np.asarray(prior)*curr.shape[1]/320 if prior is not None else answer
        uid=int(cases[0]["uid"].rsplit(":",1)[1])
        if label(source,srcbox)!=uid:
            count["excluded_incorrect_source"]+=1
            continue
        count["scored"]+=1
        category=by_category[cases[0]["category"]][kind]
        category["scored"]+=1
        houses.add(cases[0]["house"])
        for method,proposal in (("arrival",answer),("prior",None if prior is None else np.asarray(prior)*curr.shape[1]/320),("combined",union)):
            key="withheld" if proposal is None else ("correct" if label(q,proposal)==uid else "wrong")
            count[method+"_"+key]+=1
            category[method+"_"+key]+=1
    return dict(scope="Arrival/request injected, source box and the last of two distinct query views. Stage 6 checks both; Stage 7 evaluates only the last current view. No navigation, rival owner or episode score.",
        independent_houses=len(houses), thresholds="Locked on synthetic warp, object swap and background swap before this HSSD test; no household selection by outcome.",
        module_sha256=digest(Path(__file__).with_name("local_match.py")),fixtures_sha256=digest(folder/"cases.jsonl"),
        detections_sha256=digest(folder/"dino.jsonl"),groups={k:dict(v) for k,v in counts.items()},
        by_category={cat:dict(groups) for cat,groups in by_category.items()})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--hssd",type=Path)
    p.add_argument("--out",type=Path)
    args,rest=p.parse_known_args()
    if args.hssd:
        result=validate(args.hssd)
        if args.out:
            args.out.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result))
    else:
        unittest.main(argv=["test_local_match",*rest])
