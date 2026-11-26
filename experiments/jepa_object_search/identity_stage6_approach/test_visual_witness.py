"""Check the two-view fallback, including the cases where reports must be withheld."""
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from visual_witness import VisualWitnessMixin


class NoMappedPair:
    def _init_report_memory(self):
        self.base = (None,"identity_fewer_than_two_shared_timestamped_anchors")

    def _support(self,fid,ref,obs,boxes):
        return self.base

    def _rival_claims(self,node,box,obs,boxes):
        other,_ = self._support(2,REF,obs,boxes)
        return other is not None and tuple(other)==tuple(box)


class TestAgent(VisualWitnessMixin,NoMappedPair):
    def __init__(self,image,box):
        self._init_report_memory()
        self._trusted_pixels = {(1,REF.roi):(image,box)}


REF = SimpleNamespace(roi=(.4,.4,.6,.6))
BOX = np.array([100,92,188,180])


def face(serial):
    rng = np.random.default_rng(serial)
    image = np.full((88,88,3),218,np.uint8)
    cv2.circle(image,(44,44),40,(42,42,42),2)
    for x,y in rng.integers(17,71,size=(60,2)):
        shade = int(rng.integers(15,140))
        cv2.putText(image,str(int(rng.integers(10))),(int(x)-3,int(y)+3),cv2.FONT_HERSHEY_SIMPLEX,
                    .24,(shade,shade,shade),1,cv2.LINE_AA)
    cv2.line(image,(44,44),(25,25),(30,30,30),2)
    cv2.line(image,(44,44),(67,37),(30,30,30),2)
    return image


def scene(serial,background=10):
    rng=np.random.default_rng(background)
    im=cv2.GaussianBlur(rng.integers(0,255,(320,320,3),dtype=np.uint8),(5,5),0)
    a,b,c,d=BOX
    im[b:d,a:c]=face(serial)
    return im


def step_view(image,step,shift):
    frame=cv2.warpAffine(image,np.float32([[1,0,shift],[0,1,0]]),(320,320),
                         borderMode=cv2.BORDER_REFLECT_101)
    box=(BOX+np.array([shift,0,shift,0])).tolist()
    return SimpleNamespace(step_count=step,rgb=frame),[box]


class Cases(unittest.TestCase):
    def setUp(self):
        self.src=scene(1)
        self.agent=TestAgent(self.src,BOX)
        mock=patch.dict(sys.modules,{"appearance":SimpleNamespace(_rgb=np.asarray,WORK_WIDTH=320)})
        mock.start()
        self.addCleanup(mock.stop)

    def query(self,source,step,shift):
        obs,boxes=step_view(source,step,shift)
        return self.agent._support(1,REF,obs,boxes)

    def test_two_reobservations(self):
        self.assertIsNone(self.query(self.src,4,5)[0])
        self.assertEqual(self.query(self.src,5,25)[1],"multiview_visual_witness")
        self.assertEqual(self.query(self.src,5,25)[1],"multiview_visual_witness")

    def test_one_frame_repeated_or_noncontiguous(self):
        for t,x in [(2,10),(2,10),(3,10),(5,20)]:
            self.assertIsNone(self.query(self.src,t,x)[0])

    def test_other_clock_on_same_wall(self):
        for t,x in enumerate((0,8,15)):
            self.assertIsNone(self.query(scene(345),t,x)[0])

    def test_identical_clock_different_surroundings(self):
        for t,x in enumerate((0,8,15)):
            self.assertIsNone(self.query(scene(1,background=997),t,x)[0])

    def test_two_plausible_proposals(self):
        current=self.src.copy()
        a,b,c,d=BOX
        current[b:d,a+110:c+110]=self.src[b:d,a:c]
        for t,x in enumerate((2,6,10)):
            obs,boxes=step_view(current,t,x)
            boxes.append((np.asarray(boxes[0])+[110,0,110,0]).tolist())
            self.assertIsNone(self.agent._support(1,REF,obs,boxes)[0])

    def test_prior_negative_and_untrusted(self):
        obs,boxes=step_view(self.src,3,15)
        for prior in ("identity_different","identity_uncertain","unrecognized_error"):
            self.agent.base=(None,prior)
            self.assertEqual(self.agent._support(1,REF,obs,boxes),self.agent.base)
        self.agent.base=(None,"identity_missing_geometry")
        self.agent._trusted_pixels.clear()
        self.assertIsNone(self.agent._support(1,REF,obs,boxes)[0])

    def test_single_view_possible_rival_veto(self):
        self.agent._trusted_pixels[(2,REF.roi)]=(self.src,BOX)
        obs,boxes=step_view(self.src,1,10)
        self.assertTrue(self.agent._rival_claims(None,boxes[0],obs,boxes))
        self.assertIsNone(self.agent._rival_review_box)
        self.assertFalse(self.agent._visual_history)


def replay(folder):
    """Private imagery is supplied separately. Input boxes use the detector's 320px grid."""
    def rgb(filename):
        return cv2.cvtColor(cv2.imread(str(folder/filename)),cv2.COLOR_BGR2RGB)
    img=rgb("clock_12.png")
    agent=TestAgent(img,np.array([309,189,334,223])*img.shape[1]/518)
    rows=[]
    with patch.dict(sys.modules,{"appearance":SimpleNamespace(_rgb=np.asarray,WORK_WIDTH=320)}):
        for i in range(137,141):
            obs=SimpleNamespace(step_count=i,rgb=rgb(f"JEPA_{i}.png"))
            boxes=json.loads((folder/f"JEPA_{i}_boxes.json").read_text())
            box,reason=agent._support(1,REF,obs,boxes)
            rows.append(dict(step=i,report_evidence=box is not None,reason=reason))
    return rows


def independent_houses(folder):
    """Two independent query views of the same registered object or of other same-class objects."""
    def lines(name):
        return [json.loads(line) for line in (folder/name).read_text().splitlines()]
    detections={row["name"]:row["boxes"] for row in lines("dino.jsonl")}
    groups=defaultdict(list)
    for case in lines("cases.jsonl"):
        groups[(case["source"],case["kind"])].append(case)
    @lru_cache(maxsize=64)
    def rgb(name):
        return cv2.cvtColor(cv2.imread(str(folder/"observations"/(name+".png"))),cv2.COLOR_BGR2RGB)
    def middle_label(name,box):
        with np.load(folder/"observations"/(name+".npz"),allow_pickle=False) as arr:
            mask=arr["semantic"]
        x,y=np.floor((np.asarray(box[:2])+box[2:])/2).astype(int)
        return int(mask[y,x])
    totals=defaultdict(Counter)
    houses=set()
    with patch.dict(sys.modules,{"appearance":SimpleNamespace(_rgb=np.asarray,WORK_WIDTH=320)}):
        for (source,kind),cases in sorted(groups.items()):
            # Preserve frozen manifest order. Use at most two distinct queries.
            queries=list(dict.fromkeys(r["query"] for r in cases))[:2]
            counts=totals[kind]
            counts["groups"]+=1
            if len(queries)!=2:
                counts["insufficient_query_views"]+=1
                continue
            if not detections[source]:
                counts["no_source_box"]+=1
                continue
            image=rgb(source)
            center=np.array(image.shape[1::-1])/2
            refbox=min(detections[source],
                       key=lambda b:np.linalg.norm((np.asarray(b[:2])+b[2:])/2-center))
            agent=TestAgent(image,refbox)
            box=None
            for step,name in enumerate(queries):
                target=rgb(name)
                boxes=(np.asarray(detections[name])*320/target.shape[1]).tolist()
                obs=SimpleNamespace(rgb=target,step_count=step)
                box,_=agent._support(1,REF,obs,boxes)
            # Reveal simulator IDs for grading after both predictions.
            uid=int(cases[0]["uid"].rsplit(":",1)[1])
            if middle_label(source,refbox)!=uid:
                counts["source_center_not_intended"]+=1
                continue
            counts["scored"]+=1
            houses.add(cases[0]["house"])
            if not all(detections[k] for k in queries):
                counts["includes_empty_query_detection"]+=1
            if box is not None:
                box=np.asarray(box)*target.shape[1]/320
                right=middle_label(queries[-1],box)==uid
                counts["accepted_right" if right else "accepted_wrong"]+=1
            else:
                counts["withheld"]+=1
    return {"scope":"Detector-centred source and first two distinct frozen query views; injected as consecutive observations for component testing. Geometry gate is assumed to abstain. Source and accepted box centre ID audited after inference. No navigation actions.",
            "houses_scored":len(houses),"by_kind":dict(totals)}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--private-clock",type=Path)
    parser.add_argument("--private-hssd",type=Path)
    args,left=parser.parse_known_args()
    if args.private_clock:
        print(json.dumps(replay(args.private_clock)))
    if args.private_hssd:
        print(json.dumps(independent_houses(args.private_hssd)))
    unittest.main(argv=[sys.argv[0],*left])
