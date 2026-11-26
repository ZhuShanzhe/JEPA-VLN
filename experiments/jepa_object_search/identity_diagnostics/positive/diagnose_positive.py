"""Trace the frozen live identity check on a deterministic held-out HSSD positive panel."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rows(path):
    return [json.loads(row) for row in path.read_bytes().splitlines() if row]


def select(frames):
    by_house = defaultdict(list)
    for frame in frames:
        by_house[frame["house"]].append(frame)
    cases = []
    for house, group in sorted(by_house.items()):
        pool = []
        ordered = sorted(group, key=lambda r: (r["sequence"], r["view"], r["frame_id"]))
        for a, b in combinations(ordered, 2):
            if a["category"] != b["category"]:
                continue
            live = {obj["instance"]:obj for obj in b["boxes"]}
            for obj in a["boxes"]:
                uid = obj["instance"]
                if uid in live:
                    key = digest(["panel-20261126", a["frame_id"], b["frame_id"], uid])
                    pool.append((key, a, b, obj, live[uid]))
        cases.extend(sorted(pool, key=lambda r: r[0])[:12])
    return cases


def trace_sift(ref, cur):
    """Mirror the deployed local_correspondence checks. Caller asserts binary output parity."""
    if min(*ref.shape[:2], *cur.shape[:2]) < 12:
        return "small_crop"
    if not hasattr(cv2, "SIFT_create"):
        return "unavailable"
    sift = cv2.SIFT_create(nfeatures=160, contrastThreshold=.025)

    def features(im):
        scale = 128 / max(im.shape[:2])
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        keys, desc = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None)
        return np.array([k.pt for k in keys], np.float32), desc, np.array(im.shape[1::-1])

    a, da, size_a = features(ref)
    b, db, size_b = features(cur)
    if min(len(a), len(b)) < 6:
        return "few_keypoints"
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def good(left, right):
        return {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(left, right, k=2)
                if m.distance < .75*n.distance}

    ab, ba = good(da, db), good(db, da)
    pairs = [(i,j) for i,j in ab.items() if ba.get(j) == i]
    if len(pairs) < 6:
        return "few_mutual_correspondences"
    p = a[[i for i,j in pairs]]
    q = b[[j for i,j in pairs]]
    matrix, mask = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC, ransacReprojThreshold=3.)
    if matrix is None or mask is None:
        return "no_transform"
    mask = mask.ravel().astype(bool)
    if mask.sum() < 6 or mask.mean() < .67:
        return "few_geometrically_consistent"
    for pts, size in ((p[mask], size_a), (q[mask], size_b)):
        if len(np.unique(np.rint(pts),axis=0)) < 6:
            return "few_unique_coordinates"
        if min(np.ptp(pts,axis=0)/size) < .18:
            return "matches_spatially_concentrated"
    return "pass"


def interval(value, cutpoints, labels):
    return labels[np.searchsorted(cutpoints, value, side="right")]


def main(root, out):
    sys.path[:0] = [str(root/p) for p in ("jepa_reid_v23/report", "jepa_report_v19", "jepa_reid_v21/association")]
    import appearance
    import report_evidence as verifier
    from evaluate import label_boxes
    cv2.setNumThreads(1)
    data = root/"jepa_reid_v21/data"
    frames = [r for r in rows(data/"frames.jsonl") if r["split"] == "test"]
    cases = select(frames)
    by_id = {f["frame_id"]:f for f in frames}
    @lru_cache(maxsize=64)
    def image(fid):
        return cv2.cvtColor(cv2.imread(str(data/by_id[fid]["rgb"])), cv2.COLOR_BGR2RGB)

    def check(reference, frame, uid):
        im = image(frame["frame_id"])
        template = appearance.match_reference(reference, im)
        menu = verifier.eligible_boxes([{"bbox":ob["xyxy"]} for ob in frame["boxes"]], im.shape)
        cv2.setRNGSeed(7)
        winner, stage = verifier.current_support(reference, template, im, menu)
        outcome = "withheld"
        if winner is not None:
            label, = label_boxes([np.array(winner)*im.shape[1]/appearance.WORK_WIDTH], frame["boxes"])
            outcome = "unverified" if label is None else ("correct" if label["instance"] == uid else "wrong")
        return outcome, stage, template

    records = []
    for key, a, b, first, last in cases:
        uid = first["instance"]
        rgb = image(b["frame_id"])
        ref = appearance.build_reference(image(a["frame_id"]), first["xyxy"])
        assert ref is not None
        outcome, gate, template = check(ref, b, uid)
        self_check, self_gate, _ = check(ref, a, uid)
        curr = verifier.eligible_boxes([{"bbox":last["xyxy"]}], rgb.shape)
        sift = "target_box_ineligible"
        if curr:
            x,y,u,v = curr[0]
            crop = appearance._working(rgb)[y:v,x:u]
            cv2.setRNGSeed(7)
            sift = trace_sift(ref.patch, crop)
            cv2.setRNGSeed(7)
            assert (sift == "pass") == bool(verifier.local_correspondence(ref.patch, crop))
        # The following geometry is privileged and used only for post-hoc strata.
        eye_a, eye_b = [np.array(f["T_world_from_camera"])[:3,3] for f in (a, b)]
        origin = np.array(first["world_center"])
        va, vb = eye_a-origin, eye_b-origin
        angle = np.degrees(np.arccos(np.clip(va@vb/(np.linalg.norm(va)*np.linalg.norm(vb)), -1, 1)))
        minsize = min(first["pixels"], last["pixels"])/np.prod(rgb.shape[:2])
        boxes = np.array([first["xyxy"], last["xyxy"]])
        areas = (boxes[:,2:] - boxes[:,:2]).prod(1)
        scale_ratio = np.sqrt(max(areas)/min(areas))
        support = "not_localized"
        if template.matched:
            support = "unique_target_overlap" if curr and verifier.overlap(template.box, curr[0]) >= .30 else "localized_elsewhere"
        records.append(dict(key=key, house=a["house"], uid=uid, source=a["frame_id"], live=b["frame_id"],
            cat=a["category"], outcome=outcome, gate=gate, sift_target=sift, template=template.reason,
            template_target=support, self_check=self_check, self_gate=self_gate,
            object_view_deg=interval(angle, [15,45,90], ["0to15","15to45","45to90","90plus"]),
            small_visible=interval(minsize,[.02,.10], ["under2pct","2to10pct","10pct_plus"]),
            scale=interval(scale_ratio,[1.5,2],["under1p5","1p5to2","2plus"]),
            sequence="same_sequence" if a["sequence"]==b["sequence"] else "different_sequence"))
    def counts(rs, k):
        return dict(sorted(Counter(r[k] for r in rs).items()))
    groups = {}
    for k in ("object_view_deg","small_visible","scale","sequence","cat"):
        strata = defaultdict(list)
        for row in records:
            strata[row[k]].append(row)
        groups[k] = {key:dict(n=len(rs), houses=len(set(r["house"] for r in rs)), outcomes=counts(rs,"outcome"))
                     for key, rs in sorted(strata.items())}
    unmatched = [r for r in records if r["gate"] == "no_current_instance_match"]
    result = dict(panel_sha256=digest([r["key"] for r in records]), units=len(records),
        houses=len(set(r["house"] for r in records)), instances=len(set(r["uid"] for r in records)),
        unique_frames=len(set(r[z] for r in records for z in ("source","live"))),
        outcome=counts(records,"outcome"), registered_image_control=counts(records,"self_check"),
        registered_image_withheld_gates=counts([r for r in records if r["self_check"]=="withheld"],"self_gate"),
        paired_self_to_revisit=dict(Counter(r["self_check"]+"->"+r["outcome"] for r in records)),
        gates=counts(records,"gate"), strata=groups,
        no_match=dict(template=counts(unmatched,"template"), localization=counts(unmatched,"template_target"),
                      true_crop_sift_first_failed_check=counts(unmatched,"sift_target")),
        gt_roles="Independent Habitat instance labels for case selection and scoring. Oracle source box and complete eligible GT current menu as input ablation. No instance IDs or truth geometry given to either matcher.",
        registration_control="Feed source image back as live to the exact same cross-view verifier with the complete source box menu. Same row weighting as cross-view. This is a benign same-capture control, not the separate production same-capture fast path.",
        view_definition="Object-centered 3D camera-ray angular difference. Visible size uses the smaller of the two semantic mask pixel fractions. Scale is square root of max/min GT visible box area.",
        paid_calls=0,
        frozen_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(appearance.__file__),Path(verifier.__file__),data/"frames.jsonl"]})
    assert (result["units"],len(unmatched),result["outcome"].get("correct"))==(192,132,49)
    out.mkdir(parents=True,exist_ok=True)
    private = out/"private"
    private.mkdir(exist_ok=True)
    (private/"positive_cases.jsonl").write_text("".join(json.dumps(row)+"\n" for row in records))
    (out/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("project", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    main(args.project,args.out)
