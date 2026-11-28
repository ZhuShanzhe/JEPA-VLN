"""Four single-decision RGB-only GPT-4o handoffs and a separate mask-based audit."""
import argparse
import base64
from collections import defaultdict
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dino_cue_review import CAMERAS, PRIVATE, digest, evaluate_box, read


HERE = Path(__file__).resolve().parent
RECEIPTS = PRIVATE / "gpt_multiview"
MODEL = "gpt-4o-2024-11-20"
PROMPT = """In an indoor object search, select ONE numbered detector proposal worth revisiting for the named goal, or null if none has a credible object-specific visual cue. Each row is one actual view from the same position. On the left is the unmarked image; on the right is the same image with possible candidate boxes and their IDs. The views are right, center and left. Compare ALL of them. Each box is only a suggestion and may mark the wrong object. A plausible room or support surface alone is insufficient. An identifiable target or a partially seen candidate with distinctive evidence is worth selecting. When several boxes represent the same plausible target, select the clearest one. Return exactly one JSON object, {"candidate":"ID"} or {"candidate":null}."""
PREFIXES = dict(right="R", center="C", left="L")


def encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_exclusive(path, value):
    with path.open("xb") as handle:
        handle.write(encode(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def boxes_and_inputs():
    # Both inputs are inference-time data. Masks are opened only inside score().
    images, rows = read(PRIVATE / "inputs.json"), read(PRIVATE / "proposals.json")
    assert len(rows) == len(images) == 12 and [r["id"] for r in rows] == [x["id"] for x in images]
    groups = defaultdict(dict)
    for x, r in zip(images, rows):
        view, group = x["view"], x["group"]
        assert view in CAMERAS and view not in groups[group]
        path = PRIVATE / (x["id"] + ".png")
        assert digest(path) == x["rgb_sha256"]
        # Assign IDs by image coordinates, without disclosing detector confidence/rank.
        ordered = sorted(r["augment"], key=lambda b: tuple(b["xyxy"]))
        boxes = {f"{PREFIXES[view]}{n}": b for n, b in enumerate(ordered, 1)}
        groups[group][view] = dict(source=x, path=path, boxes=boxes)
    assert len(groups) == 4 and all(set(v) == set(CAMERAS) for v in groups.values())
    return list(groups.values())


def to_png(views):
    size, head = 384, 27
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    dst = Image.new("RGB", (size * 2, (size + head) * 3))
    draw = ImageDraw.Draw(dst)
    for n, view in enumerate(CAMERAS):
        y = (size + head) * n
        data = views[view]
        with Image.open(data["path"]) as im:
            assert im.size == (192, 192)
            rgb = im.convert("RGB").resize((size, size), Image.Resampling.NEAREST)
        marked = rgb.copy()
        overlay = ImageDraw.Draw(marked)
        for i, (key, b) in enumerate(data["boxes"].items()):
            x1, y1, x2, y2 = [int(np.clip(round(float(v) * 2), 0, size - 1)) for v in b["xyxy"]]
            color = ["yellow", "cyan", "lime", "magenta"][i % 4]
            overlay.rectangle((x1, y1, x2, y2), outline=color, width=2)
            tx, ty = min(x1, size - 42), max(0, y1 - 21)
            overlay.rectangle((tx, ty, tx + 39, ty + 19), fill="black")
            overlay.text((tx + 1, ty), key, font=font, fill=color)
        draw.text((4, y+3), f"{view.upper()} | original", font=font)
        draw.text((size + 4, y+3), f"{view.upper()} | candidate IDs", font=font)
        dst.paste(rgb, (0, y + head))
        dst.paste(marked, (size, y + head))
    data = io.BytesIO()
    dst.save(data, format="PNG")
    return data.getvalue()


def materials(views):
    goals = {d["source"]["goal"] for d in views.values()}
    assert len(goals) == 1
    offered = [key for v in CAMERAS for key in views[v]["boxes"]]
    label = "Goal: " + goals.pop() + ". Offered IDs: " + (", ".join(offered) or "none") + ". Return JSON."
    sheet = to_png(views)
    request = dict(model=MODEL, temperature=0, max_tokens=40, response_format={"type":"json_object"},
                   messages=[{"role":"system","content":PROMPT},{"role":"user","content":[
                       {"type":"text","text":label},
                       {"type":"image_url","image_url":{"detail":"high","url":"data:image/png;base64," +
                                                                    base64.b64encode(sheet).decode()}}]}])
    return request, offered


def infer():
    if os.environ.get("NAV_VLM_MODEL") != MODEL or os.environ.get("NAV_VLM_API_URL","").rstrip("/") != "http://127.0.0.1:28938/v1":
        raise RuntimeError("pinned APIYI gateway profile required")
    secret = os.environ["NAV_VLM_API_KEY"]
    assert secret
    RECEIPTS.mkdir(mode=0o700, exist_ok=True)
    prepared = [materials(g) for g in boxes_and_inputs()]
    for i, (request, offered) in enumerate(prepared):
        wire = encode(request)
        req_hash = hashlib.sha256(wire).hexdigest()
        success, dispatch = RECEIPTS / f"{i}.response.json", RECEIPTS / f"{i}.dispatch.json"
        if success.exists():
            assert read(success)["request_hash"] == req_hash
            continue
        # Even an unknown-billing or invalid reply consumes this event's allowance.
        write_exclusive(dispatch, {"request_hash":req_hash, "event_index":i, "model":MODEL})
        conn = http.client.HTTPConnection("127.0.0.1", 28938, timeout=480)
        try:
            conn.request("POST", "/v1/chat/completions", body=wire, headers={
                "Authorization":"Bearer "+secret,"Content-Type":"application/json"})
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f"gateway status {response.status}")
            value = json.loads(response.read(1024*1024))
        finally:
            conn.close()
        assert value["model"] == MODEL
        answer = json.loads(value["choices"][0]["message"]["content"])
        assert set(answer) == {"candidate"} and (answer["candidate"] is None or answer["candidate"] in offered)
        usage = value.get("usage", {})
        write_exclusive(success, dict(request_hash=req_hash, model=value["model"], choice=answer["candidate"],
            total_tokens=usage.get("total_tokens"), cost=usage.get("cost")))
        print("accepted event", i+1, flush=True)


def score():
    public = []
    rows = boxes_and_inputs()
    for i, views in enumerate(rows):
        request, _ = materials(views)
        rec = read(RECEIPTS / f"{i}.response.json")
        assert rec["request_hash"] == read(RECEIPTS / f"{i}.dispatch.json")["request_hash"] == hashlib.sha256(encode(request)).hexdigest()
        assert rec["model"] == MODEL
        candidates, positive = {}, []
        for v in CAMERAS:
            data = views[v]
            mask = np.load(PRIVATE / (data["source"]["id"]+".npy"), allow_pickle=False)
            envelopes = {}
            for sid in np.unique(mask):
                if sid:
                    yy, xx = np.where(mask==sid)
                    envelopes[int(sid)] = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
            if envelopes:
                positive.append(v)
            for key, box in data["boxes"].items():
                candidates[key] = evaluate_box(box, mask, envelopes)
        choice = rec["choice"]
        selected = candidates[choice] if choice is not None else {}
        top = max(candidates, key=lambda k: candidates[k]["confidence"], default=None)
        public.append(dict(event=i+1, goal=data["source"]["goal"], target_pixel_views=positive,
            n_candidates=len(candidates), any_valid=any(c["both"] for c in candidates.values()),
            detector_highest_valid=bool(top and candidates[top]["both"]), choice=choice,
            choice_center=selected.get("center"), choice_iou25=selected.get("iou25"),
            choice_valid=selected.get("both"), total_tokens=rec["total_tokens"], provider_cost=rec["cost"]))
    old = read(HERE / "dino_cue_review.json")["results"]["augment"]["by_group"]
    assert sum(x["any_valid"] for x in public) == sum(bool(y["views_with_any_correct_box"]) for y in old)
    assert sum(x["detector_highest_valid"] for x in public) == sum(y["highest_confidence_box_correct"] for y in old)
    assert len(list(RECEIPTS.glob("*.dispatch.json"))) == len(list(RECEIPTS.glob("*.response.json"))) == 4
    report=dict(model=MODEL, temperature=0, source_code_sha256=digest(Path(__file__)), prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
        inputs="Three contemporaneous RGB views, each with its unmarked original and all coordinate-ordered augment-proposal overlays. No private trajectory, simulator masks, instance labels, house IDs, detector scores, or outcomes sent. One decision per grant.",
        ground_truth_metric="box center on a target instance AND IoU >= 0.25 to visible-pixel envelope of that same instance",
        requests=4, grants=4, houses=len({x["right"]["source"]["group"].split("_")[0] for x in rows}),
        grants_with_target_pixels=sum(bool(r["target_pixel_views"]) for r in public), grants_with_candidates=sum(r["n_candidates"]>0 for r in public),
        oracle_candidate_ceiling=sum(r["any_valid"] for r in public), detector_highest_valid=sum(r["detector_highest_valid"] for r in public),
        selected=sum(r["choice"] is not None for r in public), abstained=sum(r["choice"] is None for r in public),
        selected_valid=sum(r["choice_valid"] is True for r in public), selected_invalid=sum(r["choice_valid"] is False for r in public),
        total_tokens=sum(r["total_tokens"] for r in public), by_grant=public)
    dst=HERE/"gpt_multiview_cue.json"
    dst.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["infer","score"])
    action=parser.parse_args().action
    {"infer":infer,"score":score}[action]()
