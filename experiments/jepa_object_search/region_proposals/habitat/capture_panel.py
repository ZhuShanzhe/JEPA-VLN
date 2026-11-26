"""Re-render fixed poses with Habitat. Release aggregate receipts only."""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from habitat_sim.utils.common import quat_from_angle_axis

from check_routes import semantic_scene
from evaluation.main.evaluator import _reset_environment, _env_action
from verify_habitat import deny_network, state

sys.addaudithook(deny_network)
P = Path(os.environ["NAV_ROOT"])
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(x) for x in path.read_bytes().splitlines()]


def write_rows(path, rows):
    temp = path.with_suffix(".part")
    temp.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    temp.replace(path)


def capture(private):
    from historical_trial import frozen
    _, _, start, _, _, source_step = frozen()
    clock_steps = {source_step: "registered", start["step"]: "before_approach",
                   start["step"] + len(start["planned_actions"]): "after_approach"}
    frames, truths = [], []
    root = P/"mapping_runtime/pilots/jepa_reid_v24_first2"
    for index, category in enumerate(("clock", "wall_art")):
        eid = read(root/"panel.json")["rows"][index]["episode_id"]
        ticket = read(root/"comparison/accepted/second"/(eid+".json"))
        result = Path(ticket["result"])
        assert digest(result) == ticket["result_sha256"]
        doc = read(result)
        ep, = doc["episodes"]
        env, item, dat = semantic_scene(doc)

        def add(frame, group, label):
            frame_id = f"{category}_{group}_{label}"
            rgb = np.asarray(frame["rgb"])[...,:3]
            seg = np.asarray(frame["semantic"]).squeeze()
            assert rgb.shape == (1280,1280,3) and seg.shape == (320,320)
            image = Image.fromarray(rgb).resize((518,518),Image.Resampling.BICUBIC)
            rgb_file = "rgb/" + frame_id + ".png"
            mask_file = "labels/" + frame_id + ".png"
            image.save(private/rgb_file)
            mapped = np.zeros_like(seg, dtype=np.uint8)
            boxes = []
            for iid, tid in enumerate(item.target_object_ids, 1):
                yy, xx = np.nonzero(seg == int(tid))
                if len(xx):
                    mapped[yy, xx] = iid
                    box = np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],dtype=float) * 518/320
                    boxes.append(dict(instance=iid,pixels=len(xx),xyxy=box.tolist()))
            Image.fromarray(mapped).save(private/mask_file)
            frames.append(dict(frame_id=frame_id,group=group,category=category,goal=ep["goal_text"],
                               rgb=rgb_file,width=518,height=518,rgb_sha256=digest(private/rgb_file)))
            truths.append(dict(frame_id=frame_id,boxes=boxes,mask=mask_file,mask_sha256=digest(private/mask_file)))

        try:
            if index == 0:
                selected = clock_steps
            else:
                keys = sorted({r["step"] for r in lines(result.parent/"agent/action_trace.jsonl")
                               if r.get("mapping_frame_id") and (r.get("mapping") or {}).get("is_keyframe")})
                selected = {keys[j]: "q"+str(i) for i,j in enumerate(np.linspace(0,len(keys)-1,4).round().astype(int))}
            f = _reset_environment(env,item,dat)
            for step, act in enumerate(ep["paper"]["steps"]):
                if step in selected:
                    assert np.linalg.norm(state(env)[0]-act["position_before"]) < .002
                    add(f,"archived",selected[step])
                if step >= max(selected):
                    break
                if act["action_role"] == "navigate":
                    f = env.step(_env_action(act["action"]))
            # Oracle placement is isolated from the unfixed trajectory views.
            definition = ep["paper"]["episode_definition"]
            order = 0
            for tid in item.target_object_ids:
                pts = definition["view_points"][str(tid)]
                center = np.asarray(definition["target_positions"][str(tid)])
                for j in np.linspace(0,len(pts)-1,min(6,len(pts))).round().astype(int):
                    position = np.asarray(pts[j])
                    dx,_,dz = center - position
                    quat = quat_from_angle_axis(np.arctan2(-dx,-dz), np.array([0.,1.,0.]))
                    obs = env.sim.get_observations_at(position=position, rotation=quat, keep_agent_at_new_pose=False)
                    add(obs, "target_facing", order)
                    order += 1
        finally:
            env.close()

    assert len(frames) == len({f["frame_id"] for f in frames}) == 43
    write_rows(private/"manifest.jsonl",frames)
    write_rows(private/"gt.jsonl",truths)
    stats = {}
    for frame, truth in zip(frames, truths):
        group = frame["category"]+"_"+frame["group"]
        valid = [b for b in truth["boxes"] if b["pixels"]>=103]
        stats.setdefault(group,Counter()).update(frames=1,positive_frames=int(bool(valid)),
                                                target_instances=len(valid),any_pixels=int(bool(truth["boxes"])))
    n0 = [max((b["pixels"] for b in row["boxes"]),default=0) for row in truths[:3]]
    assert n0 == [265,11693,17968], n0
    assert stats["wall_art_archived"]["any_pixels"] == 0
    assert stats["wall_art_target_facing"]["positive_frames"] == 10
    assert stats["clock_target_facing"]["positive_frames"] == 7
    info = dict(total=43, groups=stats, raw_rgb_size=[1280,1280], detector_rgb_size=[518,518],
                semantic_size=[320,320], interpolation="Pillow BICUBIC, lossless PNG",
                min_pixels=103, historical_clock_target_pixels=n0, paid_calls=0,
                scene_count=2, rendered_manifest_sha256=digest(private/"manifest.jsonl"),
                scoring_manifest_sha256=digest(private/"gt.jsonl"))
    (OUT/"capture_receipt.json").write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info),flush=True)


def detect(private):
    from proposal_worker import ProposalWorker
    worker = ProposalWorker(P/"b_verification/env/bin/python",P,os.environ.get("BOX_DEVICE","cuda"))
    outputs = []
    try:
        for f in lines(private/"manifest.jsonl"):
            image = np.asarray(Image.open(private/f["rgb"]).convert("RGB"))
            caption = "painting" if f["category"] == "wall_art" else "clock"
            found = worker.detect(image,caption)
            outputs.append(dict(frame_id=f["frame_id"],width=f["width"],height=f["height"],boxes=found))
    finally:
        worker.close()
    write_rows(private/"dino.jsonl",outputs)
    print("frames",len(outputs),"boxes",sum(len(r["boxes"]) for r in outputs))


if __name__ == "__main__":
    opts = argparse.ArgumentParser()
    opts.add_argument("mode",choices=["capture","detect"])
    args = opts.parse_args()
    private = OUT/"private"
    private.mkdir(exist_ok=True,mode=0o700)
    for sub in ("rgb","labels"):
        (private/sub).mkdir(exist_ok=True,mode=0o700)
    (capture if args.mode=="capture" else detect)(private)
