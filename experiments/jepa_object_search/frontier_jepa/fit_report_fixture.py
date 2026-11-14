"""Run the two-history Blender/Habitat module contract on real observed RGB."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from memory import ReportMemoryRanker, load_memory_ranker
from vision import ContextEncoder


SIDES = ("left", "right")
ACTION = {"move_forward":0,"turn_left":1,"turn_right":2}


def prepare(folder, encoder):
    with (folder / "scene_smoke.json").open() as f:
        scene = json.load(f)
    assert scene["status"] == "pass" and scene["actions_per_branch"] == 19
    start_img = np.asarray(Image.open(folder / "start.png").convert("RGB"))
    context = torch.from_numpy(encoder.encode(start_img[None]))
    xy = torch.tensor([scene["candidates"][b]["target_relative_xz_m"] for b in SIDES])
    plans = torch.tensor([[ACTION[s] for s in scene["candidates"][b]["planned_actions"]]
                          for b in SIDES])
    # Use the entire observed report frame without semantic mask or ground-truth box.
    reports, observed_at = [], []
    for history in SIDES:
        h = scene["report_histories"][history]
        assert h["return_position_error_m"] < .02
        rgb = np.asarray(Image.open(folder / h["report_rgb"]).convert("RGB"))
        reports.append(encoder.encode(rgb[None]))
        observed_at.append(np.asarray(h["report_observed_pose"])[[0,2]] -
                           np.asarray(scene["start"])[[0,2]])
    reports = torch.tensor(np.stack(reports))
    observed_at = torch.tensor(np.stack(observed_at), dtype=torch.float32)
    # The following assets only generate training targets. Runtime score() never reads them.
    teacher = []
    for b in SIDES:
        archive = folder / f"{b}_trajectory.npz"
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == scene["trajectory_sha256"][b]
        with np.load(archive) as data:
            frames = data["rgb"]
            assert np.array_equal(frames[0],start_img)
            teacher.append(encoder.encode(frames[1:]))
    y = torch.tensor([[scene["report_histories"][r]["candidate_labels"][b] for b in SIDES]
                      for r in SIDES],dtype=torch.float32)
    assert torch.equal(y,torch.tensor([[0,1],[1,0]],dtype=torch.float32))
    return scene,context,xy,plans,reports,observed_at,torch.tensor(np.stack(teacher)),y


def run(folder, stage, epochs, device):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    encoder = ContextEncoder(device=device)
    scene, context, xy, plans, reports, report_xy, teacher, y = prepare(folder,encoder)
    del encoder
    model = ReportMemoryRanker()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.003)
    start_loss = None
    for _ in range(epochs):
        logits = []
        for i in range(2):
            scores, predicted = model(context,xy,plans,reports[i:i+1],report_xy[i:i+1])
            logits.append(scores)
        logits = torch.stack(logits)
        bce = F.binary_cross_entropy_with_logits(logits,y)
        cosine = (1-F.cosine_similarity(predicted,model.visual(teacher))).mean()
        loss = bce + cosine
        if start_loss is None:
            start_loss = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
    torch.save({"format":"one-room-report-visual-fixture/v1","state":model.state_dict()},
               stage / "controlled.pt")
    restored = load_memory_ranker(stage / "controlled.pt")
    decisions, poses_only = {}, {}
    for i, history in enumerate(SIDES):
        scores = restored.score(context,xy,plans,reports[i:i+1],report_xy[i:i+1])
        reverse = restored.score(context,xy.flip([0]),plans.flip([0]),reports[i:i+1],report_xy[i:i+1])
        np.testing.assert_allclose(scores,reverse[::-1],atol=1e-6)
        choice = int(np.argmax(scores))
        assert y[i,choice] == 1, "Controlled same-start report swap did not change preferred branch"
        poses_only[history] = SIDES[int(torch.cdist(xy,report_xy[i:i+1]).flatten().argmax())]
        decisions[history] = {"chosen":SIDES[choice],"candidate_scores":scores}
    empty = restored.score(context,xy,plans,torch.zeros(0,768),torch.zeros(0,2))
    np.testing.assert_allclose(empty,[.5,.5])
    audit = dict(scope="training-fixture fitted on one room; no held-out test or navigation score",
                 scene_sha256=scene["scene_sha256"],trajectories_sha256=scene["trajectory_sha256"],
                 checkpoint_sha256=hashlib.sha256((stage/"controlled.pt").read_bytes()).hexdigest(),
                 seed=7,epochs=epochs,training_states=1,constructed_histories=2,
                 candidate_plans=2,fit_pairs=4,objective_start=start_loss,objective_final=float(loss.detach()),
                 label="eligible same-category instance seen on branch given constructed prior credit",
                 input_protocol="observed start RGB; preplanned candidate XY/commands; past report full-frame RGB + past observed camera XY",
                 teacher_protocol="offline actual future RGB latent target and supplied branch utility",
                 uses_semantic_images_as_input=False, uses_truth_instance_id_as_input=False,
                 uses_ground_truth_box_as_input=False,
                 controlled_decisions=decisions, no_reports_scores=empty,
                 poses_only_farthest_candidate=poses_only, out_of_sample_evaluations=0,
                 empty_memory_neutral=True, candidate_permutation_equivariant=True)
    (stage / "receipt.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene",type=Path,default=root / "frontier_scenes" / "acceptance")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=120)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    args = parser.parse_args()
    if args.output.exists() or args.epochs <= 0:
        parser.error("output must be new and epochs must be positive")
    args.output.parent.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        stage = Path(tmp) / "run"
        stage.mkdir()
        run(args.scene,stage,args.epochs,args.device)
        stage.rename(args.output)


if __name__ == "__main__":
    main()
