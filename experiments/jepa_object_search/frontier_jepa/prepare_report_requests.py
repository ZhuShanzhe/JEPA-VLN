"""Export only decision-time fields of the controlled fixture into inference examples."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--examples",type=Path,required=True)
    args = parser.parse_args()
    if args.examples.exists():
        parser.error("Output must be a new directory")
    room = json.loads((args.fixture / "scene_smoke.json").read_text())
    assert room["status"] == "pass"
    args.examples.mkdir(parents=True)
    shutil.copyfile(args.fixture / "start.png", args.examples / "start.png")
    action = {"move_forward":0,"turn_left":1,"turn_right":2}
    candidates = [{"id":name,"relative_xz_m":room["candidates"][name]["target_relative_xz_m"],
                   "planned_actions":[action[a] for a in room["candidates"][name]["planned_actions"]]}
                  for name in ("left","right")]
    for name in ("left","right"):
        report = room["report_histories"][name]
        imagename = f"observed_{name}.png"
        shutil.copyfile(args.fixture / report["report_rgb"], args.examples / imagename)
        pos = report["report_observed_pose"]
        xy = [pos[i]-room["start"][i] for i in (0,2)]
        # No branch outcomes, object IDs, semantic boxes, images beyond the decision or scores.
        req = {"observed_rgb":["start.png"],"candidates":candidates,
               "reports":[{"observed_rgb":imagename,"observed_camera_xz_m":xy}]}
        (args.examples / f"reported_{name}.json").write_text(json.dumps(req,indent=2))


if __name__ == "__main__":
    main()
