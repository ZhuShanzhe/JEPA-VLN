"""Cross-check a development-only frontier substitution with the official evaluator."""
import json
import sys
from pathlib import Path


def main(folder):
    receipt = json.loads((folder / "branch_execution.json").read_text())
    episode, = json.loads((folder / "result.json").read_text())["episodes"]
    picked = receipt["selection"]
    start, stop = picked["step"], receipt["ending_step"]
    paper = episode["paper"]
    items = [r for r in paper["agent_trace"]["decisions"] if start <= r["step"] < stop]
    assert len(items) == 1
    request = items[0]
    assert request["decision_origin"] == "vlm" and request["validation"] == "ok"
    assert request["output"]["action"] == "GOTO_FRONTIER"
    assert request["output"]["target_id"] == picked["original"]
    legal = {r["id"] for r in request["world_state"]["frontiers"]}
    assert picked["original"] != picked["executed"] and {picked["original"], picked["executed"]} <= legal
    actions = [r["action"] for r in paper["steps"] if start <= r["step"] < stop]
    assert actions == receipt["actions"] and 1 in actions and receipt["displacement_m"] > 0.1
    records = [json.loads(s) for s in (folder / "agent/frontier_probe.jsonl").read_text().splitlines()]
    assert any(r["step"] == start and r["event"] == "activation" and
               r["selected_id"] == picked["executed"] and r["activated"] for r in records)
    assert any(r["step"] == start and r["event"] == "execution_selection" and
               r["applied"] and r["original"] == picked["original"] and r["executed"] == picked["executed"]
               for r in records)
    print(json.dumps({"passed": True, "candidate_count": len(legal), "verified_actions": len(actions),
                      "displacement_m": receipt["displacement_m"], "training_steps": 0}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
