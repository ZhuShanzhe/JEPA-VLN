"""Produce a shareable, scalar-only engineering receipt from the private run artifacts."""
import json
import sys
from pathlib import Path

from verify_branch import main as verify_direction


root = Path(__file__).resolve().parents[2]
rt = root / "mapping_runtime"


def read(path):
    return json.loads(path.read_text())


runs = rt / "runs"
verify_direction(runs / "branch_execution_03")
geo = read(runs / "physical_01/geometry.json")
obs, = read(runs / "most_observed_01/interface_audit.json")
native, = read(runs / "most_native_manyon_01/result.json")["episodes"]
direction = read(runs / "branch_execution_03/branch_execution.json")
branch = read(runs / "branch_io_01/receipt.json")
judge = read(runs / "report_scoring_01/receipt.json")
checks = rt / "batch/engineering_checks"
attrs = read(checks / "text_attributes.json")
queue = read(checks / "replay06.json")
health = read(checks / "live01.json")
scope = rt / "external/scope"
sg = read(scope / "runs/goat_native_full02/verified_summary.json")
sm = read(scope / "runs/dev_many_final/verification.json")
dmem = rt / "external/three_dmem"
dm = read(dmem / "dev_many_budget_2/engineering_receipt.json")
mtu = root / "comparisons_nov14/mtu_native"
mg = read(mtu / "goat_val_seen_rank0/summary.json")
mm = read(mtu / "dev_many_native01/summary.json")
assert geo["status"] == branch["status"] == "pass"
assert obs["actions_checked"] == native["steps_taken"] == 300 and not native["timed_out"]
assert native["target_mode"] == "many" and native["required_target_count"] == 2
assert sm["native_steps"] == 1 and dm["completed_navigation_loops"] == 2 and sg["complete"]
assert queue["formal_episodes_started"] == 0 and queue["frozen_ids_checked"] == 300
assert read(dmem / "frozen_batch_preflight.json")["frozen_verified"] == 300
assert read(scope / "prepare_frozen/verified_input.json")["episodes"] == 300
assert read(mtu / "frozen_manifest_audit.json")["count"] == 300
assert attrs["checked_texts_and_no_reference_images"] == 300


def fields(item, *keys):
    return {k: item[k] for k in keys}


public = {
    "scope": "Pipeline acceptance on development tasks; no benchmark ranking",
    "frozen_text_episodes": 300, "formal_episodes_started_in_acceptance": 0,
    "panel_sha256": queue["release_panel_sha256"], "training_steps_in_acceptance": 0,
    "native_model": health["model"], "mapping": fields(geo, "observations", "actions_before_finish",
        "estimated_poses", "submaps", "sampled_map_points", "physical_displacement_span_m"),
    "mapping_capabilities": health["capabilities"],
    "observed_native_episode": {**fields(obs, "actions_checked", "scale_source", "max_candidates"),
        "verified_vlm_frontier_selections": sum(a["accepted_model_choice"] for a in obs["activations"])},
    "derived_manyon_native_episode": fields(native, "steps_taken", "trajectory_length",
        "true_positive_count", "false_positive_count", "timed_out"),
    "different_direction": {"reachable_candidates": direction["selection"]["candidate_count"],
        "verified_actions": len(direction["actions"]), "physical_displacement_m": direction["displacement_m"]},
    "branch_collection": fields(branch, "branches", "actions_per_branch", "identical_start_observation",
        "different_final_observations", "separate_online_and_future_files"),
    "controlled_reports": fields(judge, "correct_reports", "repeats", "false_positives"),
    "text_inputs": fields(attrs, "checked_texts_and_no_reference_images", "conditions",
        "distinct_attribute_target_objects"),
    "scope_native_goat": fields(sg, "native_subtasks_scored", "movement_steps", "native_frontier_decisions"),
    "scope_development_manyon": fields(sm, "native_steps", "path_m", "native_reports"),
    "three_dmem_development_manyon": fields(dm, "effective_budget", "completed_navigation_loops",
        "round_path_length_m", "candidate_reports"),
    "mtu3d_native_goat": fields(mg, "subgoals", "actions"),
    "mtu3d_development_manyon": fields(mm, "steps_taken", "trajectory_length", "true_positive_count",
        "false_positive_count", "repeat_report_count"),
    "queue_verification": {"frozen_ids_checked": queue["frozen_ids_checked"], "mode": "offline replay"},
}
with Path(sys.argv[1]).open("x") as f:
    json.dump(public, f, indent=2, allow_nan=False)
    f.write("\n")
