"""Summarize existing endpoint records without republishing any image or trajectory."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(closed, clock, painting, controls):
    rows = {}
    for name, case in zip(("clock", "painting"), closed["cases"], strict=True):
        attempt = next(a for a in case["attempts"] if not a["quarantined"])
        assert case["score"]["steps"] == attempt["physical_actions"] == 300
        flow = case["closed_candidate_flow"]
        rows[name] = dict(
            proposed=flow["v24_distinct_proposals"],
            distinct_registered=flow["v24_distinct_registered"],
            physical_confirm_actions=sum(attempt["confirmation_actual_actions"].values()),
            confirm_action_mismatches=attempt["confirmation_action_mismatches"],
            confirm_termination=attempt["confirmation_end_reasons"],
            actual_report_actions=attempt["physical_reports"],
            credited_new=case["score"]["new"],
            pose_jepa_scoring_calls=attempt["pose_forward_calls"])
    rows["clock"]["first_rechecked_source_target"] = dict(
        correct_source_click=clock["geometry_receipt"]["click_on_task_instance"],
        same_simulator_object_visible_on_arrival=clock["same_most_visible_instance_start_and_finish"],
        registered_object_in_range_on_arrival=clock["states"]["finish"]["source_instance_in_report_range"],
        reason=clock["offline_frozen_detector"]["replayed_support_reason"])
    rows["painting"]["source_frame_audit"] = dict(
        distinct_source_frames=painting["distinct_source_frames"],
        candidates_with_target_in_source=sum(
            r["entire_frame_task_pixels"] > 0
            for group in painting["by_review"].values() for r in group["records"]),
        rejected=painting["by_review"]["reject"]["total"],
        lacking_valid_review=painting["by_review"]["unreviewed"]["total"])
    measures = {}
    for key in ("positive", "missing_detection", "original_false_candidate"):
        run = controls[key]
        outcome = run["native_viewpoint_audit"]
        measures[key] = dict(correct_registered_source=run["reference_has_task_target"],
                             report_dispatched=run["report_dispatched"],
                             official_new=outcome["credited_new_instances"],
                             source_matches_credit=outcome.get("source_region_matches_credited_instance"))
    return dict(complete_development_episodes=rows, scripted_simulation_controls=measures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v24_directory", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    base = args.v24_directory
    inputs = [base / "results.json", base / "verification/closed_clock_confirm_audit.json",
              base / "verification/closed_painting_candidate_audit.json", base / "verification/free_checks.json"]
    data = summarize(*(read(path) for path in inputs))
    args.destination.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
