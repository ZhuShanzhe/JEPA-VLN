"""Offline physical receipt check for an unregistered near-field hypothesis."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import math
import os
import tempfile

import cv2
import numpy as np

import verify_views as fixture
from agents.navigator import OccupancyGrid
from candidate_scout import CandidateScoutMixin
from proposal_worker import ProposalWorker
from run_agent import LearnedVerificationFrontier
from test_chain import ReportChainTests


PROJECT = Path(os.environ["NAV_ROOT"])


def jpeg(image):
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    return encoded.tobytes()


def local_geometry(anchor, positions, box, sensor):
    # Use the same source bearing, supplied 60 cm range, and already-traversed
    # 23 cm corridor as the registered-clock fixture. No semantic or navmesh input.
    yaw = anchor[2]
    camera = np.eye(4)
    camera[:3, :3] = [[-math.sin(yaw), 0, math.cos(yaw)],
                      [math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0]]
    bearing = math.atan((1 - (box[0] + box[2]) / sensor.width) *
                        math.tan(math.radians(float(sensor.hfov)) / 2))
    point = [.6 * math.cos(yaw + bearing), .6 * math.sin(yaw + bearing), 0.]
    low = np.array([-2.5, -2.5])
    axis = (np.arange(100) + .5) * .05 - 2.5
    x, y = np.meshgrid(axis, axis)
    free = np.zeros(x.shape, bool)
    for start, stop in zip(positions, positions[1:]):
        for t in np.linspace(0, 1, 11):
            px, py = (1 - t) * np.array(start[:2]) + t * np.array(stop[:2])
            free |= (x - px) ** 2 + (y - py) ** 2 <= .23 ** 2
    return camera, point, OccupancyGrid(.05, low, free, np.zeros_like(free))


def run_case(env, item, data, ep, box, source_rgb, worker, scratch):
    frame, pose, anchor, _, positions, masks, _ = fixture.input_setup(env, item, data, ep, box)
    source_step, step = 74, 81
    before = fixture.rgb(frame)
    camera, point, grid = local_geometry(anchor, positions, box,
                                        env.sim.habitat_config.agents.main_agent.sim_sensors.rgb)
    helper = ReportChainTests()
    helper.rgb, helper.jpeg = source_rgb, jpeg(source_rgb)
    agent = helper.make_agent(LearnedVerificationFrontier)
    agent.output_dir = scratch
    agent._estimated_current_pose = pose
    agent.report_near_dist_m = .8
    agent._frontier_grid = grid
    agent._proposal_worker = worker
    agent._last_motion_failed = False
    agent._queued_pick_results = []
    agent._pending_decision_images = []
    agent._dup_review_payload = lambda: ([], [], 0)
    # This local frame store substitutes only the mapper transport; every stored
    # image is the RGB returned by the simulator at its corresponding pose.
    store = {source_step: helper.jpeg}
    agent.client.get_frame_image = lambda fid: ({"found": fid in store}, store.get(fid, b""))
    agent.client.get_frame_pose = lambda fid: camera if fid == source_step else None
    agent.client.resolve_candidates = lambda _: {"candidates": {"probe": {"found": True, "point": point}}}
    events = []
    agent._note = lambda name, iid, **values: events.append(dict(event=name, **values))
    agent._proposal_queue["probe"] = dict(candidate_id="probe", bbox=box, frame_id=source_step,
                                         pixel_norm=[500, 500], query="clock", status="pending",
                                         step=step, goal_index=None)
    agent.now = step
    receipt = agent._tool_commit_candidates([dict(candidate_id="probe", verdict="UNCERTAIN")], "clock")
    assert receipt.get("uncertain") == ["probe"]
    previous = 1
    motions = []
    start_step = step
    while True:
        obs = SimpleNamespace(rgb=fixture.rgb(frame), step_count=step, previous_action=previous,
                              goal_text=ep["goal_text"], target_mode="many", target_count=2,
                              episode_id="physical-clock")
        agent.now, agent._last_observation = step, obs
        agent._last_feed_info = {"is_keyframe": True, "frame_id": step}
        store[step] = jpeg(obs.rgb)
        action = agent._weak_action(obs)
        if action is None:
            break
        assert len(motions) < 12 and action in (1, 2, 3)
        frame, motion = fixture.physical(env, action)
        motions.append(motion)
        agent._last_motion_failed = action == 1 and motion["translation_m"] < .005
        previous, step = action, step + 1

    assert motions and agent._recent_inspection and not agent._detector_failed, str(events)
    assert not agent.instance_store.available() and agent._pending_report is None
    assert agent._proposal_queue["probe"]["status"] == "uncertain"
    agent._ev_deciding = True
    # The underlying VLM request builder is offline. Exercise the actual mixin
    # handoff to its multimodal state and verify the returned JPEG pixel for pixel.
    with patch.object(CandidateScoutMixin, "_build_decider_input", return_value=({}, None)):
        state, _ = agent._build_decider_input(obs)
    handoff = state["verification_observation"]
    returned = dict(agent._pending_decision_images)[handoff["attachment"]]
    h, w = obs.rgb.shape[:2]
    small = cv2.resize(obs.rgb, (min(640,w), max(1,round(h*min(640,w)/w))), interpolation=cv2.INTER_AREA)
    assert returned == jpeg(small)
    assert handoff["new_frame_id"] == step and step > start_step
    assert handoff["original_candidate"] == "probe"
    assert not np.array_equal(before, obs.rgb)
    score_event = next(e for e in events if e["event"] == "uncertain_view_start")
    end = next(e for e in events if e["event"] == "uncertain_view_end")
    detector_counts = [len(v) for (k, _), v in agent._local_boxes.items() if k[0] == "live"]
    assert agent._uncertain_scorer is not None
    # Only now inspect target identity / semantic pixels (native sensor is 1/4 RGB).
    after = np.asarray(frame["semantic"])
    truth = fixture.verify_pixels(item, masks[0], box, [masks[-1], after])
    targets = item.target_object_ids
    return dict(head=type(agent._uncertain_scorer).__name__, ranked_views=score_event["options"],
                initial_plan_actions=score_event["actions"], completed_actions=len(motions),
                distance_m=round(sum(m["translation_m"] for m in motions), 3),
                absolute_turn_deg=round(sum(abs(m["turn_deg"]) for m in motions),1),
                collisions=sum(m["collided"] for m in motions), detector_checks=len(detector_counts),
                return_reason=end["reason"], final_candidate_status=agent._proposal_queue["probe"]["status"],
                new_frame_handed_to_reviewer=True, exact_attachment_verified=True, registered=0, reports=0,
                after_motion_audit=dict(reference_overlaps_target=truth["reference_task_target"],
                       source_instance_pixels_before_after=truth.get("pixels_each_view"),
                       any_requested_instance_pixels_before_after=[
                           int(np.isin(mask,targets).sum()) for mask in [masks[-1],after]]))


def main():
    doc, ep, _, _ = fixture.case()
    receipt_dir = PROJECT / "mapping_runtime/pilots/jepa_reid_v23_first2/comparison/accepted/second"
    run_path = Path(fixture.load(receipt_dir / (ep["episode_id"] + ".json"))["result"])
    trace = run_path.parent / "agent/decision_trace.jsonl"
    rows = (json.loads(raw) for raw in trace.read_bytes().splitlines())
    boxes = next(row["world_state"]["local_rgb_proposals"]["boxes"] for row in rows if row["step"] == 74)
    env, item, data = fixture.semantic_scene(doc)
    worker = ProposalWorker(PROJECT / "b_verification/env/bin/python", PROJECT, "cuda")
    out = {}
    try:
        source_rgb = fixture.rgb(fixture.replay_until(env,item,data,ep["paper"]["steps"],74))
        for head in ("local", "context"):
            with patch.dict(os.environ, {"JEPA_UNCERTAIN_HEAD":head}), tempfile.TemporaryDirectory() as tmp:
                out[head] = {name:run_case(env,item,data,ep,boxes[i],source_rgb,worker,tmp)
                             for name,i in (("true_source",1),("wrong_source",0))}
            assert out[head]["true_source"]["after_motion_audit"]["reference_overlaps_target"]
            assert not out[head]["wrong_source"]["after_motion_audit"]["reference_overlaps_target"]
    finally:
        worker.close()
        env.close()
    report = dict(protocol="Controlled HM3D clock replay. Pre-input native UNCERTAIN review, source box, "
                          "60 cm range, ideal odometry, traversed-corridor map and keyframe transport. "
                          "Actual Habitat movements, detector, policy scores and inspection attachment. "
                          "No second VLM verdict or terminal navigation score.", paid_calls=0, trials=out)
    Path(__file__).with_name("uncertain_physical_results.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
