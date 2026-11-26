"""Offline state transitions in the project adapter with controlled, synthetic observations."""
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from test_chain import REPORT, TURN, ReportChainTests
from test_report_recovery import ReportRecoveryChecks
from provider_agent import BoxComparisonAgent


def run_current():
    fixture = ReportChainTests()
    fixture.setUp()
    agent = fixture.make_agent(BoxComparisonAgent)
    first = fixture.register(agent, 3, (0., .5, .5))
    agent.target_instance_id = first.iid
    agent._category = lambda _: "painting"
    agent._boxes = Mock(return_value=[dict(bbox=[80, 80, 240, 240])])
    agent._last_feed_info["is_keyframe"] = True
    first_command = agent._report_found(first.iid)
    before = bool(first.reported)
    agent._settle_report(fixture.observation(21, TURN))
    after_replaced = bool(first.reported)
    agent.now = 21
    agent._last_observation = fixture.observation(21, TURN)
    retry_command = agent._report_found(first.iid)
    agent._settle_report(fixture.observation(22, REPORT))
    after_ack = bool(first.reported)
    agent._settle_report(fixture.observation(22, REPORT))
    after_repeating_ack = int(agent._reported_count)
    again = agent._report_found(first.iid)
    sibling = fixture.register(agent, 4, (.18, .5, .5))
    sibling_open = sibling in agent.instance_store.available()
    metrics = dict(first_issued=first_command == REPORT, marked_before_ack=before,
                   marked_after_substituted_action=after_replaced, retried=retry_command == REPORT,
                   marked_after_ack=after_ack, claims_after_duplicate_ack=after_repeating_ack,
                   duplicate_issued=again == REPORT, separate_registered_sibling_available=sibling_open)
    expected = (True, False, False, True, True, 1, False, True)
    if tuple(metrics.values()) != expected:
        raise AssertionError(metrics)
    return metrics


def run_contracts():
    suites = [
        (ReportChainTests, ["canonical_memory_changes_after_observed_report_only",
                            "replaced_action_does_not_poison_later_attempt",
                            "reported_alias_cannot_spend_another_report",
                            "neighbor_same_appearance_remains_reportable",
                            "disjoint_boxes_in_same_view_protect_new_instance",
                            "identical_nearby_instances_require_unambiguous_live_regions",
                            "unrelated_observed_report_creates_no_claim",
                            "skipped_step_cannot_confirm_an_old_intent"]),
        (ReportRecoveryChecks, ["fresh_detection_executes_only_on_ack",
                                "uninformative_detection_blocks_both_report_routes",
                                "multiple_boxes_one_location_and_ambiguity"])
    ]
    tests = [suite("test_" + name) for suite, names in suites for name in names]
    result = unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(tests))
    if not result.wasSuccessful():
        raise RuntimeError("contract test failed")
    return [test._testMethodName for test in tests]


if __name__ == "__main__":
    print(json.dumps(dict(
        current_adapter=run_current(),
        isolated_production_contracts_passed=run_contracts(),
        images="generated synthetic RGB", paid_calls=0,
        scoring="no official credits inferred from synthetic acknowledgments"
    ), indent=2))
