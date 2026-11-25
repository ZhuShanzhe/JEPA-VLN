"""Run the installed adapter and synthetic tests with all IP socket connections denied."""

import hashlib
import json
from pathlib import Path
import socket
import sys
import unittest


def main():
    tried = []

    def no_remote(event, args):
        if event == "socket.getaddrinfo" or (
            event == "socket.connect" and args[0].family in (socket.AF_INET, socket.AF_INET6)
        ):
            tried.append(event)
            raise RuntimeError("External connections disabled for this evaluation")

    sys.addaudithook(no_remote)
    import report_agent
    from agents.nav_agent import NavAgent

    assert issubclass(report_agent.ReportFrontier, NavAgent)
    directory = Path(__file__).resolve().parent
    groups = {}
    suite = unittest.TestSuite()
    minimum = {"test_appearance": 8, "test_identity": 7, "test_exploration_memory": 6, "test_chain": 18}
    for module_name, required in minimum.items():
        module_suite = unittest.defaultTestLoader.loadTestsFromName(module_name)
        groups[module_name] = module_suite.countTestCases()
        assert groups[module_name] >= required, module_name
        suite.addTests(module_suite)
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    if not results.wasSuccessful() or results.skipped or tried or results.testsRun != sum(groups.values()):
        raise SystemExit("Incomplete offline verification")
    sources = sorted(p for p in directory.glob("*.py") if p.name != "replay_appearance.py")
    digest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    receipt = dict(
        entry="report_agent:ReportFrontier",
        passed=results.testsRun,
        skipped=len(results.skipped),
        groups=groups,
        ip_attempts=len(tried),
        paid_calls=0,
        scope="Synthetic observations and recorded action acknowledgements, no live navigation run.",
        source_sha256=digest,
    )
    Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
