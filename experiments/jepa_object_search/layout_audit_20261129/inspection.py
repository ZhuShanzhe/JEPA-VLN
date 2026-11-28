"""Load the layout-aware local inspection controller for metric RGB-D or live MOST map updates."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/name) for name in (
    "layout_map_bridge_20261129", "layout_detour_v2_20261129", "clarity_fix_20261129",
    "refined_confirmation_20261129", "reactive_view_20261129")]


def load_layout_inspection(device="cpu", native_map=False):
    from refined_inspection import load_inspection
    from strict_rescue import StrictRescueMemory
    from grid_policy import GridPlanner
    base = load_inspection(device)
    base.verifier = StrictRescueMemory
    if native_map:
        from layout_inspection import bind_local_inspection
        return bind_local_inspection(base, planner_factory=GridPlanner)
    base.reroute = GridPlanner(base.reroute)
    return base
