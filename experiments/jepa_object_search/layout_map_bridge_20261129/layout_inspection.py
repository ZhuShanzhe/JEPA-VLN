"""Native mapper adapter for the project's calibrated RGB-D local inspection host."""
from pathlib import Path
import sys

from most_layout_bridge import MostLayoutInspection


def bind_local_inspection(inspection, planner_factory=None):
    """Use existing initial selector, encoder and verifier, with the trained map-aware rerouter.

    Create a fresh binding per local request. observe() and move(action) supply a synchronized
    native observation plus local levelled 192x192 RGB-D, with source-relative metric odometry.
    """
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root/"layout_detour_v2_20261129"))
    from grid_policy import GridPlanner
    from inspection_session import ReactiveInspection
    planner = (planner_factory or GridPlanner)(inspection.reroute)
    return MostLayoutInspection(ReactiveInspection(inspection.initial, planner, inspection.encoder,
                                                 verifier=inspection.verifier))
