"""Run the existing navigation controller with a two-view instance memory and refined clarity."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "reactive_view_20261129"
sys.path.insert(0, str(PREVIOUS))

from inspection_session import ReactiveInspection


def load_inspection(device="cpu"):
    from memory_refit import RefinedMemory
    prior = ReactiveInspection.load(device)
    return ReactiveInspection(prior.initial, prior.reroute, prior.encoder, verifier=RefinedMemory)
