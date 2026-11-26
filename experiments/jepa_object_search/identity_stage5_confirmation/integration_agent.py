"""MOST-Nav development agent for synchronous verification with trusted references."""
from barrier_agent import BarrierAgent
from trusted_reference import TrustedReferenceMixin


class ConfirmedReferenceAgent(TrustedReferenceMixin, BarrierAgent):
    """A single report path, shared instance memory and original evaluator actions."""
