"""MOST-Nav agent with report recovery and delayed pose review."""
from delayed_pose import DelayedPoseMixin
from report_recovery import ReportRecoveryMixin
from trace_agent import TraceFrontier


class RepairFrontier(ReportRecoveryMixin, DelayedPoseMixin, TraceFrontier):
    """Share the native instance store and one acknowledged action channel."""
