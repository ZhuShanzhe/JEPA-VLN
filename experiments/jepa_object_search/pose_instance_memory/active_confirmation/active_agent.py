"""MOST-Nav with instance confirmation views and a region-to-registration audit trail."""
from candidate_evidence import CandidateEvidenceMixin
from confirm_view import ConfirmViewMixin
from repair_agent import RepairFrontier


class ActiveFrontier(ConfirmViewMixin, CandidateEvidenceMixin, RepairFrontier):
    """Run both extensions on the same native map, candidate registry and action stream."""
