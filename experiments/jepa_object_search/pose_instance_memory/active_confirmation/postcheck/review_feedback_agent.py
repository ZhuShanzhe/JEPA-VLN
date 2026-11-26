"""Optional unscored adapter. Use the V24 runtime paths plus this directory."""
from active_agent import ActiveFrontier
from review_ids import ReviewReferenceMixin


class ReviewFeedbackFrontier(ReviewReferenceMixin, ActiveFrontier):
    """Return usable identifiers for unknown references through native tools."""
