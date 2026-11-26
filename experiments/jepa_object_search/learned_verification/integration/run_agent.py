"""Project agent with exploration, learned inspection and native candidate review."""
from candidate_scout import CandidateScoutMixin
from review_feedback_agent import ReviewFeedbackFrontier
from verify_agent import VerifyActionMixin


class LearnedVerificationFrontier(VerifyActionMixin,CandidateScoutMixin,ReviewFeedbackFrontier):
    """Use a learned viewpoint score to acquire evidence for the native reviewer."""
