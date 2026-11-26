"""Feedback for invalid native review references. Kept separate from the scored agent."""


class ReviewReferenceMixin:
    def _tool_commit_candidates(self, reviews, label=""):
        if not isinstance(reviews,(list,tuple)) or not reviews:
            return super()._tool_commit_candidates(reviews,label)
        known, invalid = [], []
        for row in reviews[:16]:
            cid = str(row.get("candidate_id") or "") if isinstance(row,dict) else ""
            if cid in self._proposal_queue:
                # Delegate known IDs without filtering by verdict, state or
                # repetition. The native review path owns those rules.
                known.append(row)
            else:
                invalid.append(dict(candidate_id=cid,reason="unknown_candidate_id" if cid else "missing_candidate_id"))
        if not invalid:
            return super()._tool_commit_candidates(reviews,label)
        # Preserve the original verdict/geometry path for every valid row. Never
        # rewrite numeric mask IDs, infer a frame, or borrow a later candidate.
        result = super()._tool_commit_candidates(known,label) if known else None
        if result is not None and "error" in result:
            return result
        pending = [dict(candidate_id=c,frame_id=r.get("frame_id")) for c,r in self._proposal_queue.items()
                   if r.get("status")=="pending"]
        recent = [dict(candidate_id=c,frame_id=r.get("frame_id"),status=r.get("status"))
                  for c,r in self._proposal_queue.items() if r.get("status") and r["status"]!="pending"][-4:]
        advice = ("Use an exact pending candidate_id after inspecting its original panel. "
                  "A mask_id identifies a region within one frame." if pending else
                  "No pending candidate is available. Continue observation; completed reviews are listed below.")
        feedback = dict(invalid_reviews=invalid,pending_candidates=pending[:12],pending_count=len(pending),
                        recent_nonpending=recent, next=advice)
        self._ev_log("invalid_review_reference", invalid_reviews=invalid, pending_count=len(pending),
                     delegated_known_rows=len(known))
        if result is None:
            return dict(error=dict(code="NO_REVIEWABLE_CANDIDATE",message=advice,**feedback))
        result["review_reference_feedback"] = feedback
        return result
