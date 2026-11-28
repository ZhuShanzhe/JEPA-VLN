"""Use short-horizon visual evidence inside the priority region selected from map memory."""
from search_policy import PriorSearchMemory, probability


class PriorEvidenceMemory(PriorSearchMemory):
    def choose(self, candidates):
        idx, log = super().choose(candidates)
        if not all(c.get("evidence") is not None for c in candidates):
            return idx, log
        rows = log["candidates"]
        available = [i for i, r in enumerate(rows) if not r["blocked"]] or list(range(len(rows)))
        fresh = [i for i in available if rows[i]["unseen"] >= .35]
        if log["phase"] == "prior_search":
            eligible = [i for i in fresh if rows[i]["prior"] >= log["high_cutoff"] and rows[i]["prior"] > 0.]
        else:
            eligible = fresh or available
        for row, candidate in zip(rows, candidates):
            rate = probability(candidate["evidence"])
            # VariableEvidence already divides its cumulative log survival by length/6.
            row["forecast_6_steps"] = rate
            row["score"] = .55 * row["score"] + .45 * rate * (.25 + .75 * row["unseen"])
        return max(eligible, key=lambda i: rows[i]["score"]), log
