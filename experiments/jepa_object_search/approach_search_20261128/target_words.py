"""Read one requested object class from the public goal, keeping the full instruction in MOST."""
import re

from placement_prior import GROUPS


def requested_class(text, supported):
    goal = str(text).lower().strip()
    pieces = re.split(r"\b(?:find|locate|search for|look for|identify|report)\b", goal, maxsplit=1)
    phrase = re.split(r"\b(?:on|in|at|under|above|below|by|beside|near|next to|behind|among|with|without|that|which)\b|[,;.!?]", pieces[-1], maxsplit=1)[0]
    # Multiple target classes need the baseline's multi-target logic.
    if re.search(r"\b(?:or|and)\b", phrase):
        return None
    terms = {w:w for w in supported}
    for group, aliases in GROUPS.items():
        if group in supported:
            for term in aliases:
                terms[term] = group
    if "painting" in supported:
        terms.update({"artwork":"painting", "wall_art":"painting", "print":"painting"})
    matches = []
    for name, target in terms.items():
        words = name.replace("_", " ")
        pattern = re.escape(words[:-1]) + "(?:y|ies)" if words.endswith("y") else re.escape(words) + "(?:s|es)?"
        for match in re.finditer(r"\b" + pattern + r"\b", phrase):
            matches.append((match.end(), len(match.group()), target))
    if not matches:
        return None
    return max(matches)[2]
