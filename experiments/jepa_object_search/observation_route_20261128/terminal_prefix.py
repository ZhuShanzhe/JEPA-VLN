"""Include primitive prefixes of previously certified viewing sites in cue verification."""
from observation_sites import _Map
from routes import _centered_evidence, _end


def with_terminal_prefixes(proposal, terminal_sites, grid, pose, scale, anchor, budget):
    # Do not touch the frozen geometry, vote or tie order on no-terminal inputs.
    if not terminal_sites or budget <= 0:
        return proposal
    if proposal["mode"] != "verify":
        raise ValueError("strict viewing sites require a localized verification cue")
    online = _Map(grid,scale)
    start = (float(pose[0]*scale),float(pose[1]*scale),float(pose[2]))
    candidates = [dict(row) for row in proposal["candidates"]]
    by_actions = {tuple(row["actions"]):row for row in candidates}
    used = False
    for site in terminal_sites:
        full = tuple(int(i) for i in site["actions"])
        prefix = full[:min(4,budget)]
        if not prefix or any(a not in (1,2,3) for a in full):
            continue
        end = _end(online,start,prefix)
        if end is None:
            continue
        used = True
        row = by_actions.get(prefix)
        if row is None:
            name = "terminal/"+"".join(str(a) for a in prefix)
            row = dict(id=name,actions=list(prefix),source="terminal",site_xy=list(end[:2]),
                       face_yaw=end[2],new_surfaces=0,surface_keys=[],frontier_keys=[],
                       geometry_score=_centered_evidence(online,end,anchor),model_score=None)
            candidates.append(row)
            by_actions[prefix]=row
        row.setdefault("strict_terminals",[]).append(dict(key=site["key"],full_steps=len(full),
            site_xy=list(site["site_xy"]),site_yaw=site["site_yaw"]))
    if not used:
        return proposal
    geometric = max(candidates,key=lambda row:row["geometry_score"])["id"]
    # The regional executor remains the host fallback when nothing useful is
    # returned; an unsafe regional primitive is never introduced into model scores.
    safe_default = proposal["continue_id"] if any(r["id"]=="continue" for r in candidates) else geometric
    return dict(proposal,candidates=candidates,geometry_id=geometric,
                selected_id=safe_default,status="offered" if len(candidates)>1 else "single_certified_prefix")
