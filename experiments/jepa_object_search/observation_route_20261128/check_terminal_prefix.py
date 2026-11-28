"""Check coverage of complete strict routes under the unified four-primitive verification choice."""
from check_routes import MeasuredGrid
from observation_sites import propose_sites
from routes import observation_routes, score_observation_routes
from terminal_prefix import with_terminal_prefixes


def main():
    grid=MeasuredGrid()
    grid.obstacle[:]=False
    pose,anchor=(3.,3.,0.),(5.25,3.6)
    old=propose_sites(grid,pose,1.,[dict(anchor_xy=anchor)],budget=24)
    assert old and max(row["steps"] for row in old)>4
    prefix=lambda site:tuple(site["actions"][:4])
    # Force one duplicate by using an actual strict prefix as the continuation.
    base=observation_routes(grid,pose,1.,mode="verify",region_actions=prefix(old[0]),budget=24,anchor_xy=anchor)
    assert with_terminal_prefixes(base,[],grid,pose,1.,anchor,24) is base
    combined=with_terminal_prefixes(base,old,grid,pose,1.,anchor,24)
    choices=combined["candidates"]
    assert len(choices)==len({tuple(r["actions"]) for r in choices})
    for site in old:
        row=next(r for r in choices if tuple(r["actions"])==prefix(site))
        assert any(s["key"]==site["key"] and s["full_steps"]==site["steps"] for s in row["strict_terminals"])
    assert choices[0]["id"]=="continue" and choices[0]["strict_terminals"]
    assert all(len(row["actions"])<=4 for row in choices)
    assert all("strict_terminals" not in row for row in base["candidates"])
    novel = [r for r in choices if r["source"]=="terminal"]
    assert novel
    preferred = novel[0]
    found=score_observation_routes(combined,lambda plans:[float(p==preferred["actions"]) for p in plans])
    assert found["selected_id"]==preferred["id"]
    print(f"PASS: {len(old)} strict endpoints represented; {len(novel)} extra prefixes; exact no-endpoint parity")


if __name__=="__main__":
    main()
