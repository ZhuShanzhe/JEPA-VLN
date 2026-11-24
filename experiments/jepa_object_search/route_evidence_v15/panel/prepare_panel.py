"""Prepare separate, fixed-size MOST campaigns with independent replay. No API dispatch."""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys


KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[1]
RT = ROOT / "mapping_runtime"
BASE = RT / "pilots/jepa_round13_horizon_five"
FIRST = RT / "pilots/jepa_evidence_v15_first2"
ADAPTER = ROOT / "jepa_evidence_v15/adapter"
NATIVE = RT / "batch/campaigns/native_20261115"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1<<20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def put(path, data):
    if not isinstance(data, bytes):
        data = data.encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"frozen content changed: {path.name}")
    else:
        with path.open("xb") as f:
            f.write(data)


def write_json(path, value):
    put(path,json.dumps(value,sort_keys=True,indent=2)+"\n")


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"unexpected template segment: {before}")
    return text.replace(before, after)


def phase_details(phase):
    if phase == "first2":
        return FIRST, [1,14], 18567, 28949
    decision = read(KIT/"phase1_decision.json")
    assert decision["continue_phase2"] is True
    assert decision["first_summary_sha256"] == sha(FIRST/"five_comparison.json")
    assert (FIRST/"closed.json").is_file()
    return RT/"pilots/jepa_evidence_v15_next3", [21,32,39], 18568, 28950


def stage(phase):
    pilot, ids, mapper, relay = phase_details(phase)
    n = len(ids)
    pilot.mkdir(exist_ok=True)
    (pilot/"transport").mkdir(exist_ok=True)
    previous = read(BASE/"launch_receipt.json")
    source_panel = read(BASE/"panel.json")
    assert sha(BASE/"panel.json")==previous["panel_sha256"]
    rows=[r for r in source_panel["rows"] if r["ordinal"] in ids]
    assert [r["ordinal"] for r in rows] == ids
    write_json(pilot/"panel.json",dict(rows=rows, scope=phase))
    names = ["run_five.py","campaign.py","start_mapper.py","finish_five.py","pilot_failfast.py",
             "release_mapper.py","release_resources.py","transport/pilot_gateway.py",
             "transport/pilot_entry.sh","transport/race_tls.py"]
    for name in names:
        f = BASE/name
        assert sha(f) == previous["sha256"][str(f)]
        t = f.read_text().replace(BASE.name,pilot.name).replace("18566",str(mapper)).replace("28948",str(relay))
        t = t.replace("v13_horizon_verified_tls",f"v15_evidence_{phase}_tls").replace("v13_horizon_mapping_",f"v15_evidence_{phase}_mapping_")
        if name=="run_five.py":
            t = once(t,"jepa_pilot_agent:JepaFrontier","v15_agent:EvidenceFrontier")
            t = t.replace(str(ROOT/"jepa_env_v13/most_integration"), str(ADAPTER))
            t = once(t,"jepa_pilot_agent.py","v15_agent.py")
            t = once(t,'        for cycle in range(2):',f'        assert [row["ordinal"] for row in panel] == {ids!r}\n        for cycle in range(2):')
            t = t.replace('"/5"', repr(f"/{n}"))
        if name=="campaign.py":
            t = once(t,"total=5",f"total={n}")
            t = once(t,'    assert health(28937)',f'    assert [x["ordinal"] for x in r.p.data(D/"panel.json")["rows"]] == {ids!r}\n    assert health(28937)')
        if name=="pilot_failfast.py":
            old=t
            t = once(t,"jepa_pilot_agent:JepaFrontier","v15_agent:EvidenceFrontier")
            target=pilot/name
            if target.exists() and target.read_text()==old and not (pilot/"preflight_receipt.json").exists():
                target.write_text(t)
        if name=="finish_five.py":
            t = once(t,"[1,14,21,32,39]",repr(ids))
            t = once(t,"5 preselected exploratory pairs, original native references",f"{n} frozen {phase} development pairs, original native references")
            t = t.replace("jepa_horizon","jepa_evidence")
        if name=="release_resources.py":
            t = t.replace("== 5",f"== {n}")
        if name=="release_mapper.py":
            t = once(t,"assert len(panel)==5",f'assert [r["ordinal"] for r in panel]=={ids!r}')
            t = once(t,'p.data(D/"paired_results.json")["pairs"]','p.data(D/"five_comparison.json")["cases"]')
        if name.endswith(".py"):
            ast.parse(t)
        put(pilot/name,t)
    (pilot/"transport/pilot_entry.sh").chmod(0o700)
    print("staged", phase, ids)


def freeze(phase):
    pilot, ids, _, _ = phase_details(phase)
    smoke = ADAPTER/"no_api_check.json"
    s = read(smoke)
    assert s["passed"] is True and s["no_paid_calls"] is True
    files = s["sha256"]
    checkpoints = s["checkpoints"]
    for p,h in files.items():
        assert Path(p).is_absolute() and sha(p)==h,p
    assert checkpoints and set(checkpoints)<=set(files) and str(ADAPTER/"v15_agent.py") in files
    assert any(str(ROOT/"jepa_evidence_v15/train") in x for x in checkpoints)
    previous = read(BASE/"launch_receipt.json")
    config={*map(str,pilot.glob("*.py")),*map(str,(pilot/"transport").glob("*.py")),
            str(pilot/"transport/pilot_entry.sh"),str(smoke),str(pilot/"panel.json"),str(Path(__file__).resolve()),
            *(set(files)-set(checkpoints))}
    config.update(str(ROOT/"private_runtime/most_vlm"/n) for n in ("gateway.py","pool.py","deadline_http.py"))
    config.update(str(KIT.parent/"analysis"/n) for n in ("assess_panel.py","protocol.json"))
    if phase=="next3":
        config.add(str(KIT/"phase1_decision.json"))
    rows=read(pilot/"panel.json")["rows"]
    assert [r["ordinal"] for r in rows]==ids
    for row in rows:
        assert read(row["ref"])["result_sha256"]==row["result_sha256"]
    gate=dict(panel_sha256=sha(pilot/"panel.json"),paired_authorized=True,live_nonpaid_smoke_pass=True,
              no_future_leakage=True,worker_python=previous["worker_python"],checkpoints=checkpoints,
              config=sorted(config),sha256={p:sha(p) for p in sorted(set(checkpoints)|config)},
              native_ref_sha256=[r["result_sha256"] for r in rows])
    pre=pilot/"preflight_receipt.json"
    write_json(pre, gate)
    sys.path.insert(0,str(pilot))
    import run_five as run
    body=inspect.getsource(run.setup).split("    service, env = p.service(PORT)")[0]+"    return lock\n"
    body=once(body,'gate=p.data(D/"launch_receipt.json")','gate=p.data(D/"preflight_receipt.json")')
    ctx=dict(vars(run))
    exec(compile(body,"<pre-service lock initialization>","exec"),ctx)
    ctx["setup"]()
    import campaign
    assert campaign.health(28937) and run.p.service(18555)
    dest=pilot/"launch_receipt.json"
    if dest.exists():
        assert dest.samefile(pre)
    else:
        os.link(pre,dest)
    campaign.gate()
    print("selection, model lock, references, local services checked", ids)


if __name__=="__main__":
    os.umask(0o077)
    pa=argparse.ArgumentParser()
    pa.add_argument("action",choices=["stage","freeze"])
    pa.add_argument("phase",choices=["first2","next3"])
    args=pa.parse_args()
    {"stage":stage,"freeze":freeze}[args.action](args.phase)
