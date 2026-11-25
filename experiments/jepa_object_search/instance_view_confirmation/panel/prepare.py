"""Prepare and freeze two strictly paired MOST episodes. Paid dispatch is separate."""
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
BASE = RT / "pilots/jepa_evidence_v15_first2"
PILOT = RT / "pilots/jepa_confirm_v18_first2"
AGENT = ROOT / "jepa_confirm_v18/agent"
V16 = ROOT / "jepa_variable_v16"
NAMES = ["run_five.py", "campaign.py", "start_mapper.py", "finish_five.py", "pilot_failfast.py",
         "release_mapper.py", "release_resources.py", "transport/pilot_gateway.py",
         "transport/pilot_entry.sh", "transport/race_tls.py"]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_bytes())


def put(p, data):
    if p.exists():
        assert p.read_bytes() == data, p.name
    else:
        with p.open("xb") as f:
            f.write(data)


def write_json(p, value):
    put(p,(json.dumps(value,indent=2,sort_keys=True)+"\n").encode())


def replace_once(s, before, after):
    assert s.count(before)==1, before
    return s.replace(before,after)


def stage():
    previous=read(BASE/"launch_receipt.json")
    assert sha(BASE/"panel.json")==previous["panel_sha256"]
    assert (BASE/"closed.json").exists()
    PILOT.mkdir(exist_ok=True)
    (PILOT/"transport").mkdir(exist_ok=True)
    rows=read(BASE/"panel.json")["rows"]
    assert [(r["ordinal"],r["quota"]) for r in rows]==[(1,2),(14,3)]
    write_json(PILOT/"panel.json",{"rows":rows,"scope":"v18 two paired development tasks"})
    for name in NAMES:
        orig=BASE/name
        assert sha(orig)==previous["sha256"][str(orig)]
        t=orig.read_text().replace(BASE.name,PILOT.name).replace("18567","18569").replace("28949","28951")
        t=t.replace("v15_evidence_first2","v18_confirm_first2")
        t=t.replace("v15_agent:EvidenceFrontier","confirm_agent:ConfirmFrontier")
        if name=="run_five.py":
            t=t.replace(str(ROOT/"jepa_evidence_v15/adapter"),str(AGENT)).replace("/v15_agent.py","/confirm_agent.py")
            assert str(AGENT/"confirm_agent.py") in t
        if name=="finish_five.py":
            t=t.replace("jepa_evidence","jepa_confirm")
        if name.endswith(".py"):
            ast.parse(t)
        put(PILOT/name,t.encode())
    (PILOT/"transport/pilot_entry.sh").chmod(0o700)


def freeze():
    stage()
    smoke=AGENT/"no_api_check.json"
    s=read(smoke)
    assert s["passed"] and s["no_paid_calls"], "new production adapter must pass offline checks"
    for path,digest in s["sha256"].items():
        assert Path(path).is_absolute() and sha(path)==digest,path
    assert str(AGENT/"confirm_agent.py") in s["sha256"]
    previous=read(BASE/"launch_receipt.json")
    prior_smoke=read(ROOT/"jepa_evidence_v15/adapter/no_api_check.json")
    ck=sorted(set(prior_smoke["checkpoints"]) | set(s.get("checkpoints",[])))
    configs={str(PILOT/n) for n in NAMES}
    configs.update(str(p) for p in AGENT.glob("*.py"))
    configs.update(s["sha256"])
    configs.update(prior_smoke["sha256"])
    configs.update(str(V16/"adapter"/n) for n in ["variable_agent.py","variable_routes.py","variable_policy.py",
                   "delayed_forks.py","variable_worker.py"])
    configs.update(map(str,[V16/"model/variable_evidence.py",V16/"model/calibration.json",smoke,PILOT/"panel.json",Path(__file__)]))
    configs.update(str(ROOT/"private_runtime/most_vlm"/n) for n in ("gateway.py","pool.py","deadline_http.py"))
    hashes={path:sha(path) for path in sorted(configs | set(ck))}
    assert all(hashes[path]==h for path,h in prior_smoke["sha256"].items())
    gate=dict(panel_sha256=sha(PILOT/"panel.json"), paired_authorized=True,live_nonpaid_smoke_pass=True,
              no_future_leakage=True,worker_python=previous["worker_python"],checkpoints=ck,
              config=sorted(configs-set(ck)),sha256=hashes,
              native_ref_sha256=[r["result_sha256"] for r in read(PILOT/"panel.json")["rows"]])
    pre=PILOT/"preflight_receipt.json"
    write_json(pre,gate)
    sys.path.insert(0,str(PILOT))
    import run_five as r
    prefix=inspect.getsource(r.setup).split("    service, env = p.service(PORT)")[0]+"    return lock\n"
    prefix=replace_once(prefix,'gate=p.data(D/"launch_receipt.json")','gate=p.data(D/"preflight_receipt.json")')
    scope=dict(vars(r))
    exec(compile(prefix,"<native reference validation>", "exec"),scope)
    scope["setup"]()
    dest=PILOT/"launch_receipt.json"
    if dest.exists():
        assert pre.samefile(dest)
    else:
        os.link(pre,dest)
    import campaign
    campaign.gate()
    print("Two paired cases frozen; source, native references, camera protocol and local services verified.")


if __name__=="__main__":
    os.umask(0o077)
    {"stage":stage,"freeze":freeze}[sys.argv[1]]()
