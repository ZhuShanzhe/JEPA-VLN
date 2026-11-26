"""Prepare the normal evaluator's frozen one-episode development configuration."""
import ast
import inspect
import json
import os
from pathlib import Path
import sys
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT/"identity_stage5_confirmation"
sys.path.append(str(SOURCE))
from development_run import SOURCES, digest, load, replace_once, save

PILOTS = ROOT/"mapping_runtime/pilots"
PARENT = PILOTS/"identity_stage5_trusted_clock_r2"
STUDY = PILOTS/"identity_stage6_active_report"
RUNTIME = ("approach_agent.py","approach_motion.py","visual_witness.py")


def main():
    os.umask(0o077)
    prior = load(PARENT/"launch_receipt.json")
    assert (PARENT/"closed.json").exists()
    assert digest(PARENT/"panel.json") == prior["panel_sha256"]
    subprocess.run(["bash",str(HERE/"run_environment.sh"),"-m","unittest",
                    "test_approach_agent.ApproachTests","test_visual_witness"],check=True)
    save(STUDY/"panel.json",(PARENT/"panel.json").read_bytes())
    for name in SOURCES:
        parent = PARENT/name
        assert digest(parent) == prior["sha256"][str(parent)]
        src = parent.read_text().replace(PARENT.name,STUDY.name).replace("stage5_trusted_clock_r2","stage6_active_report")
        src = src.replace("18589","18593").replace("28971","28975")
        if name == "run_five.py":
            src = replace_once(src,"integration_agent:ConfirmedReferenceAgent","approach_agent:ApproachAgent")
            src = replace_once(src,repr(str(SOURCE/"integration_agent.py")),repr(str(HERE/"approach_agent.py")))
            src = replace_once(src,"adapter_root="+repr(str(SOURCE))+",module_roots=[",
                               "adapter_root="+repr(str(HERE))+",module_roots=["+repr(str(SOURCE))+", ")
        if name.endswith(".py"):
            ast.parse(src)
        save(STUDY/name,src)
    (STUDY/"transport/pilot_entry.sh").chmod(0o700)
    files = {path:v for path,v in prior["sha256"].items() if not Path(path).is_relative_to(PARENT)}
    assert all(digest(path)==v for path,v in files.items())
    verified=(*RUNTIME,"prepare_evaluation.py","run_environment.sh","test_approach_agent.py","test_visual_witness.py")
    for path in [*(HERE/n for n in verified),STUDY/"panel.json",*(STUDY/n for n in SOURCES)]:
        files[str(path)]=digest(path)
    manifest = dict(prior,sha256=files,config=sorted(set(files)-set(prior["checkpoints"])))
    preflight=STUDY/"preflight_receipt.json"
    content=json.dumps(manifest,sort_keys=True,indent=2)+"\n"
    if preflight.exists() and preflight.read_text()!=content:
        assert not any((STUDY/p).exists() for p in ("launch_receipt.json","spec.json","comparison"))
        preflight.rename(STUDY/("unlaunched_"+digest(preflight)+".json"))
    save(preflight,content)
    sys.path.insert(0,str(STUDY))
    import run_five
    before,sep,_=inspect.getsource(run_five.setup).partition("    service, env = p.service(PORT)")
    assert sep
    before=replace_once(before,'gate=p.data(D/"launch_receipt.json")','gate=p.data(D/"preflight_receipt.json")')
    scope=dict(vars(run_five))
    exec(compile(before+"    return lock\n","<freeze only>","exec"),scope)
    lock=scope["setup"]()
    assert lock["model"]=="gpt-4o-2024-11-20" and lock["protocol"]["budget"]==300
    launch=STUDY/"launch_receipt.json"
    if launch.exists():
        assert launch.samefile(preflight)
    else:
        os.link(preflight,launch)
    print("Verified the isolated evaluator entry and the 300-action development protocol.")


if __name__=="__main__":
    main()
