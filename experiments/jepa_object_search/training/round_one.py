"""Freeze prepared inputs or run the two audited first-round training arms."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def write_once(path, value):
    new = path.with_suffix(".tmp." + str(os.getpid()))
    new.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+"\n")
    try:
        os.link(new, path)
    finally:
        new.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze","check","train"])
    parser.add_argument("--round-dir", required=True, type=Path)
    args = parser.parse_args()
    folder = args.round_dir.resolve(strict=True)
    kit = Path(__file__).resolve().parent
    data = folder/"grid_final"
    text = folder/"final_queries_bge.npz"
    outputs = folder/"first_fit"
    lock = folder/"first_round.json"
    receipt = json.loads((data/"receipt.json").read_bytes())
    if receipt["schema"] != "most-branch-training/local-grid-v1" or receipt["index_sha256"] != sha(data/"index.jsonl"):
        raise ValueError("Final local-grid dataset fingerprint differs")
    rows = [json.loads(line) for line in (data/"index.jsonl").read_bytes().splitlines()]
    labels = {}
    for row in rows:
        path = (data/row["path"]).resolve(strict=True)
        if not path.is_relative_to(data) or sha(path)!=row["sha256"]:
            raise ValueError("Training input changed")
        group = "/".join((row["source"], row["split"], row["label_kind"]))
        labels[group] = labels.get(group,0)+1
    for split in ("train","dev"):
        for source, label in (("hssd","hssd_visible_evidence_delta"),("blender","new_sustained_exposure"),("blender","contextual_exposure")):
            if not labels.get(f"{source}/{split}/{label}"):
                raise ValueError("Expected both natural HSSD and blind/cued Blender decisions in both splits")
    manifest = dict(epochs=12, seed=17, batch=8, lr=0.001,
                    arms=["direct","jepa"], dataset_index=receipt["index_sha256"],
                    data_receipt=sha(data/"receipt.json"), texts=sha(text), groups=labels,
                    training_code=sha(kit/"trainer.py"), launch_code=sha(Path(__file__)),
                    visual_protocol=receipt["feature"])
    if (args.mode != "freeze" or lock.exists()) and json.loads(lock.read_bytes())!=manifest:
        raise ValueError("Round inputs, configuration, or code differ from their frozen versions")
    cmd=[sys.executable,"-B",str(kit/"trainer.py"),"--data",str(data),"--text-cache",str(text),
         "--epochs","12","--seed","17","--batch","8","--threads","2"]
    if args.mode in ("freeze","check"):
        subprocess.run(cmd+["--device","cpu","--check-only"],check=True)
        if args.mode=="freeze" and not lock.exists():
            write_once(lock,manifest)
        return
    # The armed launcher owns the process/accelerator locks. Recheck the frozen baseline
    # here too so a direct invocation cannot start the fit early.
    from launch_when_ready import checked_baseline
    if not checked_baseline(folder.parent)["ready"]:
        raise RuntimeError("Waiting for the validated native baseline")
    subprocess.run(cmd+["--device","cuda","--output",str(outputs)],check=True)
    result=json.loads((outputs/"receipt.json").read_bytes())
    if set(result["arms"])!={"direct","jepa"}:
        raise ValueError("Incomplete training arms")
    for arm, rec in result["arms"].items():
        if rec["checkpoint_sha256"]!=sha(outputs/f"{arm}.pt"):
            raise ValueError("Training checkpoint checksum mismatch")
    print("Finished and verified both local training arms.",flush=True)
    if (kit/"publish_round.py").is_file():
        proc=subprocess.run([sys.executable,str(kit/"publish_round.py"),str(outputs),"--publish"])
        write_once(folder/"publication_exit.json",dict(returncode=proc.returncode))
        if proc.returncode:
            print("Training completed. GitHub publication requires follow-up.",flush=True)


if __name__=="__main__":
    main()
