"""Compare independent GPT responses for byte-identical observations across the two arms."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


def success(path):
    records=[json.loads(line) for line in path.read_text().splitlines()]
    successes={}
    for r in records:
        if r["event"]=="success":
            uid=r["hash"]
            assert uid not in successes and r["model"]=="gpt-4o-2024-11-20"
            successes[uid]=r["decision"]
    return successes


def main():
    here=Path(__file__).resolve().parent
    previous=here.parent/"gpt_region_20261128"
    total=Counter()
    homes=0
    for i in range(8):
        assert (here/"runs"/f"{i}_inspect.json").is_file()
        assert (previous/"runs"/f"{i}_gpt.json").is_file()
        slug=sha256(str(i).encode()).hexdigest()[:16]
        name=f"gpt_{slug}.jsonl"
        new=success(here/"private_events"/name)
        old=success(previous/"private_events"/name)
        common=old.keys() & new.keys()
        homes+=bool(common)
        total["identical_requests"]+=len(common)
        for key in common:
            a,b=old[key],new[key]
            total["mode_same"]+=a["mode"]==b["mode"]
            total["either_inspect"]+="inspect" in (a["mode"],b["mode"])
            total["region_suggestion_same"]+=a["region_id"]==b["region_id"]
            total["evidence_same"]+=a["target_evidence"]==b["target_evidence"]
    dst=here/"repeated_requests.json"
    result={"houses_with_identical_requests":homes,**dict(total)}
    dst.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
