from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def main(python, project, root):
    plan = json.loads((project / "jepa_separation_v8_20261123/primitives/plan.json").read_text())["houses"]
    assert sum(map(len, plan.values())) == len(set(sum(plan.values(), []))) == 20
    (root / "logs").mkdir(parents=True, exist_ok=True)
    def one(item):
        split, house = item
        with (root / "logs" / (house + ".log")).open("ab") as f:
            subprocess.run([python, str(root / "collect_views.py"), str(project), str(root), house, split],
                stdout=f, stderr=f, check=True, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                                        MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet"))
        print(split, house, flush=True)
    with ThreadPoolExecutor(4) as ex:
        for split, houses in plan.items():
            list(ex.map(one, [(split,h) for h in houses]))
            rows = [json.loads(f.read_bytes()) for f in sorted((root / "rows" / split).rglob("*.json"))]
            assert {r["house"] for r in rows} == set(houses)
            tmp = root / (split + ".index.pending")
            tmp.write_text(json.dumps(rows))
            tmp.replace(root / (split + ".index.json"))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
