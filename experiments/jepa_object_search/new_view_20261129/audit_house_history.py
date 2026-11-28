"""List recorded use of building families before the new-view experiment; read no credentials or raw frames."""
from collections import defaultdict
import json
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DISK = ROOT.parent
ASSETS = DISK / "jepa_vln_data_20261014/raw/hssd-hab"
PATTERN = re.compile(rb"(?<![0-9a-zA-Z])1[0-9]{8}(?:_1[0-9]{8})?(?![0-9a-zA-Z])")


def main():
    available = {p.name.split(".")[0] for p in (ASSETS / "scenes").glob("*.scene_instance.json")}
    families = {h.split("_")[0] for h in available}
    used = defaultdict(set)
    def register(data, source):
        for tag in PATTERN.findall(data):
            family = tag.decode().split("_")[0]
            if family in families:
                used[family].add(source)
    # Check actual project output names, explicit split/episode manifests, and
    # recorded baseline runs. Omit downloaded asset listings and setup receipts.
    selected = [x for x in ROOT.iterdir() if x.is_dir() and
                x.name.startswith(("jepa_", "identity_", "pose_horizon_", "path_feedback_", "multiview_",
                                   "observation_route_", "local_target_", "prior_approach_", "approach_search_",
                                   "committed_inspect_", "gpt_", "route_execution_", "a_exploration",
                                   "training_round", "frontier_jepa", "box_sufficiency_"))]
    external = [DISK/"jepa_vln_baselines_20261015/results", DISK/"jepa_instance_bench_20261016/results",
                DISK/"jepa_vln_paper_bench_20261017/results"]
    for base in selected+external:
        if not base.exists():
            continue
        label = ("results/"+base.parent.name) if base in external else base.name
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git", "cache", "assets", "raw",
                         "weights", "downloads", "repo", "repos", "official", "third_party", "site-packages"}]
            directory = Path(root)
            for name in dirs + files:
                register(name.encode(), label)
            # JSON-lines model decisions in prior results name actual episodes.
            for name in files:
                f = directory/name
                if not name.endswith((".json", ".jsonl")):
                    continue
                if any(x in name.lower() for x in ("token","api","credential","key","source_inventory","asset","license","budget","download")):
                    continue
                manifest = ("cohort" in name or name in {"panel.json","natural_design.json","manifest.jsonl"} or
                            (base in external and name in {"results.jsonl","episodes.jsonl","episodes.json","results.json","metrics.jsonl"}))
                if manifest and f.stat().st_size<=32_000_000:
                    register(f.read_bytes(), label)
    split = ""
    for line in (ASSETS/"scene_splits.yaml").read_text().splitlines():
        if line.endswith(":"): split=line.strip()[:-1]
        if split=="val" and line.startswith("- "):
            register(line.encode(),"official_val_used_in_compact430")
    rows={h:sorted(used[h.split("_")[0]]) for h in sorted(available)}
    unused=[h for h,sources in rows.items() if not sources]
    result=dict(installed_scenes=len(available),unused_in_checked_records=unused,by_scene=rows,
                scope="Prior project named outputs and explicit research cohorts, previous baseline results, entire previously evaluated official validation group. Acquisition of raw meshes, rendering QA and material conversion do not train or evaluate a policy.")
    (HERE/"house_history_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(f"{len(available)} installed scenes; {len(unused)} have no recorded use in checked sources")
    print(" ".join(unused))


if __name__=="__main__":
    main()
