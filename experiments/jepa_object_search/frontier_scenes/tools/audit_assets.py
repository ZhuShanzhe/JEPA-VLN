"""Check the released annotations and official evaluator without using HM3D geometry."""
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
release = root / "most_release"
out = root / "acceptance"
out.mkdir(exist_ok=True)
sys.path.insert(0, str(release))
from evaluation.dataset import load_custom_dataset
from evaluation.main.metrics import aggregate_metrics

def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

counts = Counter()
rows = []
ids = set()
query_ids = set()
houses = set()
no_text = []
missing_viewpoints = {}
quota_viewpoints_shortfall = []
quota_distribution = Counter()
pool_surplus = Counter()
for folder in sorted((release / "benchmark_dataset/semantic").iterdir()):
    if not folder.is_dir():
        continue
    files = [str(folder / (name + ".json")) for name in ("episodes", "queries", "goals")]
    raw = json.loads(Path(files[0]).read_text())["episodes"]
    episodes = load_custom_dataset(*files)
    assert len(raw) == len(episodes)
    for meta, episode in zip(raw, episodes):
        assert meta["episode_id"] == episode.episode_id and episode.episode_id not in ids
        assert meta["query_id"] not in query_ids and meta["scene_id"] == episode.scene_id == folder.name
        ids.add(episode.episode_id)
        query_ids.add(meta["query_id"])
        houses.add(episode.scene_id)
        counts[meta["track"]+"_"+meta["target_mode"]] += 1
        if meta["query_type"] in ("many_category", "many_attribute"):
            quota = meta["many_target_count"]
            assert episode.target_count == quota and quota >= 2
            if not (any(episode.natural_language) or meta.get("instruction")):
                no_text.append(episode.episode_id)
            unavailable = [x for x in episode.target_object_ids if not episode.view_points.get(x)]
            if unavailable:
                missing_viewpoints[episode.episode_id] = unavailable
            capacity = sum(episode.target_cluster_sizes.get(x, 1) for x in episode.target_object_ids
                           if episode.view_points.get(x))
            quota_distribution[quota] += 1
            pool_surplus["candidate_capacity_exceeds_quota" if capacity > quota else (
                         "candidate_capacity_equals_quota" if capacity == quota else "shortfall")] += 1
            if capacity < quota:
                quota_viewpoints_shortfall.append(episode.episode_id)
            rows.append(dict(scene=episode.scene_id, episode_id=episode.episode_id, query_id=meta["query_id"],
                             condition=meta["query_type"], quota=quota))
assert len(houses) == 36 and len(ids) == len(query_ids) == 750 and sorted(counts.values()) == [150]*5
assert len(rows) == 300
frozen = out / "most_many_category_attribute_frozen.jsonl"
frozen.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in sorted(rows, key=lambda r:r["episode_id"])))
synthetic = [dict(target_mode="many", task_type="mos", sr=v, f1=v, coverage=v, required_target_count=2,
                  true_positive_count=2*v, false_positive_count=1-v) for v in (0,1)]
m = aggregate_metrics(synthetic)
assert m["sr"] == .5 and m["f1"] == .5 and abs(m["micro_precision_many"] - 2/3) < 1e-8
release_files = [p for p in sorted(release.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
source_hashes = out / "most_release_files.sha256"
source_hashes.write_text("".join(digest(p) + "  " + p.relative_to(release).as_posix() + "\n" for p in release_files))
base = root.parent / "a_exploration/official/hssd_v025"
hssd = {}
splits = {}
for split in ("train", "val"):
    files = sorted((base / split / "content").glob("*.json.gz"))
    splits[split] = {p.name.split(".")[0] for p in files}
    n = 0
    for p in files:
        with gzip.open(p, "rt") as f:
            n += len(json.load(f)["episodes"])
    hssd[split] = dict(houses=len(splits[split]), episodes=n)
assert not (splits["train"] & splits["val"])
assert hssd == {"train":{"houses":122,"episodes":120552}, "val":{"houses":42,"episodes":1248}}
result = dict(status="pass", zip_sha256=digest(root / "downloads/most_release.zip"),
              houses=len(houses), episodes=len(ids), queries=len(query_ids), conditions=counts,
              many_category_attribute_episodes=len(rows), selected_houses=len({r["scene"] for r in rows}),
              selected_manifest_sha256=digest(frozen), release_files_sha256=digest(source_hashes),
              original_file_count=len(release_files), scorer_synthetic_macro_sr=m["sr"],
              scorer_synthetic_micro_precision=m["micro_precision_many"], hssd_v025=hssd,
              many_without_text=len(no_text), many_with_missing_candidate_viewpoints=len(missing_viewpoints),
              many_with_viewpoint_capacity_below_quota=len(quota_viewpoints_shortfall),
              many_quota_distribution=quota_distribution, many_cluster_capacity_comparison=pool_surplus,
              hssd_train_val_disjoint=True,
              limits="MOST simulation requires licensed HM3D scene geometry/semantics. Scorer test is synthetic.")
(out / "data_audit.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
