"""Construct a private, exact frozen view for the original multi-house evaluator."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import sys
import tempfile

project = Path(os.environ["NAV_ROOT"])
scenes = project / "frontier_scenes"
release = scenes / "most_release"
sys.path.insert(0, str(release))
from evaluation.dataset import load_custom_dataset

manifest = scenes / "acceptance/most_many_category_attribute_frozen.jsonl"
digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
assert digest == json.loads((scenes/"acceptance/data_audit.json").read_text())["selected_manifest_sha256"]
rows = [json.loads(line) for line in manifest.read_text().splitlines()]
by_id = {r["episode_id"]:r for r in rows}
assert len(rows) == len(by_id) == 300
assert Counter(r["condition"] for r in rows) == {"many_category":150,"many_attribute":150}
runtime = project / "mapping_runtime"
final = runtime / "frozen_manyon"
src = release / "benchmark_dataset/semantic"

def check(folder):
    seen = set()
    for house in sorted(folder.iterdir()):
        original = json.loads((src/house.name/"episodes.json").read_text())
        subset = json.loads((house/"episodes.json").read_text())
        assert subset["episodes"] == [e for e in original["episodes"] if e["episode_id"] in by_id]
        for k in ("queries.json","goals.json","goal_labels.json"):
            assert (house/k).resolve(strict=True) == (src/house.name/k).resolve(strict=True)
        data = load_custom_dataset(*(str(house/(name+".json")) for name in ("episodes","queries","goals")))
        for ep in data:
            metadata = by_id[ep.episode_id]
            assert ep.scene_id == metadata["scene"] == house.name and ep.target_count == metadata["quota"]
            assert ep.episode_id not in seen
            seen.add(ep.episode_id)
    assert seen==set(by_id) and len(list(folder.iterdir()))==36

def main():
    if final.exists():
        check(final)
    else:
        with tempfile.TemporaryDirectory(dir=runtime) as tmp:
            target=Path(tmp)/"data"
            target.mkdir()
            for house in sorted(src.iterdir()):
                if not house.is_dir():
                    continue
                data=json.loads((house/"episodes.json").read_text())
                data["episodes"]=[e for e in data["episodes"] if e["episode_id"] in by_id]
                dest=target/house.name
                dest.mkdir()
                (dest/"episodes.json").write_text(json.dumps(data,ensure_ascii=False)+"\n")
                for k in ("queries.json","goals.json","goal_labels.json"):
                    (dest/k).symlink_to(house/k)
            check(target)
            target.rename(final)
    # An all-category task outside the 300 frozen trials, selected solely by stable ID.
    ids=[e["episode_id"] for p in src.glob("*/episodes.json")
         for e in json.loads(p.read_text())["episodes"] if e["query_type"]=="all_category"]
    probe=min(ids)
    assert probe not in by_id
    print(json.dumps(dict(episodes=len(rows),houses=36,sha256=digest,engineering_episode=probe)))

if __name__=="__main__":
    main()
