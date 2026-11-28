"""Freeze current-frame encodings. Future whole-frame and instance encodings are training targets only."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pose_horizon_20261128"))
from long_data import FrozenImageEncoder


def encode(folder, dest, encoder):
    with np.load(folder / "controls.npz", allow_pickle=False) as raw:
        rows = {key: raw[key] for key in raw.files}
    src = [f"{i}_source.jpg" for i in rows["source_index"]]
    crop = [f"{i}_crop.jpg" for i in rows["source_index"]]
    names = sorted((set(src) | set(crop) | set(rows["future_file"])) - {""})
    groups = {}
    vectors = {}
    for name in names:
        groups.setdefault(hashlib.sha256((folder / name).read_bytes()).digest(), []).append(name)
    keys = list(groups)
    for start in range(0, len(keys), 64):
        batch = keys[start:start + 64]
        imgs = []
        for key in batch:
            with Image.open(folder / groups[key][0]) as img:
                imgs.append(np.array(img.convert("RGB")))
        for key, vec in zip(batch, encoder.encode(imgs)):
            for name in groups[key]:
                vectors[name] = vec
    zero = np.zeros(1024, np.float16)
    vectors[""] = zero
    rows["context"] = np.stack([np.r_[vectors[s], vectors[c]] for s, c in zip(src, crop)])
    rows["future"] = np.stack([vectors[f] for f in rows["future_file"]])
    labels = {}
    if "future_box" in rows:
        for name, box in zip(rows["future_file"], rows["future_box"]):
            x, y, r, b = map(int, box)
            if name and 0 <= x < r and 0 <= y < b:
                labels[(name, x, y, r, b)] = None
    keys = list(labels)
    for start in range(0, len(keys), 64):
        batch = keys[start:start + 64]
        imgs = []
        for name, x, y, r, b in batch:
            with Image.open(folder / name) as image:
                imgs.append(np.asarray(image.convert("RGB").crop((x, y, r, b))))
        labels.update(zip(batch, encoder.encode(imgs)))
    roi = np.zeros_like(rows["future"])
    if "future_box" in rows:
        for i, (name, box) in enumerate(zip(rows["future_file"], rows["future_box"])):
            key = (name, *map(int, box))
            if key in labels:
                roi[i] = labels[key]
    rows["future_roi"] = roi
    temp = dest.with_suffix(".partial.npz")
    np.savez_compressed(temp, **rows)
    temp.replace(dest)
    print(folder.name, len(src), len(labels), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-houses", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val", "blind"))
    args = parser.parse_args()
    os.umask(0o077)
    torch.set_num_threads(2)
    dest = HERE / "private_data"
    dest.mkdir(exist_ok=True)
    pattern = (args.split or "*") + "_*"
    encoder = FrozenImageEncoder(HERE.parent)
    while True:
        render = HERE / ("private_render_blind60" if args.split == "blind" else "private_render")
        for marker in sorted(render.glob(pattern + "/controls.npz")):
            if args.split is None and marker.parent.name.startswith("blind_"):
                continue
            file = dest / (marker.parent.name + ".npz")
            if not file.exists():
                encode(marker.parent, file, encoder)
        if len([p for p in dest.glob(pattern + ".npz") if ".partial" not in p.name]) >= args.wait_houses:
            break
        time.sleep(3)
