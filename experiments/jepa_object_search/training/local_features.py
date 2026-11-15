"""Re-encode decision and future RGB separately into a 2-time × 2 × 2 V-JEPA token grid."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

from data_prep import read, records, rgb, sha, within


class GridEncoder:
    def __init__(self, root, device):
        sys.path.insert(0, str(root/"frontier_jepa"))
        from vision import ContextEncoder
        self.backbone = ContextEncoder(root, device)
        self.cache = {}

    @torch.inference_mode()
    def __call__(self, frames):
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or not len(frames):
            raise ValueError("Expected nonempty uint8 RGB history")
        indices = np.linspace(0, len(frames)-1, 4, dtype=np.int64)
        raw = frames[indices]
        key = (raw.shape, hashlib.sha256(raw.tobytes()).digest())
        if key not in self.cache:
            resized = np.stack([np.asarray(Image.fromarray(x).resize((384,384), Image.Resampling.BILINEAR))
                                for x in raw])
            b = self.backbone
            video = torch.from_numpy(resized).permute(3,0,1,2)[None].to(b.device).float().div(255.)
            token = b.encoder((video-b.mean)/b.std).float()
            if token.shape != (1, 2*24*24, 768) or not torch.isfinite(token).all():
                raise ValueError(f"Unexpected four-frame V-JEPA token lattice {token.shape}")
            # Upstream tubelet size 2, patch size 16, token flatten order time, row, column.
            grid = token.reshape(2,2,12,2,12,768).mean(dim=(2,4)).reshape(8,768)
            self.cache[key] = grid.cpu().numpy()
        return self.cache[key]


def npz_rgb(path):
    with np.load(path, allow_pickle=False) as z:
        return z["rgb"]


def clips(root, sid, n, extra_rows):
    base = root/"a_exploration/paired_branches_v2"
    if sid.startswith("hssd/"):
        key = sid.removeprefix("hssd/")
        if key.startswith("report-pair"):
            folder, history = key.split("/")
            k = int(folder.removeprefix("report-pair"))
            r = root/"frontier_jepa/runs"/f"hssd_report_train_pair{k}"
            data = within(base, read(r/"receipt.json")["source"])
            decision = read(within(r/"decision", history+".json"))
            context = np.stack([rgb(within(r/"decision", f)) for f in decision["observed_rgb"]])
        else:
            data = within(base, key)
            context = npz_rgb(data/"context.npz")
        return context, (npz_rgb(data/f"branch{i:03d}.npz") for i in range(n))
    if sid.startswith("blender/"):
        _, layout, _ = sid.split("/")
        if n != 2:
            raise ValueError("Expected left/right fixture candidates")
        if layout == "report":
            p = root/"frontier_scenes/acceptance"
            start = rgb(p/"start.png")[None]
            paths = [p/(side+"_trajectory.npz") for side in ("left","right")]
        else:
            p = root/"frontier_scenes/attribute_fixture/clips"
            start = rgb(within(p, layout+"_start.png"))[None]
            paths = [within(p, layout+"_"+side+".npz") for side in ("left","right")]
        return start, (npz_rgb(path)[1:] for path in paths)
    if sid.startswith("extra/") and extra_rows:
        extra, row = extra_rows[sid.removeprefix("extra/")]
        with np.load(within(extra,row["path"]), allow_pickle=False) as z:
            return z["context_rgb"], z["future_rgb"]
    raise ValueError(f"No raw RGB source for {sid}. Supply --extra-raw for extra examples.")


def verify_pool(grid, reference):
    # This also checks that local re-encoding used exactly the pooled pack's frame order,
    # resize, candidate order and checkpoint, up to CPU/GPU floating-point variation.
    old = np.asarray(reference, dtype=np.float32)
    new = np.mean(grid, axis=-2)
    if old.shape != new.shape or not np.isfinite(new).all():
        raise ValueError("Invalid local or reference grids")
    relative = np.linalg.norm(new-old, axis=-1) / np.maximum(np.linalg.norm(old, axis=-1), 1.e-6)
    error = float(relative.max())
    if error > 0.015:
        raise ValueError(f"Feature fingerprint differs from immutable pooled inputs ({error:.3g})")
    return error


def reusable_grid(args, root):
    if not args.reuse_grid:
        if args.reuse_pooled:
            raise ValueError("--reuse-pooled requires --reuse-grid")
        return {}
    grid = args.reuse_grid.resolve(strict=True)
    receipt = read(grid/"receipt.json")
    if (receipt["schema"] != "most-branch-training/local-grid-v1"
            or sha(grid/"index.jsonl") != receipt["index_sha256"]):
        raise ValueError("Invalid reusable grid manifest")
    candidates = [args.reuse_pooled] if args.reuse_pooled else list((root/"training_ready/data").iterdir())
    pooled = next((p for p in candidates if (p/"index.jsonl").is_file() and
                   sha(p/"index.jsonl") == receipt["pooled_index_sha256"]), None)
    if pooled is None or read(pooled/"receipt.json")["index_sha256"] != receipt["pooled_index_sha256"]:
        raise ValueError("Specify the original audited --reuse-pooled pack")
    src = {row["id"]:row for row in records(pooled/"index.jsonl")}
    local = {row["id"]:row for row in records(grid/"index.jsonl")}
    if len(local)!=receipt["states"] or set(src)!=set(local):
        raise ValueError("Reusable grid coverage mismatch")
    return {sid:(grid, pooled, src[sid], local[sid]) for sid in local}


def reuse_row(record, pooled_input, existing, target_path):
    grid, pooled, old, cached = existing
    def metadata(row):
        return {k:v for k,v in row.items() if k not in ("path","sha256")}
    if metadata(record)!=metadata(old) or metadata(record)!=metadata(cached):
        raise ValueError(f"Reusable record metadata mismatch: {record['id']}")
    before = within(pooled,old["path"])
    local = within(grid,cached["path"])
    if old["sha256"] != record["sha256"] or sha(before)!=old["sha256"] or sha(local)!=cached["sha256"]:
        raise ValueError(f"Reusable row or source checksum mismatch: {record['id']}")
    with np.load(local,allow_pickle=False) as z:
        if set(z.files)!=set(pooled_input):
            raise ValueError("Unexpected local fields")
        for key in z.files:
            if key not in ("context","future") and not np.array_equal(z[key],pooled_input[key]):
                raise ValueError(f"Reusable nonvisual data changed: {key}")
        err=max(verify_pool(z["context"],pooled_input["context"]),verify_pool(z["future"],pooled_input["future"]))
    if target_path is not None:
        shutil.copyfile(local,target_path)
    return err


def run(args):
    root, data = args.root.resolve(), args.data.resolve()
    receipt = read(data/"receipt.json")
    if receipt["schema"] != "most-branch-training/v1" or sha(data/"index.jsonl") != receipt["index_sha256"]:
        raise ValueError("Expected audited mean-pooled input pack")
    if not args.check_only and (args.output is None or args.output.exists()):
        raise ValueError("Choose a fresh --output")
    extra_rows = {}
    for directory in args.extra_raw:
        for row in records(directory/"index.jsonl"):
            if row["id"] in extra_rows:
                raise ValueError(f"Duplicate raw Blender ID: {row['id']}")
            extra_rows[row["id"]] = (directory, row)
    raw_hashes = {}
    rows = records(data/"index.jsonl")
    if args.check_only:
        # One real example per independent raw source type; no weights or caches written.
        seen, selected = set(), []
        for r in rows:
            key = (r["source"], r["label_kind"], r["id"].split("/")[0])
            if key not in seen:
                selected.append(r)
                seen.add(key)
        rows = selected
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    encoder = None
    reusable = reusable_grid(args,root)
    reused = 0
    max_error = 0.
    with tempfile.TemporaryDirectory(dir=None if args.check_only else args.output.parent) as folder:
        target = Path(folder)/"pack"
        if not args.check_only:
            (target/"states").mkdir(parents=True)
        result = []
        for i, row in enumerate(rows):
            inp = within(data,row["path"])
            if sha(inp) != row["sha256"]:
                raise ValueError("Pooled source fingerprint changed")
            with np.load(inp, allow_pickle=False) as z:
                fields = {k:z[k] for k in z.files}
            if row["id"].startswith("extra/"):
                sid = row["id"].removeprefix("extra/")
                directory, raw_row = extra_rows[sid]
                if any(raw_row[key] != row[key] for key in ("house","split","query","label_kind")):
                    raise ValueError("Raw Blender metadata mismatch")
                raw_path = within(directory,raw_row["path"])
                raw_hashes[sid] = sha(raw_path)
                if raw_row.get("sha256",raw_hashes[sid]) != raw_hashes[sid]:
                    raise ValueError("Raw Blender checksum changed")
                with np.load(raw_path,allow_pickle=False) as z:
                    for key in ("xy","actions","gain","step_m","turn_deg","region_presence"):
                        if key in z and not np.array_equal(z[key],fields[key]):
                            raise ValueError(f"Raw Blender {key} differs from pooled pack")
            relative = Path("states")/f"{i:06d}.npz"
            if row["id"] in reusable:
                err=reuse_row(row,fields,reusable[row["id"]],None if args.check_only else target/relative)
                max_error=max(err,max_error)
                reused+=1
            else:
                if encoder is None:
                    encoder=GridEncoder(root, args.device)
                ctx, futures = clips(root, row["id"], len(fields["gain"]), extra_rows)
                context = encoder(ctx)
                future = np.stack([encoder(f) for f in futures])
                max_error = max(max_error, verify_pool(context, fields["context"]), verify_pool(future, fields["future"]))
                fields.update(context=context, future=future)
                if not args.check_only:
                    np.savez(target/relative, **fields)
            if not args.check_only:
                result.append({**row, "path":relative.as_posix(), "sha256":sha(target/relative)})
            if i and i % 32 == 0:
                print(f"Encoded {i+1}/{len(rows)}", flush=True)
        if not args.check_only:
            index = target/"index.jsonl"
            index.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in result))
            receipt.update(schema="most-branch-training/local-grid-v1", index_sha256=sha(index),
                           pooled_index_sha256=sha(data/"index.jsonl"),
                           raw_blender_sha256=raw_hashes,
                           feature="V-JEPA2.1 ViT-B/16, 4 uniform frames, 2 temporal bins x 4 quadrants",
                           max_relative_pool_error=max_error, reused_states=reused)
            (target/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
            target.rename(args.output)
    print(json.dumps(dict(check_only=args.check_only, checked_states=len(rows),
                          local_grid=[2,2,2,768], reused_states=reused,max_relative_pool_error=max_error)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extra-raw", type=Path, action="append", default=[])
    parser.add_argument("--reuse-grid", type=Path)
    parser.add_argument("--reuse-pooled", type=Path, help="Original source pack for --reuse-grid")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())
