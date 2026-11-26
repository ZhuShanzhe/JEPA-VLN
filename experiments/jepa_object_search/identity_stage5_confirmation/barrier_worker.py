"""Two finite, offline-only stages for a stationary local identity query."""
import json
import os
from pathlib import Path
import sys

import numpy as np


def stream_vggt_weights(model):
    """Keep the unmodified forward order, tensor shapes and dtype on CUDA.

    Only the active attention block (or active prediction head) needs resident
    weights. Upstream RoPE is shared between blocks; hooks surround its use.
    """
    import torch
    active = [model.aggregator.patch_embed, *model.aggregator.frame_blocks,
              *model.aggregator.global_blocks, model.camera_head, model.depth_head]
    def load_block(module, args):
        module.cuda()
    def release_block(module, args, result):
        module.cpu()
    for part in active:
        part.cpu()
        part.register_forward_pre_hook(load_block)
        part.register_forward_hook(release_block)
    # Native geometry forward does not use tracking or point heads.
    for part in (model.track_head, model.point_head):
        if part is not None:
            part.cpu()
    torch.cuda.empty_cache()


def reconstruct(root, directory):
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from agents import navigator
    from replay_anchored import infer_anchors
    from shared_anchor_pair import split_camera_height_scales

    torch.cuda.reset_peak_memory_stats()
    model = VGGT().eval()
    checkpoint = root / "mapping_runtime/weights/torch/hub/checkpoints/model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.cuda()
    stream_vggt_weights(model)
    images = np.load(directory / "images.npy", allow_pickle=False)
    try:
        depth, K, poses, _ = infer_anchors(model, images, 1., pose_encoding_to_extri_intri)
    except ValueError as exc:
        if str(exc) != "unobservable scale in past anchors":
            raise
        (directory / "unobservable.json").write_text(json.dumps({"reason":"unobservable_motion_in_past"}))
        return
    scales = split_camera_height_scales(depth[1:9], K[1:9], poses[1:9], navigator, 1.5)
    np.savez(directory / "bundle.npz", depth=depth, K=K, poses=poses)
    (directory / "floor.json").write_text(json.dumps(scales, allow_nan=False))
    (directory / "gpu.json").write_text(json.dumps(dict(
        peak_allocated_mib=round(torch.cuda.max_memory_allocated()/1024**2),
        peak_reserved_mib=round(torch.cuda.max_memory_reserved()/1024**2))))


def score(root, directory):
    from bridge import load_verifier
    from shared_anchor_pair import verify_pair
    from optics import skew_view

    query = json.loads((directory / "query.json").read_text())
    bundle_dir = directory.parent
    images = np.load(bundle_dir / "images.npy", allow_pickle=False)
    def metric(fid):
        with np.load(bundle_dir / f"map_{fid}.npz", allow_pickle=False) as data:
            return skew_view(data, query["scale"])

    anchors = [(fid, slot, np.load(bundle_dir / f"rgb_{fid}.npy", allow_pickle=False), metric(fid))
               for fid, slot in query["anchors"]]
    with np.load(bundle_dir / "bundle.npz", allow_pickle=False) as bundle:
        result = verify_pair(images, bundle["depth"], bundle["K"], bundle["poses"],
                             metric(query["source"]), np.asarray(query["source_box"]),
                             np.asarray(query["boxes"]), anchors,
                             json.loads((bundle_dir / "floor.json").read_text()),
                             load_verifier(root, "jepa", "cpu"))
    (directory / "verdict.json").write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    os.umask(0o077)
    root = Path(os.environ["NAV_ROOT"])
    sys.path[:0] = [str(root / sub) for sub in ("identity_stage4_geometry/provider",
        "mapping_runtime/env_build/src/VGGT_SPARK", "private_sources/MOST-Nav", "identity_stage3_model/integration")]
    {"reconstruct":reconstruct, "score":score}[sys.argv[1]](root, Path(sys.argv[2]))
