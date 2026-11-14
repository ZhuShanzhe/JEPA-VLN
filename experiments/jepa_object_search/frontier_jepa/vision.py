"""Convert a decision-time HSSD RGB history to the official pooled V-JEPA feature."""
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import torch


class ContextEncoder:
    def __init__(self, project_root=None, device="cuda"):
        root = Path(project_root or Path(__file__).resolve().parents[1])
        source = root / "src" / "vjepa2"
        sys.path.insert(0, str(source))
        # The upstream backbone uses absolute app.* imports under its own tree.
        if "app" not in sys.modules:
            app = types.ModuleType("app")
            app.__path__ = [str(source / "app")]
            sys.modules["app"] = app
        from src.hub.backbones import _clean_backbone_key, vjepa2_1_vit_base_384
        encoder, head = vjepa2_1_vit_base_384(pretrained=False)
        del head
        path = root / "models" / "vjepa2" / "vjepa2_1_vitb_dist_vitG_384.pt"
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        encoder.load_state_dict(_clean_backbone_key(checkpoint["ema_encoder"]), strict=True)
        self.encoder = encoder.eval().to(device)
        self.device = torch.device(device)
        self.mean = torch.tensor([.485,.456,.406], device=device).reshape(1,3,1,1,1)
        self.std = torch.tensor([.229,.224,.225], device=device).reshape(1,3,1,1,1)

    @torch.inference_mode()
    def encode(self, observed_rgb):
        frames = np.asarray(observed_rgb)
        if frames.ndim != 4 or frames.shape[-1] != 3 or not len(frames) or frames.dtype != np.uint8:
            raise ValueError("Expected a nonempty history of observed uint8 RGB frames (T,H,W,3)")
        indices = np.linspace(0, len(frames)-1, 4, dtype=np.int64)
        resized = np.stack([np.asarray(Image.fromarray(frames[i]).resize(
            (384,384), Image.Resampling.BILINEAR)) for i in indices])
        video = torch.from_numpy(resized).permute(3,0,1,2).unsqueeze(0)
        normalized = (video.to(self.device).float().div(255.) - self.mean) / self.std
        pooled = self.encoder(normalized).float().mean(1)[0].cpu().numpy()
        if pooled.shape != (768,) or not np.isfinite(pooled).all():
            raise ArithmeticError("Unexpected encoder result")
        return pooled
