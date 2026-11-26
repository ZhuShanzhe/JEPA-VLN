"""Use MOST-Nav's own automatic regions and return boxes on the caller's RGB grid."""
import sys
from pathlib import Path

import numpy as np


class NativeRegions:
    def __init__(self, project, device="cuda"):
        project = Path(project)
        sys.path.insert(0, str(project / "private_sources/MOST-Nav"))
        from mapping.sam_backend import SAMRefiner
        from mapping.server import MappingServer
        self.grid = MappingServer._vggt_frame_grid
        self.sam = SAMRefiner(project / "mapping_runtime/weights/sam_vit_h_4b8939.pth",
                              model_type="vit_h", device=device)
        if not self.sam.load():
            raise RuntimeError(self.sam.disabled_reason)
        if device.startswith("cuda") and self.sam._predictor.model.device.type != "cuda":
            raise RuntimeError("Expected GPU SAM")
        # The service returns [] for both an empty scene and an inference
        # exception. Preserve that distinction for the offline evaluator.
        generate = self.sam._generator.generate
        self.failure = None
        def checked_generate(image):
            try:
                return generate(image)
            except Exception as exc:
                self.failure = exc
                raise
        self.sam._generator.generate = checked_generate

    def propose(self, rgb):
        im = np.asarray(rgb)
        if im.ndim != 3 or im.shape[2] != 3 or im.dtype != np.uint8:
            raise ValueError("Expected uint8 HxWx3 RGB")
        h,w = im.shape[:2]
        grid = self.grid(im)
        gh,gw = grid.shape[:2]
        uncropped_h = max(14, round(h * (518./w)/14) * 14)
        cropped_top = (uncropped_h-gh)//2
        self.failure = None
        regions = self.sam.segment_all(grid)
        if self.failure is not None:
            raise RuntimeError("Official SAM inference failed") from self.failure
        rows = []
        for mask in regions:
            a,b,c,d = mask["bbox"]
            x,y = mask["centroid"]
            box = [a*w/gw,(b+cropped_top)*h/uncropped_h,c*w/gw,(d+cropped_top)*h/uncropped_h]
            rows.append(dict(mask_id=int(mask["mask_id"]),xyxy=box,bbox=box,
                             score=float(mask["iou"]), area_frac=float(mask["area_frac"]),
                             centroid=[x*w/gw,(y+cropped_top)*h/uncropped_h],
                             native_bbox=list(mask["bbox"]),native_centroid=list(mask["centroid"]),
                             view="native_sam"))
        return rows
