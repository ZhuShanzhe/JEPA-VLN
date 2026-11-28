"""Project observed local RGB features into the frozen route predictor's latent space."""
import hashlib
import sys
from pathlib import Path

import numpy as np


SURFACES = ("wall", "table", "desk", "counter", "shelf", "cabinet",
            "sofa", "bed", "sink", "floor", "doorway", "hallway")
# The observer's left/right and upper/lower portions overlap at boundaries.
BOXES = tuple((u-.19, v-.19, u+.19, v+.19)
              for v in (.24, .50, .76) for u in (.21, .50, .79))


class LocalViewPredictor:
    """Use one RGB scan ordered right, center, left (signed yaw -30, 0, +30).

    Patches always refer to the central image. The jointly trained route
    predictor forecasts the whole future frame, not a future segmentation.
    We compare its endpoint feature with each presently visible anchor.
    """

    def __init__(self, model, placement):
        self.model = model
        self.prior = placement
        self.cache_key = None
        self.state = None
        sys.path.insert(0, str(Path(model.project) / "jepa_variable_v16/model"))

    def _observe(self, views, target):
        import torch
        from PIL import Image
        from torch.nn.functional import normalize

        views = np.asarray(views)
        if views.ndim != 4 or views.shape[0] != 3 or views.shape[-1] != 3 or views.dtype != np.uint8:
            raise ValueError("expected three ordered uint8 RGB views")
        key = (str(target), views.shape, hashlib.blake2b(views.tobytes(), digest_size=16).digest())
        if key == self.cache_key:
            return self.state
        m, p = self.model, self.prior
        if len(m.encoder.cache) > 450:
            m.encoder.cache.clear()
        height, width = views.shape[1:3]
        crops, quality = [], []
        for x0, y0, x1, y1 in BOXES:
            region = views[1, int(y0*height):max(int(y0*height)+1, round(y1*height)),
                           int(x0*width):max(int(x0*width)+1, round(x1*width))]
            spatial_std = np.mean(np.std(region.astype(np.float32), axis=(0, 1)))
            # No visual support from constant images or almost uniform crops.
            quality.append(float(np.clip((spatial_std-1.) / 8., 0., 1.)))
            crops.append(np.asarray(Image.fromarray(region).resize((112, 112), Image.Resampling.BILINEAR)))
        with torch.inference_mode():
            words = m.encoder.words([target])
            scan, _ = m.encoder.encode(views, words)
            feat, _ = m.encoder.encode(np.stack(crops), words)
            features = np.asarray(feat[:, 0, :], dtype=np.float32)
            prompts = p.visual_prompts(m)
            sim = features @ prompts.T
            logits = 20. * (sim-sim.max(axis=1, keepdims=True))
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)
            room = m.reader(m.tensor(scan[1:2])).softmax(-1).cpu().numpy()
            room = np.repeat(room, len(crops), axis=0)
            semantic = p.features(target, room, prob[:, :9])[:, 1]
            support = 1. / (1. + np.exp(-semantic))
            text = normalize(words @ m.encoder.proj, dim=-1)
        patches = []
        for i, (x0, y0, x1, y1) in enumerate(BOXES):
            idx = int(np.argmax(prob[i]))
            u, v = (x0+x1)/2, (y0+y1)/2
            bg = float(prob[i, 9:].sum())
            patches.append(dict(patch_id=i, view_index=1, yaw_offset_deg=0.,
                                u=u, v=v, uv=[u, v], bbox_norm=[x0, y0, x1, y1],
                                surface=SURFACES[idx], surface_prob=float(prob[i,idx]),
                                background_mass=bg, quality=quality[i],
                                support=float(support[i] * (1.-bg) * quality[i])))
        self.cache_key = key
        self.state = dict(patches=patches, scan=np.asarray(scan, dtype=np.float32),
                          features=features, text=text)
        return self.state

    def patches(self, views, target):
        return self._observe(views, target)["patches"]

    def score(self, views, target, plans):
        import torch
        from env_routes import prior_vector
        from variable_evidence import feature_rows

        plans = [np.asarray(plan) for plan in plans]
        if not plans or any(a.ndim != 1 or not 1 <= len(a) <= 24 or a.dtype.kind not in "iu"
                            or not np.isin(a, [1, 2, 3]).all() for a in plans):
            raise ValueError("expected actual Habitat forward/left/right prefixes of 1..24 steps")
        state = self._observe(views, target)
        m = self.model
        rows, taps = feature_rows(state["scan"], [a-1 for a in plans], m.old.route_inputs)
        canonical = self.prior.key(target) or target
        with torch.inference_mode():
            prior = m.tensor(prior_vector(m.raw, canonical))[None]
            _, _, forecast = m.heads["jepa"](m.tensor(rows)[None, :, None], state["text"], prior)
            ends = np.cumsum([len(t) for t in taps])-1
            future = forecast[0, :, 0, :].cpu().numpy()[ends]
        # Similarity is a relative retrieval score. These values are for
        # ranking sites; they are not probabilities of actually finding an item.
        similarity = future @ state["features"].T
        centered = similarity-similarity.mean(axis=1, keepdims=True)
        match = 1./(1.+np.exp(-12.*centered))
        support = np.asarray([p["support"] for p in state["patches"]])
        scores = match * support[None]
        return dict(patches=state["patches"], route_patch_scores=scores.tolist(),
                    latent_similarity=similarity.tolist())
