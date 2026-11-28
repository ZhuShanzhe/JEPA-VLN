"""Learned region and local structure priors for goal-driven exploration."""
import json
import re
import sys
from pathlib import Path

import numpy as np


CONTEXTS = ("room edge", "table", "desk", "counter", "shelf", "cabinet", "sofa", "bed", "sink")
PROMPTS = ("indoor wall", "tabletop", "office desk", "kitchen countertop", "shelving unit",
           "storage cabinet", "sofa", "bed", "sink", "open floor", "open doorway", "empty hallway")
GROUPS = {
    "clock": {"clock", "wall_clock", "mantel_clock", "alarm_clock", "grandfather_clock"},
    "cup": {"cup", "mug", "teacup", "coffee_mug"},
    "painting": {"painting", "picture_frame", "picture"},
    "lamp": {"lamp", "table_lamp", "floor_lamp", "wall_lamp", "ceiling_lamp"},
    "mirror": {"mirror", "wall_mirror", "hand_mirror"},
    "table": {"table", "coffee_table", "dining_table", "end_table", "side_table"},
    "shelf": {"shelf", "bookcase", "bookshelf", "shelving"},
    "counter": {"counter", "countertop", "kitchen_counter", "kitchen_island"},
    "cabinet": {"cabinet", "chest_of_drawers", "cupboard", "kitchen_cabinet"},
    "sofa": {"sofa", "couch", "loveseat"},
    "television": {"television_receiver", "television", "tv", "television_set"},
}


def normalize(name):
    return re.sub(r"\.n\.\d+$", "", name.lower().strip()).replace(" ", "_")


def categories(row):
    keys = {normalize(row.get(key, "")) for key in ("wnsynsetkey", "main_wnsynsetkey", "main_category")}
    keys.discard("")
    keys.update(group for group, values in GROUPS.items() if keys & values)
    return keys


class PlacementPrior:
    def __init__(self, record):
        self.record = record
        self.rooms = record["rooms"]
        self.targets = record["targets"]
        self.visual = None

    @classmethod
    def load(cls, path):
        obj = cls(json.loads(Path(path).read_text(encoding="utf-8")))
        weights = Path(path).with_name("context_calibration.json")
        if weights.exists():
            obj.visual = json.loads(weights.read_text())
            if obj.visual["prior_sha256"] != __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest():
                raise ValueError("visual calibration uses a different placement prior")
        return obj

    def key(self, target):
        name = normalize(target)
        if name in self.targets:
            return name
        return next((group for group, aliases in GROUPS.items() if name in aliases and group in self.targets), None)

    def score(self, target, room_probs, context_scores=None):
        """One prior per direction. Room/context distributions must come from observations or forecasts.

        Without local evidence this estimates room occupancy. With observed local context
        it estimates target proximity to a comparable anchor, separately from occupancy.
        An all-zero context row explicitly falls back to room occupancy.
        """
        room = np.asarray(room_probs, dtype=float)
        if room.ndim != 2 or room.shape[1] != len(self.rooms) or not np.isfinite(room).all() or np.any(room < 0):
            raise ValueError("invalid room distributions")
        denom = room.sum(axis=1, keepdims=True)
        if np.any(denom <= 0):
            raise ValueError("room probabilities must have positive mass")
        room = room / denom
        key = self.key(target)
        if key is None:
            return np.full(len(room), 0.5)
        data = self.targets[key]
        coarse = room @ np.asarray(data["room"])
        if context_scores is None:
            return coarse
        context = np.asarray(context_scores, dtype=float)
        if context.shape != (len(room), len(CONTEXTS)) or not np.isfinite(context).all() or np.any(context < 0):
            raise ValueError("invalid observed structure scores")
        mass = context.sum(axis=1)
        local_probs = np.einsum("nr,rc->nc", room, np.asarray(data["local"]))
        return np.divide((local_probs*context).sum(axis=1), mass, out=coarse.copy(), where=mass>0)

    @staticmethod
    def visual_prompts(model):
        import torch
        with torch.inference_mode():
            if not hasattr(model,"placement_text_features"):
                t = model.encoder.words(PROMPTS)@model.encoder.proj
                model.placement_text_features = torch.nn.functional.normalize(t,dim=-1).cpu().numpy()
            return model.placement_text_features

    @staticmethod
    def current_cues(ctx, prompt_features):
        # Full-view feature of each observed direction. Extra background prompts
        # absorb uninformative views; the vision model never gets instance labels.
        similarity = np.asarray(ctx,dtype=np.float32)[:,0,:] @ prompt_features.T
        logits = 20*(similarity-similarity.max(1,keepdims=True))
        dist = np.exp(logits); dist /= dist.sum(1,keepdims=True)
        return dist[:,:len(CONTEXTS)]

    @staticmethod
    def local_directions(plans, current):
        """Route-relevant evidence from the three real departure frames only."""
        result = []
        heads=np.deg2rad([-30.,0.,30.])
        for actions in plans:
            yaw=np.cumsum((actions==1).astype(float)-(actions==2))*np.pi/6.
            x=np.cumsum((actions==0)*.25*np.cos(yaw))
            y=np.cumsum((actions==0)*.25*np.sin(yaw))
            distance=np.hypot(x,y)
            angle=np.where(distance>.05,np.arctan2(y,x),yaw)
            delta=np.abs(np.angle(np.exp(1j*(angle[:,None]-heads))))
            nearest=delta.argmin(axis=1)
            # Beyond 2 m, around a corner or behind the camera, the originating
            # view provides no localized evidence for this point.
            weight=np.maximum(0,1-distance/2)*np.maximum(0,1-delta.min(1)/(np.pi/4))
            result.append((current[nearest]*weight[:,None]).mean(0))
        return np.asarray(result)

    def features(self,target,room,cues):
        base=self.score(target,room)
        local=self.score(target,room,cues)
        ref=self.score(target,room,np.ones_like(cues))
        logit=lambda p:np.log(np.clip(p,1e-5,1-1e-5)/np.clip(1-p,1e-5,1))
        # Compare local-target proximity against other support types under the
        # same room distribution. Occupancy itself never substitutes for proximity.
        evidence=np.clip(logit(local)-logit(ref),-2.,2.)*cues.sum(-1)
        return np.stack([logit(base),evidence],axis=-1)

    def calibrated(self, target, room, cues, use_context=True, arm="jepa"):
        if self.visual is None:
            raise RuntimeError("missing trained visual context calibration")
        f=self.features(target,room,cues)
        b,w,d=self.visual["arms"][arm]["coefficients"]
        return 1/(1+np.exp(-np.clip(b+w*f[:,0]+(d*f[:,1] if use_context else 0.),-30,30)))

    def score_current(self, model, rgb, target, *, use_context=True, arm="jepa"):
        """Scene-conditioned suitability of the supplied actually observed directions."""
        import torch
        with torch.inference_mode():
            ctx,_ = model.encoder.encode(np.asarray(rgb),model.encoder.words([target]))
            room = model.reader(model.tensor(ctx)).softmax(-1).cpu().numpy()
            return self.calibrated(target, room, self.current_cues(ctx,self.visual_prompts(model)),use_context,arm)

    def forecast(self, model, rgb, target, plans, arm="jepa", *, use_context=True):
        return self.forecast_with_evidence(model,rgb,target,plans,arm,use_context=use_context)["prior"]

    def forecast_with_evidence(self, model, rgb, target, plans, arm="jepa", *, use_context=True):
        """Return the early search prior and the frozen target-specific route score.

        Plans use Habitat 1=forward,2=left,3=right, 1..24 actual steps. Observed
        scans are ordered -30, 0, +30 degrees. Unknown text falls back to the
        prior rather than extrapolating a target-specific head.
        """
        import torch
        from torch.nn.functional import normalize as unit
        if arm not in ("jepa","direct"):
            raise ValueError("unknown arm")
        plans = [np.asarray(plan) for plan in plans]
        if not plans or any(p.ndim!=1 or not 1<=len(p)<=24 or p.dtype.kind not in "iu" or
                            not np.isin(p,[1,2,3]).all() for p in plans):
            raise ValueError("expected nonempty actual Habitat forward/left/right prefixes up to 24")
        views = np.asarray(rgb)
        if views.ndim!=4 or views.shape[0]!=3 or views.shape[-1]!=3 or views.dtype!=np.uint8:
            raise ValueError("expected three observed RGB uint8 camera views")
        folder = str(Path(model.project)/"jepa_variable_v16/model")
        if folder not in sys.path:
            sys.path.insert(0,folder)
        from variable_evidence import feature_rows
        old_vocab={normalize(w):w for w in model.raw["p_target_given_room"]}
        canonical=old_vocab.get(normalize(target))
        if canonical is None:
            canonical=old_vocab.get(self.key(target))
        word=canonical if canonical is not None else target
        with torch.inference_mode():
            tokens = model.encoder.words([word])
            ctx, _ = model.encoder.encode(views,tokens)
            text = unit(tokens@model.encoder.proj,dim=-1)
            rows,taps = feature_rows(np.asarray(ctx,np.float32), [p-1 for p in plans],model.old.route_inputs)
            # The original target-specific head also consumes the *old* room
            # prevalence. Each endpoint is independent; preserve its own tap.
            prior = model.tensor(model.raw["p_target_given_room"].get(word,[0.]*len(self.rooms)))[None]
            logits,r,_ = model.heads[arm](model.tensor(rows)[None,:,None],text,prior)
            dists = r.softmax(-1).reshape(-1,len(self.rooms)).cpu().numpy()
        if len(model.encoder.cache)>600:
            model.encoder.cache.clear()
        by_route = np.split(dists,np.cumsum([len(t) for t in taps])[:-1])
        room = [np.average(d, axis=0, weights=np.diff([0,*t])) for d,t in zip(by_route,taps)]
        current = self.current_cues(ctx,self.visual_prompts(model))
        cues = self.local_directions([p-1 for p in plans],current)
        semantic_prior=self.calibrated(target,room,cues,use_context,arm)
        if canonical is None:
            return dict(prior=semantic_prior,evidence=semantic_prior.copy(),evidence_supported=False)
        if model.config[arm]["mix"]!=1:
            raise ValueError("frozen legacy full head is expected to have unit blend")
        gains=np.split(logits.flatten().cpu().numpy(),np.cumsum([len(t) for t in taps])[:-1])
        pooled=[np.logaddexp.reduce(g+np.log(np.diff([0,*t])/t[-1])) for g,t in zip(gains,taps)]
        evidence=1/(1+np.exp(-np.asarray(pooled)))
        return dict(prior=semantic_prior,evidence=evidence,evidence_supported=True)
