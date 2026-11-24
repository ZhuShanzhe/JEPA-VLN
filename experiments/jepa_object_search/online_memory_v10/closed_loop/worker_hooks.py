"""Causal observation/decision interface, running inside the frozen V8 encoder process."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


class Hooks:
    def __init__(self, root, pilot):
        directory = root / "jepa_memory_v10_20261124/models"
        sys.path.insert(0, str(directory))
        from v10_live import RouteMemory
        weights = directory / "v10.json"
        self.memory = RouteMemory(weights)
        self.arm = json.loads((pilot / "arm.json").read_text())["memory_arm"]
        self.step = self.xy = self.heading = self.query = None
        self.path = None
        print("V10 online scorer", hashlib.sha256(weights.read_bytes()).hexdigest(),flush=True)

    def observe(self, request, ground, enc, clip, words, phrase):
        step, query = request["step"], request["query"]
        if request["reset"]:
            self.memory.reset()
            self.step = None
        if self.step is not None and (step!=self.step+1 or query!=self.query):
            raise RuntimeError("episode memory history mismatch")
        score = None
        if request["frame"] is not None:
            shape, buf = request["frame"]
            rgb = np.frombuffer(buf,np.uint8).reshape(shape).copy()
            q = phrase(query)
            if q not in words:
                words[q] = torch.from_numpy(clip.texts([q])[q]).float()
            text = words[q][None]
            grid = enc.local([rgb])
            code, anc, sim = enc.compressed(grid,text)
            v, _ = ground(code.half().float(),anc.half().float(),sim.half().float(),text)
            score = float(v.max())
        self.step, self.xy, self.heading, self.query = step, request["xz"], request["heading"], query
        self.memory.observe(self.xy,step,self.heading,visual_score=score,collided=request["collided"])
        # Private current-episode state is checkpointed atomically after each real observation.
        if self.path is None:
            import os
            self.path=Path(os.readlink("/proc/self/fd/1")).parent/"v10_memory.json"
        scratch=self.path.with_suffix(".tmp")
        self.memory.save(scratch)
        scratch.replace(self.path)
        return dict(step=step, reset=bool(request["reset"]), measured_rgb=score is not None,
                    visual_score=score, collided=bool(request["collided"]))

    def rank(self, request, fields):
        if self.step is None or request["query"]!=self.query:
            raise RuntimeError("missing fresh episode memory")
        pairs=request["pairs"]
        assert len({p["horizon"] for p in pairs})==1
        assert all(p["ref"]==pairs[0]["ref"] for p in pairs)
        plans=[pairs[0]["ref"]]+[p["alt"] for p in pairs]
        def flat(name):
            rows=fields[name]
            if any(abs(rows[0]["reference_score"]-r["reference_score"])>1e-5 for r in rows):
                raise RuntimeError("inconsistent scores for identical host plan")
            return [rows[0]["reference_score"]]+[r["alternative_score"] for r in rows]
        inputs=(plans,self.xy,self.heading,self.step,flat("comparisons"),flat("observed_comparisons"),flat("geometry_comparisons"))
        original=fields["comparisons"]
        fields["v8_comparisons"]=original
        for key,mode in (("full","full"),("frozen","fixed"),("no_jepa","observed")):
            out=self.memory.rank(*inputs,arm=mode)
            scores=out["scores"]
            assert len(scores)==len(plans)
            rows=[dict(row,reference_score=float(scores[0]),alternative_score=float(scores[i+1]))
                  for i,row in enumerate(original)]
            fields[key+"_comparisons"]=rows
            if mode==self.arm:
                fields["comparisons"]=rows
            if key=="full":
                fields["memory_features"]=out["memory_features"]
        fields["memory_step"]=self.step
