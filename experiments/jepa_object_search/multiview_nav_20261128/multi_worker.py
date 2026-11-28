"""Frozen route scores plus retrieval of one chosen box in its actual source camera."""
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np


def selected_box_scores(local, views, goal, plans, view_index, box):
    import torch
    from PIL import Image
    from env_routes import prior_vector
    from variable_evidence import feature_rows

    views = np.asarray(views)
    if view_index not in (0, 1, 2) or len(box) != 4:
        raise ValueError("invalid selected camera or box")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("invalid normalized box")
    plans = [np.asarray(p) for p in plans]
    if not plans or any(p.ndim != 1 or not 1 <= len(p) <= 24 or
                        p.dtype.kind not in "iu" or not np.isin(p, [1,2,3]).all() for p in plans):
        raise ValueError("invalid action plan")
    state = local._observe(views, goal)  # retain the trained [right,center,left] context
    m = local.model
    frame = views[view_index]
    h, w = frame.shape[:2]
    crop = frame[int(y0*h):int(np.ceil(y1*h)), int(x0*w):int(np.ceil(x1*w))]
    image = np.asarray(Image.fromarray(crop).resize((112,112), Image.Resampling.BILINEAR))
    rows, taps = feature_rows(state["scan"], [p-1 for p in plans], m.old.route_inputs)
    with torch.inference_mode():
        words = m.encoder.words([goal])
        features, _ = m.encoder.encode(image[None], words)
        key = np.asarray(features[0,0], dtype=np.float32)
        prior = m.tensor(prior_vector(m.raw, local.prior.key(goal) or goal))[None]
        _, _, latent = m.heads["jepa"](m.tensor(rows)[None,:,None], state["text"], prior)
        ends = np.cumsum([len(t) for t in taps])-1
        future = latent[0,:,0,:].cpu().numpy()[ends]
    cos = future @ key / np.maximum(np.linalg.norm(future, axis=1)*np.linalg.norm(key), 1e-8)
    if not np.isfinite(cos).all():
        raise ValueError("nonfinite route-to-box similarity")
    return ((1+np.clip(cos, -1, 1))/2).tolist()


def serve(project, fd):
    old = project/"local_target_20261128/localvision"
    sys.path[:0] = [str(old), str(project/"jepa_env_v13/models"), str(project/"jepa_variable_v16/model"),
                    str(project/"approach_search_20261128/eval4/locked")]
    from env_routes import EnvironmentRoutes
    from placement_prior import PlacementPrior
    from variable_evidence import VariableEvidence
    from local_vision import LocalViewPredictor
    m = EnvironmentRoutes(project)
    prior = PlacementPrior.load(project/"approach_search_20261128/eval4/locked/learned_prior.json")
    var = VariableEvidence(project)
    local = LocalViewPredictor(m, prior)
    with Connection(fd) as channel:
        while True:
            try:
                req = channel.recv()
            except EOFError:
                return
            if req["mode"] == "observe":
                views, goal, plans = np.asarray(req["views"]), req["goal"], req["plans"]
                current = prior.score_current(m, views, goal).tolist()
                score = prior.forecast(m, views, goal, plans).tolist() if plans else []
                evidence = var.predict(m, views, goal, [np.asarray(a)-1 for a in plans])["evidence_rate"] if plans else []
                channel.send(dict(current=current, score=score, evidence=list(evidence), patches=local.patches(views,goal)))
            elif req["mode"] == "selected_box":
                channel.send(selected_box_scores(local, views, goal, req["plans"], req["view_index"], req["bbox_norm"]))
            else:
                raise ValueError(req["mode"])


def start_worker(project, log_path, cpu=False):
    left, right = socket.socketpair()
    try:
        env = {k:v for k,v in os.environ.items() if k not in ("PYTHONPATH","PYTHONHOME")}
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        if cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        py = project.parent/"jepa_instance_bench_20261016/policy_env/bin/python"
        with Path(log_path).open("ab") as log:
            proc = subprocess.Popen([str(py),"-Bu",str(Path(__file__).resolve()),str(project),str(right.fileno())],
                                    pass_fds=(right.fileno(),), stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, env=env)
        return Connection(left.detach()), proc
    finally:
        left.close()
        right.close()


if __name__ == "__main__":
    serve(Path(sys.argv[1]), int(sys.argv[2]))
