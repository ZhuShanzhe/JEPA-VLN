"""Stop-on-confirm runtime for one localized instance. Requires native RGB-D and measured odometry."""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from frame_confirm import FrameConfirmer


def save_confirmation(directory, frame, verification, step):
    """Publish the private RGB, crop and metadata as one directory; repeat of the same frame is idempotent."""
    path=Path(directory)
    path.mkdir(mode=0o700,parents=True,exist_ok=True)
    output=path/"confirmed"
    image=np.asarray(frame["rgb"],dtype=np.uint8)[...,:3]
    x,y,r,b=map(int,verification["box"])
    if not (0<=x<r<=image.shape[1] and 0<=y<b<=image.shape[0]):
        raise ValueError("invalid confirmed box")
    receipt=dict(step=step,box=[x,y,r,b],rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                 p_clear=float(verification["p_clear"]),p_same=float(verification["p_same"]))
    def checked():
        saved=json.loads((output/"evidence.json").read_text())
        if saved!=receipt:raise FileExistsError("use a distinct output directory for each instance request")
        with Image.open(output/"frame.png") as saved_rgb, Image.open(output/"crop.png") as saved_crop:
            if not np.array_equal(saved_rgb,image) or not np.array_equal(saved_crop,image[y:b,x:r]):
                raise IOError("previously saved evidence differs")
        return output
    if output.exists():return checked()
    with tempfile.TemporaryDirectory(dir=path,prefix=".confirm-") as stage:
        bundle=Path(stage)/"confirmed";bundle.mkdir(mode=0o700)
        Image.fromarray(image).save(bundle/"frame.png")
        Image.fromarray(image[y:b,x:r]).save(bundle/"crop.png")
        (bundle/"evidence.json").write_text(json.dumps(receipt)+"\n")
        try:bundle.rename(output)
        except OSError:
            if not output.is_dir():raise
    return checked()


class ReactiveInspection:
    def __init__(self,initial,reroute,encoder,verifier=FrameConfirmer):
        self.initial,self.reroute,self.encoder,self.verifier=initial,reroute,encoder,verifier

    @classmethod
    def load(cls,device="cpu"):
        from reactive_route import RoutePlanner
        from observation_actor import ObservationActor
        route=RoutePlanner.load()
        return cls(ObservationActor.load(device),route,route.encoder)

    def run(self,source,source_box,forward_steps,budget,move,directory):
        """move(action) performs exactly one primitive and returns rgb/depth/x/z/yaw/bump.

        x, z and yaw are accumulated observations in the initial local frame. budget includes
        the shared left scan and turn back. Completion means the accepted evidence is saved.
        An accepted frame ends this local request immediately, including at the initial pose.
        """
        if forward_steps not in (4,8,12) or budget<0:raise ValueError("invalid initial request or budget")
        if not np.allclose([source[k] for k in ("x","z","yaw")],0.,atol=1.e-4):
            raise ValueError("initial observation must define the odometry origin")
        history=[source]; plan=deque((2,3));initial_attempted=False
        verifier=self.verifier(source["rgb"],source["depth"],source_box)
        context=None
        while True:
            frame=history[-1];t=len(history)-1
            pose=tuple(frame[k] for k in ("x","z","yaw"))
            verified=verifier.observe(frame["rgb"],frame["depth"],pose)
            if verified["accept"]:
                dest=save_confirmation(directory,frame,verified,t)
                return dict(status="confirmed",actions_used=t,evidence_directory=str(dest))
            if t>=budget:return dict(status="budget_exhausted",actions_used=t)
            if frame.get("bump",False):plan.clear()
            if not plan:
                if not initial_attempted:
                    from prior_cues_score import visual_input
                    x,y,r,b=map(int,source_box)
                    rgb=frame["rgb"][...,:3]
                    context=self.encoder.encode([visual_input(rgb),visual_input(rgb[y:b,x:r])]).ravel()
                    first=self.initial.choose(context,source_box,frame["depth"],history[1]["depth"],forward_steps,budget-t)
                    plan.extend(first["actions"]);initial_attempted=True
                if not plan:
                    decision=self.reroute.choose(context,source_box,history[0]["depth"],history,budget-t)
                    plan.extend(decision["actions"])
                    if not plan:return dict(status="no_supported_view",actions_used=t,reason=decision["reason"])
                if len(plan)>budget-t:raise ValueError("planner exceeded remaining actions")
            action=int(plan.popleft())
            if action not in (1,2,3):raise ValueError("unexpected navigation primitive")
            observed=dict(move(action))
            if action==1:
                displacement=np.hypot(observed["x"]-frame["x"],observed["z"]-frame["z"])
                observed["bump"]=bool(observed.get("bump",False) or displacement<.23)
            history.append(observed)
