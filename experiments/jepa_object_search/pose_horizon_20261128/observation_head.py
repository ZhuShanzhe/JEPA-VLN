"""One-cue observation head. The caller supplies only positions inside the region it already chose."""
import numpy as np

from refine_view import correct_view


class HorizonCameraPredictor:
    """Keep the short-range checkpoint intact; use the long-range head for plans of 12–108 actions."""
    def __init__(self, short, long):
        self.short=short
        self.long=long

    @classmethod
    def from_checkpoints(cls, directory, device="cpu"):
        from pathlib import Path
        from eval_pose import CameraPredictor
        folder=Path(directory)
        return cls(CameraPredictor(folder/"joint_long.pt",device),CameraPredictor(folder/"long_cue_joint.pt",device))

    def __call__(self, context, action, pose):
        action=np.asarray(action)
        if action.ndim!=2 or not len(action):
            raise ValueError("expected a nonempty batch of action plans")
        steps=(action!=0).sum(axis=1)
        if np.any((steps<1)|(steps>108)):
            raise ValueError("supported action lengths are 1 through 108")
        result={}
        for mask,model in ((steps<=11,self.short),(steps>=12,self.long)):
            if not mask.any():continue
            output=model(np.asarray(context)[mask],action[mask],np.asarray(pose)[mask])
            for name, v in output.items():
                if name not in result:
                    result[name]=np.empty((len(action),)+v.shape[1:],dtype=v.dtype)
                result[name][mask]=v
        return result


def propose(predictor, source, crop, plans, camera_features, clearance, initial, *, articulated=False):
    """Return a local pose proposal and its future embedding for the executor, without moving the agent."""
    x=np.asarray(camera_features,np.float32)
    source=np.asarray(source,np.float32)
    crop=np.asarray(crop,np.float32)
    if source.shape!=(1024,) or crop.shape!=(1024,) or x.ndim!=2 or x.shape[1]!=20 or not len(x):
        raise ValueError("a single current image, cue and at least one 20-D camera are required")
    actions=np.asarray(plans)
    if actions.ndim!=2 or len(actions)!=len(x):
        raise ValueError("a primitive plan is required for each proposed pose")
    supported=np.asarray(clearance,bool).copy()
    if supported.shape!=(len(x),):
        raise ValueError("each proposal needs its own current-observation clearance flag")
    if not articulated:
        supported &= np.isclose(x[:,2],1.31) & np.isclose(x[:,6],0.) & np.isclose(x[:,7],1.)
    context=np.broadcast_to(np.concatenate((source,crop)),(len(x),2048)).copy()
    out=predictor(context,actions,x)
    index=correct_view(x,out["p_reach"],out["p_joint"],supported,initial)
    if index is None:
        return None
    return dict(index=index,action=actions[index][actions[index]!=0].tolist(),
                camera=x[index,:8].copy(),predicted_future=out["predicted_future"][index].copy(),
                reach=float(out["p_reach"][index]),visible_if_reached=float(out["p_visible_if_reach"][index]),
                joint=float(out["p_joint"][index]))
