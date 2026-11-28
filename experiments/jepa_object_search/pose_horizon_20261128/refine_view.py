"""Local camera correction for one visual cue and already chosen region. No route scores or evaluator labels."""
import math
import numpy as np


def correct_view(pose, p_reach, p_joint, feasible, origin, *, confidence=.5, gain=.05):
    """Return an index into the offered local cameras, or None if no supported safe camera remains."""
    pose=np.asarray(pose)
    r=np.asarray(p_reach)
    j=np.asarray(p_joint)
    feasible=np.asarray(feasible,dtype=bool)
    if pose.ndim!=2 or pose.shape[1]<8 or any(x.shape!=(len(pose),) for x in (r,j,feasible)):
        raise ValueError("candidate arrays must share a row for each camera")
    if not 0<=origin<len(pose) or not np.isfinite(pose[:,:8]).all():
        raise ValueError("invalid camera")
    if not np.isfinite(r).all() or not np.isfinite(j).all() or np.any((j<0)|(r>1)|(j>r+1e-6)):
        raise ValueError("expected coherent joint and reach probabilities")
    if feasible[origin] and j[origin]>=confidence:
        return origin
    dxz=pose[:,:2]-pose[origin,:2]
    distance=np.linalg.norm(dxz,axis=1)
    height=abs(pose[:,2]-pose[origin,2])
    yaw=np.arctan2(pose[:,4],pose[:,5])
    yaw_diff=abs(np.angle(np.exp(1j*(yaw-yaw[origin]))))
    pitch=np.arctan2(pose[:,6],pose[:,7])
    pitch_diff=abs(pitch-pitch[origin])
    supported=feasible & (distance<=1.5+1e-5) & (height<=.5+1e-5) & (yaw_diff<=math.pi/2+1e-5)
    supported &= pitch_diff<=math.radians(40)+1e-5
    # Movement clearance remains mandatory at execution; this gate only admits measured alternatives.
    indices=np.flatnonzero(supported & (r>=confidence) & (j>=confidence))
    if len(indices):
        # Equivalent scores favor the smallest change from the current proposal.
        best=min(indices,key=lambda i:(-float(j[i]),float(distance[i]),float(height[i]),
                                      float(yaw_diff[i]),float(pitch_diff[i]),int(i)))
        if not feasible[origin] or j[best]>=j[origin]+gain:
            return int(best)
    return int(origin) if feasible[origin] else None
