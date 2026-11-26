"""Fit each proposal boundary to observed RGB edges with a learned alignment score."""
import cv2
import numpy as np


OFFSETS = np.array([-.42, -.30, -.20, -.12, -.07, -.035, 0, .035, .07, .12, .20, .32, .50, .75, 1.0])


def image_channels(rgb):
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected uint8 RGB pixels")
    lab = cv2.cvtColor(np.ascontiguousarray(image[:, :, :3]), cv2.COLOR_RGB2LAB).astype("f") / 255
    return cv2.GaussianBlur(lab, (3, 3), .65)


def edge_options(image, box, score, side):
    h, w = image.shape[:2]
    axis = side % 2
    extent = max(1., box[axis+2]-box[axis])
    cross = max(1., box[3-axis]-box[1-axis])
    limit, other_limit = (w,h) if axis==0 else (h,w)
    sign = -1 if side<2 else 1
    positions = np.clip(box[side]+sign*extent*OFFSETS, 0, limit)
    sample = (box[1-axis]+cross*np.linspace(.07,.93,13)).astype(int).clip(0,other_limit-1)
    def fetch(d):
        indices=np.rint(positions-.5+sign*d).astype(int).clip(0,limit-1)
        if axis==0:
            return image[sample[None,:],indices[:,None]]
        return image[indices[:,None],sample[None,:]]
    offset = (positions-box[side])/extent*sign
    z=np.flatnonzero(OFFSETS==0)[0]
    features=[np.abs(offset), offset**2, offset, np.abs(offset)*score, np.abs(offset)*np.log1p(extent),
              np.abs(offset)*(axis==1), np.abs(offset)*(sign>0)]
    for d in (1, max(2,.035*extent), max(4,.11*extent)):
        inward, outward=fetch(-d),fetch(d)
        delta=np.linalg.norm(outward-inward,axis=2)
        within=np.linalg.norm(fetch(-2*d)-inward,axis=2)
        beyond=np.linalg.norm(fetch(2*d)-outward,axis=2)
        for cue in (delta.mean(1),np.quantile(delta,.25,axis=1),np.quantile(delta,.75,axis=1),
                    np.mean(delta/(.035+within+beyond),axis=1)):
            features.extend((cue, cue-cue[z], cue*np.abs(offset)))
    # Edges on the frame boundary are partly unobservable.
    features.append(((positions<=1)|(positions>=limit-1)).astype(float))
    return positions,np.column_stack(features).astype("f")


class BoxRefiner:
    """Input/output XYXY in ORIGINAL RGB pixels. Scores, frame IDs and order are preserved."""
    def __init__(self, checkpoint):
        with np.load(checkpoint,allow_pickle=False) as z:
            self.mean,self.scale,self.coef=(z[k] for k in ("mean","scale","coef"))
            self.strength=float(z["strength"].item())

    def refine(self,rgb,proposal_rows,category=None):
        if not proposal_rows:
            return []
        image=image_channels(rgb)
        output=[]
        for row in proposal_rows:
            original=np.asarray(row.get("xyxy",row.get("bbox")),float)
            h,w=image.shape[:2]
            if original.shape!=(4,) or not np.isfinite(original).all() or (original[2:]<=original[:2]).any():
                raise ValueError("Invalid source box")
            box=np.clip(original,[0,0,0,0],[w,h,w,h])
            if (box[2:]<=box[:2]).any():
                raise ValueError("Source box outside RGB")
            score=float(row["score"])
            if not np.isfinite(score):
                raise ValueError("Invalid detector score")
            updated=box.copy()
            for side in range(4):
                pos,feat=edge_options(image,box,score,side)
                best=int(np.argmax((feat-self.mean)/self.scale@self.coef))
                updated[side]+=self.strength*(pos[best]-box[side])
            if np.any(updated[2:]-updated[:2]<1):
                updated=box
            xyxy=[float(x) for x in updated]
            output.append(dict(row,xyxy=xyxy,bbox=xyxy.copy()))
        return output
