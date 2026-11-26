"""Context-conditioned inspection score for potentially incomplete reference regions."""
import numpy as np
from active_view_score import ActiveViewScorer, motion_vector, source_vector


def context_features(image,box):
    h,w=image.shape[:2]
    x,y,u,v=np.clip(np.r_[np.floor(box[:2]),np.ceil(box[2:])],[0,0,0,0],[w,h,w,h]).astype(int)
    if u<=x or v<=y:
        raise ValueError("Empty reference region")
    dx,dy=max(u-x,.10*w),max(v-y,.10*h)
    a,b=max(0,int(x-dx)),max(0,int(y-dy))
    c,d=min(w,int(u+dx+1)),min(h,int(v+dy+1))
    ctx=np.array(image[b:d,a:c,:3],copy=True)
    local=source_vector(image[y:v,x:u,:3],(x,y,u,v),image.shape)
    ctx[y-b:v-b,x-a:u-a]=128
    return np.r_[local,source_vector(ctx,(a,b,c,d),image.shape)].astype("f")


class WeakViewScorer(ActiveViewScorer):
    def score(self,rgb,box_xyxy_px,candidates):
        if not len(candidates):
            return np.zeros(0,dtype="f")
        if not np.isfinite(box_xyxy_px).all() or np.asarray(rgb).dtype!=np.uint8:
            raise ValueError("RGB and finite region required")
        vec=context_features(np.asarray(rgb),box_xyxy_px)
        actions=motion_vector(candidates)
        x=np.column_stack([np.repeat(vec[None],len(actions),axis=0),actions])
        return self._predict(x)
