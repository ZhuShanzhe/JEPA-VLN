"""Score already reachable verification views using only the remembered RGB region and relative pose."""
import numpy as np
from PIL import Image


def source_vector(crop, box, shape):
    h,w=shape[:2]
    x,y,u,v=np.asarray(box,dtype=np.float32)/[w,h,w,h]
    image=np.asarray(Image.fromarray(np.ascontiguousarray(crop[:,:,:3],dtype=np.uint8)).resize(
        (24,24), Image.Resampling.BILINEAR),dtype=np.float32)/255.
    tiles=image.reshape(4,6,4,6,3)
    means=tiles.mean(axis=(1,3)).ravel()
    std=tiles.std(axis=(1,3)).ravel()
    return np.r_[means,std,x,y,u,v,np.log(max(.0001,(u-x)*(v-y)))].astype("f")


def motion_vector(candidates):
    p=np.asarray(candidates,dtype="f").reshape(-1,3)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite relative camera motion")
    f,r,y=p.T
    return np.stack([f,r,np.hypot(f,r),np.sin(y),np.cos(y),f*np.cos(y)+r*np.sin(y),
                     np.abs(r),np.abs(np.arctan2(np.sin(y),np.cos(y)))],axis=1)


class ActiveViewScorer:
    """forward, right in metres; yaw positive right in radians from the reference camera."""
    def __init__(self, checkpoint):
        with np.load(checkpoint,allow_pickle=False) as z:
            self.w={key:z[key] for key in z.files}

    def score(self,rgb,box_xyxy_px,candidates):
        if len(candidates)==0:
            return np.empty(0,dtype="f")
        rgb=np.asarray(rgb)
        if rgb.ndim!=3 or rgb.shape[2]<3 or rgb.dtype!=np.uint8:
            raise ValueError("Expected uint8 RGB or RGBA")
        h,w=rgb.shape[:2]
        box=np.asarray(box_xyxy_px,dtype=float)
        if box.shape!=(4,) or not np.isfinite(box).all():
            raise ValueError("Invalid reference box")
        x,y,u,v=np.clip(np.r_[np.floor(box[:2]),np.ceil(box[2:])],[0,0,0,0],[w,h,w,h]).astype(int)
        if u<=x or v<=y:
            raise ValueError("Empty reference crop")
        features=source_vector(rgb[y:v,x:u],(x,y,u,v),rgb.shape)
        m=motion_vector(candidates)
        a=np.column_stack([np.repeat(features[None],len(m),axis=0),m])
        return self._predict(a)

    def _predict(self,a):
        w=self.w
        a=np.clip((a-w["mean"])/w["scale"],-8,8)
        a=np.maximum(a@w["w1"].T+w["b1"],0)
        a=np.maximum(a@w["w2"].T+w["b2"],0)
        return (a@w["w3"].T+w["b3"]).ravel()
