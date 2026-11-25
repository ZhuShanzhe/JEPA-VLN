"""Short-term appearance evidence tied to the original RGB capture and canonical object."""
import cv2
import numpy as np


def small_gray(rgb):
    gray = cv2.cvtColor(np.asarray(rgb, np.uint8)[..., :3], cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (320, 320), interpolation=cv2.INTER_AREA)


def evidence_patch(jpeg, point):
    gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    p = np.asarray(point,dtype=float)
    if gray is None or p.shape != (2,) or not np.isfinite(p).all() or not (0 <= p).all() or not (p <= 1000).all():
        return None
    gray = cv2.resize(gray,(320,320),interpolation=cv2.INTER_AREA)
    x,y = np.rint(p * 319 / 1000).astype(int)
    crop = gray[max(0,y-14):min(320,y+15),max(0,x-14):min(320,x+15)].copy()
    if min(crop.shape)<18 or np.std(crop)<9:
        return None
    return crop


def live_match(template, rgb):
    gray = small_gray(rgb)
    best = (0.,0.)
    for scale in (.65,.82,1.,1.25,1.65,2.):
        patch = cv2.resize(template,None,fx=scale,fy=scale)
        if np.std(patch)<6:
            continue
        corr = cv2.matchTemplate(gray,patch,cv2.TM_CCOEFF_NORMED)
        _,v,_,(x,y) = cv2.minMaxLoc(corr)
        h,w = patch.shape
        corr[max(0,y-h//2):y+h//2+1,max(0,x-w//2):x+w//2+1] = -1.
        gap = v-float(corr.max())
        if v > best[0]:
            best = (v,gap)
    return best[0]>=.77 and best[1]>=.055,round(best[0],4),round(best[1],4)
