"""Calibrate an RGB proposal's localization quality, independent of target identity."""
import math
import re


def coarse_category(text):
    groups=(("clock",r"\bclocks?\b"),("painting",r"\b(?:paintings?|pictures?|wall art|wall_art|artworks?)\b"))
    hits=[name for name,pattern in groups if re.search(pattern,str(text or ""),re.I)]
    return hits[0] if len(hits)==1 else str(text or "").strip()[:128]


def features(row, width, height):
    x,y,u,v=row.get("xyxy", row.get("bbox"))
    x,u=x/width,u/width
    y,v=y/height,v/height
    w,h=max(1e-5,u-x),max(1e-5,v-y)
    return [float(row["score"]),math.log(w*h),abs(math.log(w/h)),min(x,y,1-u,1-v),
            float(row.get("view","full")=="full"), abs((x+u)/2-.5), abs((y+v)/2-.5)]


def quality(spec,row,width,height):
    if spec["mode"]=="detector":
        return float(row["score"])
    z=spec["bias"]
    for a,b,c,d in zip(features(row,width,height),spec["mean"],spec["scale"],spec["weights"]):
        z+=(a-b)/c*d
    return 1/(1+math.exp(-min(40.,max(-40.,z))))


def best(spec,rows,width,height):
    return max(((quality(spec,r,width,height),i) for i,r in enumerate(rows)),default=(0.,None))


def overlap(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)
