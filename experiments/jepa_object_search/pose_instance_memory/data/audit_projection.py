"""Check visible surface agreement between different real rendered viewpoints."""
from collections import Counter,defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def sample(src,dst,root):
    source=np.load(root/src["depth"],allow_pickle=False)
    dest=np.load(root/dst["depth"],allow_pickle=False)
    u,v=np.meshgrid(np.arange(4,384,8),np.arange(4,384,8))
    z=source[v,u]
    m=np.isfinite(z)&(z>.1)&(z<15)
    x=np.stack([u[m]+.5,v[m]+.5,np.ones(m.sum())])
    cloud=np.linalg.solve(np.array(src["K"]),x)*z[m][None,:]
    ps=np.array(src["T_world_from_camera"])
    pd=np.array(dst["T_world_from_camera"])
    transformed=pd[:3,:3].T@(ps[:3,:3]@cloud+ps[:3,3:4]-pd[:3,3:4])
    uv=np.array(dst["K"])@transformed
    uv=uv[:2]/np.maximum(uv[2:3],1e-5)
    valid=(uv.min(axis=0)>=0)&(uv.max(axis=0)<384)&(transformed[2]>.1)
    tx,ty=uv[:,valid].astype(int)
    observed=dest[ty,tx]
    predicted=transformed[2,valid]
    agrees=np.abs(observed-predicted)<=np.maximum(.08,.025*predicted)
    with Image.open(root/src["rgb"]) as a, Image.open(root/dst["rgb"]) as b:
        rgb1=np.asarray(a,dtype=float)[v[m][valid][agrees],u[m][valid][agrees]]
        rgb2=np.asarray(b,dtype=float)[ty[agrees],tx[agrees]]
    return int(valid.sum()), int(agrees.sum()),float(abs(rgb1-rgb2).sum())


def audit(root):
    rows=[json.loads(s) for s in (root/"frames.jsonl").read_text().splitlines()]
    groups=defaultdict(list)
    for r in rows:
        groups[r["split"],r["sequence"]].append(r)
    d={}
    for split in ("train","val","test"):
        counts=Counter();color_total=0.;many=Counter();pair_counts=Counter();peaks=Counter()
        for (s,seq),frames in groups.items():
            if s!=split:continue
            for r in frames:
                cat=r["category"]; n=len(r["boxes"])
                peaks[cat]=max(peaks[cat],n);many[cat]+=n>8;pair_counts[cat]+=n*(n-1)//2
            for a,b in combinations(frames,2):
                inside,agreed,color=sample(a,b,root)
                counts["frame_pairs"]+=1;counts["projected_overlap_pixels"]+=inside
                counts["depth_consistent_pixels"]+=agreed;color_total+=color
        d[split]=dict(**counts,
                      depth_agreement_fraction=counts["depth_consistent_pixels"]/counts["projected_overlap_pixels"],
                      depth_consistent_rgb_mae_0_255=color_total/(3*counts["depth_consistent_pixels"]),
                      max_boxes_per_frame_by_category=dict(peaks),frames_over_8_boxes_by_category=dict(many),
                      covisible_annotation_pairs_by_category=dict(pair_counts))
    (root/"projection_audit.json").write_text(json.dumps(d,indent=2))
    print(json.dumps(d))


if __name__=="__main__":
    audit(Path(sys.argv[1]))
