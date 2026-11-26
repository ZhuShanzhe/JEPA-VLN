"""Replay bounded RGB history on real online mapper snapshots from accepted navigation."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

from sequence_tracking import chain_geometry


def run(p,arm):
    base=p/"identity_stage4_geometry"
    sys.path[:0]=[str(base/"provider"),str(p/"identity_stage3_model/train"),str(p/"jepa_reid_v21/association")]
    from optics import skew_view
    from identity import IdentityVerifier
    from map_views import rgb_aligned
    study=p/"mapping_runtime/pilots"/("identity_stage4_clock_v2_"+arm)
    eid=json.loads((study/"panel.json").read_bytes())["rows"][0]["episode_id"]
    ticket=json.loads((study/"comparison/accepted/second"/(eid+".json")).read_bytes())
    path=Path(ticket["result"])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==ticket["result_sha256"]
    frames=p/"mapping_runtime/runs"/ticket["evidence"]["server_run"]/"mapping/diagnostics"/(eid+"_frames")
    verifier=IdentityVerifier(p,"jepa","cpu")
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(173)
    @lru_cache(maxsize=64)
    def buffered(fid):
        return cv2.imread(str(frames/f"rgb_{fid:06d}.jpg"),cv2.IMREAD_GRAYSCALE)
    def gray(im):
        return cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    def img(f):
        bgr=cv2.imread(str(f))
        return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    counts=Counter()
    reject=Counter()
    verdicts=Counter()
    pair_steps=set()
    usable_steps=set()
    for marker in sorted((path.parent/"private_stage4_pairs").glob("*/complete.json")):
        rec=json.loads(marker.read_bytes())
        folder=marker.parent
        live=img(folder/"live.png")
        current=rec["step"]+1
        archive_live=buffered(current)
        assert archive_live is not None and archive_live.shape==live.shape[:2]
        err=np.mean(np.abs(archive_live.astype(float)-gray(live)))
        assert err<8,"endpoint did not match its timestamp"
        bs=np.asarray(rec["live_boxes_work_width"],float).reshape(-1,4)*live.shape[1]/rec["work_width"]
        counts["pairs"]+=1
        pair_steps.add(current)
        scale=rec["ruler"]["scale"]
        source=rec["source"]
        if source not in rec["anchors"] or not scale or not len(bs):
            reject["missing_source_scale_or_candidates"]+=1
            continue
        def load_anchor(k):
            rgb=img(folder/f"anchor_{k}.png")
            with np.load(folder/f"anchor_{k}.npz",allow_pickle=False) as z:
                snap={key:z[key] for key in z.files}
            if not rgb_aligned(snap,rgb,8):
                return rgb,None
            return rgb,skew_view(snap,scale)
        src,sv=load_anchor(source)
        if sv is None:
            reject["source_view_invalid"]+=1
            continue
        counts["source_view_valid"]+=1
        last="no_recent_committed_anchor"
        delivered=False
        for k in sorted(rec["anchors"],reverse=True):
            if not 1<=current-k<=24:
                continue
            rgb,view=load_anchor(k)
            if view is None:
                continue
            size=rgb.shape[1::-1]
            initial=cv2.resize(buffered(k),size,interpolation=cv2.INTER_AREA)
            if np.mean(np.abs(initial.astype(float)-gray(rgb)))>10:
                last="anchor_image_mismatch"
                continue
            seq=[gray(rgb)]
            for i in range(k+1,current):
                seq.append(cv2.resize(buffered(i),size,interpolation=cv2.INTER_AREA))
            seq.append(cv2.resize(gray(live),size,interpolation=cv2.INTER_AREA))
            geom,last=chain_geometry(view,seq,live.shape[:2])
            if geom is None:
                continue
            depth,K,pose=geom
            if not delivered:
                counts["current_pose_available"]+=1
                delivered=True
            support=[np.isfinite(depth[max(0,int(y)):int(np.ceil(v)),max(0,int(x)):int(np.ceil(u))]).sum() for x,y,u,v in bs]
            if min(support)<9:
                last="object_window_lacks_depth"
                continue
            sh,sw=src.shape[:2]
            z=(sv["grid"]-sv["c2w"][:3,3])@sv["c2w"][:3,2]
            sd=np.full((sh,sw),np.nan,np.float32)
            sd[::8,::8]=np.where((z>.01)&(z<100),z,np.nan)
            roi=np.array(rec["source_normalized_roi"])*[sw,sh,sw,sh]
            a=verifier.encode(src,[roi],sd,sv["K"],sv["c2w"])
            b=verifier.encode(live,bs,depth,K,pose)
            scores=verifier.score(a,b)
            if not np.isfinite(scores).all():
                last="frozen_head_requires_more_support"
                continue
            counts["identity_scored"]+=1
            usable_steps.add(current)
            decision,=verifier.decide(scores)
            verdicts[decision["state"]]+=1
            break
        else:
            reject[last]+=1
    assert counts["identity_scored"]+sum(reject.values())==counts["pairs"]
    return dict(counts=counts,distinct_steps=len(pair_steps),scored_steps=len(usable_steps),
                terminal_reasons=reject,identity_verdicts=verdicts)


if __name__=="__main__":
    p=Path(sys.argv[1])
    arm=sys.argv[2]
    result=run(p,arm)
    Path(sys.argv[3]).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(result)
