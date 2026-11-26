"""Offline frozen instance-verification API. Invalid or ambiguous evidence is withheld."""
from pathlib import Path
import sys
import numpy as np
import torch


def imports(project):
    for path in (project/"jepa_reid_v20/model",project/"jepa_reid_v21/model"):
        if str(path) not in sys.path:
            sys.path.insert(0,str(path))
    from encode_proposals import evidence_image,support
    from reid_model import Encoder,mask_context,preprocess
    from pose_head import PoseHead
    from pose_inference import score_matrix
    return Encoder,mask_context,preprocess,evidence_image,support,PoseHead,score_matrix


def encode_frame_from_features(features,points,c2w):
    return dict(features=np.asarray(features,dtype=np.float32),points=list(points),c2w=c2w)


class IdentityVerifier:
    def __init__(self,project,mode="jepa",device="cuda",checkpoint_dir=None):
        if mode not in ("jepa","direct"):
            raise ValueError("unknown verification objective")
        project=Path(project)
        self.device=device
        self.utils=imports(project)
        self.project=project
        self.mode=mode
        self.encoder=None
        weights=project/"identity_stage3_model/train/private" if checkpoint_dir is None else Path(checkpoint_dir)
        path=weights/(mode+".pt")
        ck=torch.load(path,map_location="cpu",weights_only=True)
        if ck["mode"]!=mode:
            raise ValueError("checkpoint objective mismatch")
        self.head=self.utils[-2]().to(device).eval()
        self.head.load_state_dict(ck["head"],strict=True)
        self.high,self.low,self.margin=(ck[k] for k in ("high","low","margin"))

    @torch.inference_mode()
    def encode(self,rgb,xyxy_list,depth,K,c2w):
        Encoder,mask,preprocess,evidence,support,*_=self.utils
        rgb=np.asarray(rgb)
        if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:
            raise ValueError("expected uint8 RGB")
        h,w=rgb.shape[:2]
        depth=np.asarray(depth)
        K=np.asarray(K)
        pose=np.asarray(c2w)
        indices,crops=[],[]
        points=[None]*len(xyxy_list)
        features=np.zeros((len(points),2304),dtype="float32")
        if depth.shape!=(h,w) or K.shape!=(3,3) or pose.shape!=(4,4) or not (
            np.isfinite(K).all() and np.isfinite(pose).all() and K[0,0]>0 and K[1,1]>0
            and abs(np.linalg.det(K))>1e-8 and np.allclose(pose[3],[0,0,0,1],atol=1e-5)
            and np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=.005)
            and np.linalg.det(pose[:3,:3])>.99):
            return encode_frame_from_features(features,points,None)
        for i,box in enumerate(xyxy_list):
            box=np.asarray(box,float)
            if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[:2]<0) or np.any(box[2:]>[w,h]) or min(box[2:]-box[:2])<2:
                continue
            point=support(depth,box,K,pose)
            if point is None or not np.isfinite(point).all():
                continue
            indices.append(i)
            crops.append(evidence(rgb,box,mask,preprocess))
            points[i]=point
        if crops and self.encoder is None:
            self.encoder=Encoder(self.project).to(self.device).eval()
        for start in range(0,len(crops),16):
            a,b=zip(*crops[start:start+16])
            features[indices[start:start+16]]=self.encoder(torch.stack(a).to(self.device),torch.stack(b).to(self.device)).cpu().numpy()
        return encode_frame_from_features(features,points,pose)

    @torch.inference_mode()
    def score(self,source,live):
        result=self.utils[-1](self.head,"pose_jepa",source,live)
        return np.where(np.isfinite(result),result,np.nan)

    def decide(self,matrix):
        matrix=np.asarray(matrix)
        if matrix.ndim!=2:
            raise ValueError("expected a source by live score matrix")
        decisions=[]
        for scores in matrix:
            state,index="uncertain",None
            if len(scores) and np.isfinite(scores).all():
                order=np.argsort(-scores,kind="stable")
                first=order[0]
                if scores[first]>=self.high and (len(order)==1 or scores[first]-scores[order[1]]>=self.margin):
                    state,index="same",int(first)
                elif scores[first]<=self.low:
                    state="different"
            decisions.append(dict(state=state,index=index))
        # Two independent memories competing for a single current box do not
        # establish the identity of either. A caller may pass known aliases separately.
        seen={}
        for n,decision in enumerate(decisions):
            if decision["state"]=="same":
                seen.setdefault(decision["index"],[]).append(n)
        for conflicts in seen.values():
            if len(conflicts)>1:
                for n in conflicts:
                    decisions[n]=dict(state="uncertain",index=None)
        return decisions
