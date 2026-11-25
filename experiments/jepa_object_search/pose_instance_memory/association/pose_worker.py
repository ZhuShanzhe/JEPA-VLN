"""Score metric cross-view observations offline in the research policy environment."""
import atexit
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np


class PoseWorker:
    def __init__(self, interpreter, project, weights, device):
        from model_worker import MatcherProcess
        self.pipe=MatcherProcess.__new__(MatcherProcess)
        self.pipe.process=subprocess.Popen(
            [str(interpreter),"-u",str(Path(__file__).resolve()),str(project),str(weights),device],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1"))
        atexit.register(self.pipe.close)
        try:
            if self.pipe.read(150)!={"ready":True}:
                raise RuntimeError("Invalid pose worker response")
        except Exception:
            self.pipe.close()
            raise

    def compare(self,left,right):
        arrays={}
        for k,(region,pose,point) in zip(("a","b"),(left,right)):
            arrays.update({k+"obj":region[0],k+"ctx":region[1],k+"box":region[2],
                           k+"pose":pose,k+"point":point})
        stream=io.BytesIO()
        np.savez(stream,**arrays)
        data=stream.getvalue()
        self.pipe.process.stdin.write(struct.pack(">I",len(data))+data)
        self.pipe.process.stdin.flush()
        result=self.pipe.read(75)
        if "error" in result:
            raise ValueError(result["error"])
        return result


def serve(project,weights,device):
    channel=sys.stdout
    sys.stdout=sys.stderr
    sys.path[:0]=[str(project/"jepa_reid_v21/model"),str(project/"jepa_reid_v20/model")]
    import torch
    from pose_inference import load_head,score_matrix
    from reid_model import Encoder,mask_context,preprocess
    torch.set_num_threads(3)
    enc=Encoder(project).to(device).eval()
    head,mode,threshold=load_head(weights,device)
    if mode!="pose_jepa":
        raise ValueError("Expected pose_jepa checkpoint")
    def frame(q,k):
        c=mask_context(q[k+"ctx"],q[k+"box"])
        return dict(c2w=q[k+"pose"],points=[q[k+"point"]],
                    features=enc(preprocess(q[k+"obj"])[None].to(device),preprocess(c)[None].to(device)).cpu().numpy())
    print(json.dumps({"ready":True}),file=channel,flush=True)
    while header:=sys.stdin.buffer.read(4):
        length,=struct.unpack(">I",header)
        if length>64*1024*1024:
            raise ValueError("Oversized input")
        with np.load(io.BytesIO(sys.stdin.buffer.read(length)),allow_pickle=False) as q:
            try:
                score=float(score_matrix(head,mode,frame(q,"a"),frame(q,"b"))[0,0])
                if not np.isfinite(score):
                    raise ValueError("Incomplete camera geometry")
                result=dict(score=score,threshold=threshold,match=score>=threshold)
            except (ValueError,RuntimeError,KeyError) as exc:
                result=dict(error=type(exc).__name__)
        print(json.dumps(result,allow_nan=False),file=channel,flush=True)


if __name__=="__main__":
    serve(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
