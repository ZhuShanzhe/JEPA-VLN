"""Local RGB services for original masks and free crop ranking in their own environments."""
import atexit
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np


class LocalBoxService:
    def __init__(self,python,project,service,device):
        from proposal_worker import ProposalWorker
        self.proc=subprocess.Popen([str(python),"-u",__file__,str(project),service,device],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            env={**os.environ,"TRANSFORMERS_OFFLINE":"1","HF_HUB_OFFLINE":"1"})
        self.read=lambda delay:ProposalWorker.read(self,delay)
        atexit.register(self.close)
        try:
            if self.read(180)!={"ready":True}:
                raise RuntimeError(service+" failed to initialize")
        except BaseException:
            self.close()
            raise

    def query(self,rgb,category,boxes=()):
        buf=io.BytesIO()
        np.savez(buf,rgb=np.ascontiguousarray(rgb),category=str(category),boxes=json.dumps(boxes,allow_nan=False))
        blob=buf.getvalue()
        self.proc.stdin.write(struct.pack(">I",len(blob))+blob)
        self.proc.stdin.flush()
        reply=self.read(180)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["boxes"]

    def close(self):
        from proposal_worker import ProposalWorker
        ProposalWorker.close(self)


def serve(project,service,device):
    reply=sys.stdout
    sys.stdout=sys.stderr
    base=Path(__file__).resolve().parents[1]
    if service=="sam":
        sys.path.insert(0,str(base/"sam"))
        from native_regions import NativeRegions
        engine=NativeRegions(project,device)
    elif service=="review":
        sys.path.insert(0,str(base/"compare"))
        from region_rank import RegionRanker
        engine=RegionRanker(project,device)
    else:
        raise ValueError(service)
    print(json.dumps({"ready":True}),file=reply,flush=True)
    while header:=sys.stdin.buffer.read(4):
        if len(header)!=4:
            raise EOFError("Partial header")
        n,=struct.unpack(">I",header)
        if n>64*1024*1024:
            raise ValueError("Frame exceeds limit")
        blob=sys.stdin.buffer.read(n)
        if len(blob)!=n:
            raise EOFError("Partial image")
        try:
            with np.load(io.BytesIO(blob),allow_pickle=False) as data:
                image=data["rgb"]
                if service=="sam":
                    result=engine.propose(image)
                else:
                    result=engine.rank(image,json.loads(data["boxes"].item()),data["category"].item())
            answer={"boxes":result}
        except Exception as exc:
            answer={"error":service+":"+type(exc).__name__}
        print(json.dumps(answer,allow_nan=False),file=reply,flush=True)


if __name__=="__main__":
    serve(Path(sys.argv[1]),sys.argv[2],sys.argv[3])
