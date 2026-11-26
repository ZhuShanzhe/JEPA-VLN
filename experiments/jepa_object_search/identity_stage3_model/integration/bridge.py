"""Exchange two measured RGB-D observations with the frozen local identity head."""
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np


def load_verifier(project, mode, device):
    path=project/"identity_stage3_model/train/identity.py"
    spec=importlib.util.spec_from_file_location("stage3_identity_verification",path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.IdentityVerifier(project,mode,device)


def infer(verifier, source, live):
    src=verifier.encode(*source)
    cur=verifier.encode(*live)
    scores=verifier.score(src,cur)
    result,=verifier.decide(scores)
    return result


class Worker:
    """The MOST process and the learned encoder run in their original environments."""
    def __init__(self, project, mode, device):
        import atexit
        from proposal_worker import ProposalWorker
        self.proc=subprocess.Popen([str(project/"b_verification/env/bin/python"),"-u",__file__,
                                    str(project),mode,device],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                    env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
        self.read=lambda timeout:ProposalWorker.read(self,timeout)
        try:
            if self.read(120)!={"ready":True}:
                raise RuntimeError("identity worker startup failed")
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def decide(self, src, live):
        buf=io.BytesIO()
        parts={f"{side}{i}":np.asarray(value) for side,obs in (("s",src),("l",live)) for i,value in enumerate(obs)}
        np.savez(buf,**parts)
        msg=buf.getvalue()
        if len(msg)>128*1024**2:
            raise ValueError("input too large")
        self.proc.stdin.write(struct.pack(">I",len(msg))+msg)
        self.proc.stdin.flush()
        verdict=self.read(120)
        if "error" in verdict:
            raise ValueError(verdict["error"])
        return verdict

    def close(self):
        import atexit
        atexit.unregister(self.close)
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc.stdin.close()
        self.proc.stdout.close()


def serve(project,mode,device):
    output=sys.stdout
    sys.stdout=sys.stderr
    model=load_verifier(project,mode,device)
    print(json.dumps(dict(ready=True)),file=output,flush=True)
    stream=sys.stdin.buffer
    while size:=stream.read(4):
        n,=struct.unpack(">I",size)
        if n>128*1024**2:
            raise ValueError("oversized observation")
        try:
            with np.load(io.BytesIO(stream.read(n)),allow_pickle=False) as arrays:
                sides=[[arrays[f"{tag}{i}"] for i in range(5)] for tag in ("s","l")]
                reply=infer(model,*sides)
        except (ValueError,RuntimeError) as e:
            reply={"error":type(e).__name__}
        print(json.dumps(reply,allow_nan=False),file=output,flush=True)


if __name__=="__main__":
    serve(Path(sys.argv[1]),sys.argv[2],sys.argv[3])
