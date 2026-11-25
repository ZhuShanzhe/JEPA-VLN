"""Persistent offline detector on a local pair of pipes, with explicit timeouts."""
import atexit
import io
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import time

import numpy as np


class ProposalWorker:
    def __init__(self,python,project,device="cpu",mode=None):
        self.proc=subprocess.Popen([str(python),"-u",str(Path(__file__).resolve()),
                                    str(project),device,mode or "configured"],
                                   stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                   env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
        atexit.register(self.close)
        try:
            if self.read(160)!={"ready":True}:
                raise RuntimeError("Detector failed startup")
        except Exception:
            self.close()
            raise

    def read(self,timeout):
        deadline=time.monotonic()+timeout
        out=bytearray()
        while len(out)<8_000_000:
            left=deadline-time.monotonic()
            if left<=0 or not select.select([self.proc.stdout],[],[],left)[0]:
                raise TimeoutError("Local detector timed out")
            chunk=os.read(self.proc.stdout.fileno(),16384)
            if not chunk:
                raise EOFError("Local detector exited")
            out.extend(chunk)
            if out.endswith(b"\n"):
                return json.loads(out)
        raise ValueError("Detector response too large")

    def detect(self,rgb,category):
        blob=io.BytesIO()
        np.savez(blob,rgb=rgb,category=str(category))
        raw=blob.getvalue()
        self.proc.stdin.write(struct.pack(">I",len(raw))+raw)
        self.proc.stdin.flush()
        row=self.read(90)
        if "error" in row:
            raise RuntimeError(row["error"])
        return row["boxes"]

    def close(self):
        atexit.unregister(self.close)
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc.stdin.close()
        self.proc.stdout.close()


def serve(project,device,mode):
    reply=sys.stdout
    sys.stdout=sys.stderr
    from proposals import LocalCandidateDetector
    model=LocalCandidateDetector(project,device,mode)
    print(json.dumps({"ready":True}),file=reply,flush=True)
    stream=sys.stdin.buffer
    while True:
        header=stream.read(4)
        if not header:
            return
        n,=struct.unpack(">I",header)
        if n>64*1024*1024:
            raise ValueError("Frame too large")
        data=stream.read(n)
        if len(data)!=n:
            raise EOFError("Partial frame")
        try:
            with np.load(io.BytesIO(data),allow_pickle=False) as x:
                result={"boxes":model.detect(x["rgb"],x["category"].item())}
        except Exception as exc:
            result={"error":type(exc).__name__}
        print(json.dumps(result,allow_nan=False),file=reply,flush=True)


if __name__=="__main__":
    serve(Path(sys.argv[1]),sys.argv[2],None if sys.argv[3]=="configured" else sys.argv[3])
