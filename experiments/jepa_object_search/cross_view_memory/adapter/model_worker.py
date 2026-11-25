"""Run the optional frozen matcher in its policy environment over two local pipes."""
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


class MatcherProcess:
    def __init__(self, python, project, weights, device):
        script = Path(__file__).resolve()
        self.process = subprocess.Popen(
            [str(python), "-u", str(script), str(project), str(weights), str(device)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        atexit.register(self.close)
        try:
            if self.read(150) != {"ready": True}:
                raise RuntimeError("matcher startup failed")
        except Exception:
            self.close()
            raise

    def read(self, seconds):
        pipe = self.process.stdout
        deadline = time.monotonic() + seconds
        line = bytearray()
        while len(line) < 4096:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([pipe], [], [], remaining)[0]:
                raise TimeoutError("re-identification worker timed out")
            byte = os.read(pipe.fileno(), 1)
            if not byte:
                raise EOFError("re-identification worker exited")
            if byte == b"\n":
                return json.loads(line)
            line.extend(byte)
        raise ValueError("oversized worker response")

    def compare(self, source, candidate, yaw):
        buf = io.BytesIO()
        np.savez(buf, source=source[0], a_context=source[1], a_box=source[2],
                 candidate=candidate[0], b_context=candidate[1], b_box=candidate[2], yaw=yaw)
        payload = buf.getvalue()
        self.process.stdin.write(struct.pack(">I", len(payload)))
        self.process.stdin.write(payload)
        self.process.stdin.flush()
        response = self.read(60)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        atexit.unregister(self.close)


def serve(project, weights, device):
    response_pipe = sys.stdout
    sys.stdout = sys.stderr  # Encoder logging cannot corrupt the response protocol.
    sys.path.insert(0, str(project / "jepa_reid_v20/model"))
    from reid_model import load_matcher, mask_context
    model = load_matcher(project, weights, device=device)
    print(json.dumps({"ready": True}), file=response_pipe, flush=True)
    stream = sys.stdin.buffer
    while header := stream.read(4):
        length, = struct.unpack(">I", header)
        if length > 64 * 1024 * 1024:
            raise ValueError("image packet exceeds maximum")
        with np.load(io.BytesIO(stream.read(length)), allow_pickle=False) as packet:
            try:
                a = mask_context(packet["a_context"], packet["a_box"].tolist())
                b = mask_context(packet["b_context"], packet["b_box"].tolist())
                result = model.compare(packet["source"], packet["candidate"], float(packet["yaw"]),
                                       reference_context=a, candidate_context=b)
                row = {"score": float(result["score"]), "threshold":float(result["threshold"]),
                       "match":bool(result["match"])}
            except Exception as exc:
                row = {"error": type(exc).__name__}
        print(json.dumps(row, allow_nan=False), file=response_pipe, flush=True)


if __name__ == "__main__":
    serve(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
