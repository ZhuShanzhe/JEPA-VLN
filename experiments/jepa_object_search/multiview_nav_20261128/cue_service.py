"""Online handoff from three observed RGBs through frozen DINO and pinned GPT-4o."""
import atexit
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image

from cue_render import CAMERAS, MODEL, PREFIXES, encode, materials

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / "committed_inspect_20261128/private_dino/gpt_multiview"


def log(path, record):
    with path.open("ab") as f:
        f.write(encode(record) + b"\n")
        f.flush()
        os.fsync(f.fileno())


class CueSelector:
    def __init__(self, log_dir, *, allow_new=True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.calls = self.log_dir / "cue_calls.jsonl"
        self.decisions = self.log_dir / "cue_decisions.jsonl"
        self.last_receipt = None
        self.allow_new = allow_new
        self.worker = None
        self.old = {}
        for path in sorted(OLD.glob("*.response.json")):
            rec = json.loads(path.read_text())
            dispatched = json.loads(path.with_name(path.name.replace("response","dispatch")).read_text())
            assert dispatched["model"] == rec["model"] == MODEL and rec["request_hash"] == dispatched["request_hash"]
            self.old[rec["request_hash"]] = rec
        atexit.register(self.close)

    def close(self):
        if self.worker is not None:
            self.worker.stdin.close()
            self.worker.wait(timeout=30)
            self.worker.stdout.close()
            self.worker = None

    def _propose(self, paths, goal):
        if self.worker is None:
            env = {k:v for k,v in os.environ.items() if k not in {"PYTHONPATH","PYTHONHOME"} and not k.startswith("NAV_VLM_")}
            env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="2")
            with (self.log_dir / "dino.log").open("ab") as logs:
                self.worker = subprocess.Popen([str(ROOT / "b_verification/env/bin/python"), "-u", str(HERE / "cue_worker.py")],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=logs, env=env, text=True, cwd=str(ROOT))
            if json.loads(self.worker.stdout.readline()) != {"status":"ready"}:
                raise RuntimeError("detector startup failed")
        self.worker.stdin.write(json.dumps(dict(frames=[str(p) for p in paths], goal=goal))+"\n")
        self.worker.stdin.flush()
        predictions = json.loads(self.worker.stdout.readline())
        assert isinstance(predictions,list) and len(predictions)==3
        return predictions

    def _choose(self, req_hash, wire, offered):
        base = dict(request_hash=req_hash, model=MODEL)
        with (self.log_dir / "cue_lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            history = []
            if self.calls.exists():
                with self.calls.open() as f:
                    history = [e for e in map(json.loads,f) if e["request_hash"]==req_hash]
            prior = next((e for e in history if e["event"]=="success"),None)
            if req_hash in self.old:
                choice = self.old[req_hash]["choice"]
                kind = "prior_cache"
            elif prior:
                choice,kind = prior["choice"],"run_cache"
            else:
                known_denial = bool(history and history[-1]["event"] == "error"
                                    and history[-1].get("retryable_denial") is True)
                if history and not known_denial:
                    raise RuntimeError("unresolved dispatch for this cue; check private receipts before retry")
                if not self.allow_new:
                    raise RuntimeError("new cue request disabled for this replay")
                if os.environ.get("NAV_VLM_MODEL")!=MODEL or os.environ.get("NAV_VLM_API_URL","").rstrip("/")!="http://127.0.0.1:28938/v1":
                    raise RuntimeError("pinned APIYI profile missing")
                key = os.environ["NAV_VLM_API_KEY"]
                assert key
                log(self.calls, dict(base,event="dispatch",at=time.time()))
                conn = http.client.HTTPConnection("127.0.0.1",28938,timeout=480)
                status,usage = None,{}
                try:
                    conn.request("POST","/v1/chat/completions",body=wire,headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
                    resp=conn.getresponse()
                    status=resp.status
                    if status != 200:
                        raise RuntimeError("gateway response error")
                    value=json.loads(resp.read(1024*1024))
                    usage=value.get("usage") or {}
                    if value["model"] != MODEL:
                        raise ValueError("wrong model")
                    answer=json.loads(value["choices"][0]["message"]["content"])
                    if set(answer)!={"candidate"} or (answer["candidate"] is not None and answer["candidate"] not in offered):
                        raise ValueError("unknown candidate")
                    choice,kind=answer["candidate"],"success"
                except Exception as exc:
                    log(self.calls,dict(base,event="error",status=status,error_type=type(exc).__name__,total_tokens=usage.get("total_tokens"),cost=usage.get("cost"),
                                        retryable_denial=status in (402,429),at=time.time()))
                    raise
                finally:
                    conn.close()
                log(self.calls,dict(base,event="success",choice=choice,total_tokens=usage.get("total_tokens"),
                                    cost=usage.get("cost"),at=time.time()))
            if choice is not None and choice not in offered:
                raise RuntimeError("cached candidate not offered")
            return kind, choice

    def select(self, views, goal):
        if not isinstance(goal,str) or not goal.strip() or len(goal)>128 or len(views)!=3:
            raise ValueError("invalid search goal or views")
        with tempfile.TemporaryDirectory(dir=self.log_dir) as temp:
            paths = []
            for name, frame in zip(CAMERAS, views):
                rgb=np.asarray(frame)
                if rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<2 or rgb.dtype!=np.uint8:
                    raise ValueError("expected three uint8 RGB observations")
                path=Path(temp).resolve()/(name+".png")
                Image.fromarray(rgb).save(path)
                paths.append(path)
            detections=self._propose(paths,goal)
            items={}
            candidates={}
            for index,(v,path,pred) in enumerate(zip(CAMERAS,paths,detections)):
                sorted_boxes=sorted(pred,key=lambda b:tuple(b["xyxy"]))
                boxes={f"{PREFIXES[v]}{n}":b for n,b in enumerate(sorted_boxes,1)}
                items[v]=dict(path=path,boxes=boxes,source={"goal":goal})
                height,width=np.asarray(views[index]).shape[:2]
                limits=np.array((width,height,width,height))
                for key,box in boxes.items():
                    bb=np.array(box["xyxy"],dtype=float)
                    if bb.shape!=(4,) or not np.all(np.isfinite(bb)) or np.any((bb<0)|(bb>limits)) or np.any(bb[:2]>=bb[2:]):
                        raise ValueError("invalid candidate rectangle")
                    candidates[key]=dict(view_index=index,bbox_norm=(bb/limits).tolist(),candidate_id=key,category=goal)
            if candidates:
                request,offered=materials(items)
                wire=encode(request)
                req_hash=hashlib.sha256(wire).hexdigest()
                event,choice=self._choose(req_hash,wire,offered)
            else:
                req_hash,choice,event=None,None,"no_candidates"
            self.last_receipt=dict(event=event,model=MODEL,candidates=len(candidates),request_hash=req_hash,choice=choice)
            log(self.decisions,self.last_receipt)
            return candidates.get(choice)
