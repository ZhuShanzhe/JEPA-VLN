"""One pinned multimodal high-level decision via the existing private project gateway."""
import base64
import fcntl
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image


MODEL = "gpt-4o-2024-11-20"
BASE = "http://127.0.0.1:28938/v1"
REGION_FIELDS = ("id", "distance_m", "bearing_deg", "turn_deg", "prior", "novelty", "unseen", "steps", "blocked")
PATCH_FIELDS = ("patch_id", "surface", "support", "bbox_norm")
MEMORY_FIELDS = {"remaining_steps", "step", "region_entered", "previous_contact", "checked_sites", "recent", "active_local"}
FORBIDDEN = re.compile(r"(secret|password|token|api_key|oracle|ground_truth|trajectory|scene_id|episode_id|instance_id|true_position|goal_views)",re.I)
SYSTEM = """You decide where an embodied camera should search next for MULTIPLE DISTINCT instances of the named target, using the same limited movement and turn budget throughout. Only current observations and online map candidates are known. Select one supplied reachable region ID. Consider what the target is, which scene areas plausibly contain it, new ground that could be covered, and past searches and blocked routes. Prefer a distinct promising unexplored part after checking one location. If a target is clear, choose a region that approaches that target or the next promising area if this location was already checked. A possible support surface by itself is usually too weak to justify a costly close inspection. Patches and semantic scores are imperfect hints; look at the images.

Use mode inspect ONLY if a current central image patch contains a specific cue whose ambiguity a closer or lateral view could resolve: a partly seen target, an occluded but specifically indicated target, or uncertainty whether a visually observed object is a new instance. Use mode explore for general searching, currently clear targets, and ordinary walls, tables, or furniture without specific evidence. The lower planner handles viewpoint selection and movement and can decline an unreachable inspection.

The RGB strip shows, left to right on the page: the camera turned RIGHT 30 degrees, the CENTRAL camera, the camera turned LEFT 30 degrees. Positive bearing is to the robot's left. Only patches on the central image have selectable IDs (0..8 in row-major order, top left to bottom right); their boxes are normalized [left,top,right,bottom]. Never invent an ID. Return only JSON with region_id (string), mode (explore or inspect), patch_id (an offered integer for inspect, null for explore), target_evidence (none, suspected, or clear, your visual judgment only), and cue (none, partial_target, occlusion, similar_instance, or target_visible)."""


class GateFailure(RuntimeError):
    pass


def clean(value):
    if isinstance(value,np.generic):
        value=value.item()
    if value is None or isinstance(value,(bool,str,int)):
        return value
    if isinstance(value,float) and math.isfinite(value):
        return round(value,4)
    if isinstance(value,(tuple,list)):
        return [clean(v) for v in value]
    if isinstance(value,dict) and all(isinstance(k,str) and not FORBIDDEN.search(k) for k in value):
        return {k:clean(v) for k,v in value.items()}
    raise ValueError("unsupported observation metadata")


def encoded(data):
    return json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def strip_png(views):
    if len(views)!=3:
        raise ValueError("expected right, center, left camera frames")
    imgs=[np.asarray(v) for v in views]
    if any(v.dtype!=np.uint8 or v.shape!=imgs[0].shape or v.ndim!=3 or v.shape[2]!=3 for v in imgs):
        raise ValueError("expected matched uint8 RGB cameras")
    buf=io.BytesIO()
    Image.fromarray(np.concatenate(imgs,axis=1)).save(buf,format="PNG",compress_level=2)
    return buf.getvalue()


def decision(value,ids,patch_ids):
    if not isinstance(value,dict) or not isinstance(value.get("region_id"),str) or value["region_id"] not in ids:
        raise GateFailure("unknown_region")
    mode=value.get("mode")
    p=value.get("patch_id")
    if mode not in ("explore","inspect") or "patch_id" not in value:
        raise GateFailure("bad_mode")
    if mode=="explore" and p is not None or mode=="inspect" and (type(p) is not int or p not in patch_ids):
        raise GateFailure("unknown_patch")
    ev=value.get("target_evidence")
    cue=value.get("cue")
    if ev not in ("none","suspected","clear") or cue not in ("none","partial_target","occlusion","similar_instance","target_visible"):
        raise GateFailure("bad_evidence_code")
    return dict(region_id=value["region_id"],mode=mode,patch_id=p,target_evidence=ev,cue=cue)


class GPTGate:
    def __init__(self,episode_id,log_dir):
        if os.environ.get("NAV_VLM_MODEL")!=MODEL or os.environ.get("NAV_VLM_API_URL","").rstrip("/")!=BASE:
            raise GateFailure("activate the existing APIyi pinned GPT-4o development profile")
        self.token=os.environ.get("NAV_VLM_API_KEY")
        if not self.token:
            raise GateFailure("missing private loopback token")
        folder=Path(log_dir)
        folder.mkdir(mode=0o700,parents=True,exist_ok=True)
        slug=hashlib.sha256(str(episode_id).encode()).hexdigest()[:16]
        self.path=folder/("gpt_"+slug+".jsonl")

    def choose(self,views,goal,regions,patches,memory):
        ids={r["id"] for r in regions}
        patch_ids={p["patch_id"] for p in patches}
        if not ids or len(ids)!=len(regions) or any(not isinstance(k,str) or len(k)>64 for k in ids):
            raise ValueError("invalid supplied region IDs")
        if len(patch_ids)!=len(patches) or any(type(p) is not int or not 0<=p<9 for p in patch_ids):
            raise ValueError("invalid supplied patch IDs")
        if not isinstance(goal,str) or not goal.strip() or len(goal)>512:
            raise ValueError("expected a short goal phrase")
        if any(k not in MEMORY_FIELDS for k in memory):
            raise ValueError("memory contains unrecognized fields")
        summary=clean(dict(goal=goal,region_columns=REGION_FIELDS,
                           regions=[[r.get(k) for k in REGION_FIELDS] for r in regions],
                           patch_columns=PATCH_FIELDS,patches=[[p.get(k) for k in PATCH_FIELDS] for p in patches],
                           memory=memory))
        visual=strip_png(views)
        text=encoded(summary).decode()
        req_hash=hashlib.sha256(encoded([MODEL,SYSTEM,text,hashlib.sha256(visual).hexdigest(),"png-high",0,160])).hexdigest()
        fd=os.open(self.path,os.O_CREAT|os.O_RDWR|os.O_APPEND,0o600)
        with os.fdopen(fd,"a+",encoding="utf8") as journal:
            fcntl.flock(journal.fileno(),fcntl.LOCK_EX)
            journal.seek(0)
            def event(kind,**facts):
                journal.write(json.dumps(dict(at=time.time(),event=kind,hash=req_hash,**facts),separators=(",",":"))+"\n")
                journal.flush()
                os.fsync(journal.fileno())
            history=[e for e in map(json.loads,journal) if e["hash"]==req_hash]
            prior=next((e for e in reversed(history) if e["event"]=="success"),None)
            if prior:
                result=decision(prior["decision"],ids,patch_ids)
                event("cache",decision=result)
                return result
            if history and history[-1]["event"] in {"dispatch","transport_uncertain"}:
                raise GateFailure("unresolved prior dispatch; check private receipts before retry")
            body=dict(model=MODEL,temperature=0,max_tokens=160,response_format={"type":"json_object"},
                      messages=[{"role":"system","content":SYSTEM},
                                {"role":"user","content":[{"type":"text","text":"Choose and return JSON. "+text},
                                 {"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(visual).decode(),"detail":"high"}}]}])
            event("dispatch",model=MODEL,step=memory.get("step"))
            conn=http.client.HTTPConnection("127.0.0.1",28938,timeout=480)
            try:
                conn.request("POST","/v1/chat/completions",body=encoded(body),headers={
                    "Authorization":"Bearer "+self.token,"Content-Type":"application/json"})
                resp=conn.getresponse()
                payload=json.loads(resp.read(1024*1024))
            except (OSError,ValueError,http.client.HTTPException):
                event("transport_uncertain")
                raise GateFailure("transport_uncertain") from None
            finally:
                conn.close()
            if resp.status!=200:
                err=payload.get("error",{})
                code=err.get("code","") if isinstance(err,dict) else ""
                code=code if isinstance(code,str) and re.fullmatch("[a-zA-Z0-9_]{1,48}",code) else "unspecified"
                event("gateway_http",status=resp.status,code=code)
                raise GateFailure("gateway_http_"+str(resp.status)+"_"+code)
            if payload.get("model")!=MODEL:
                event("wrong_model")
                raise GateFailure("unexpected returned model")
            usage=payload.get("usage") or {}
            facts=dict(model=MODEL,input_tokens=usage.get("prompt_tokens"),output_tokens=usage.get("completion_tokens"),
                       total_tokens=usage.get("total_tokens"),provider_reported_cost=usage.get("cost"))
            try:
                result=decision(json.loads(payload["choices"][0]["message"]["content"]),ids,patch_ids)
            except (GateFailure,IndexError,KeyError,ValueError,TypeError) as e:
                reason=str(e) if isinstance(e,GateFailure) else "invalid_json_object"
                event("invalid_decision",reason=reason,**facts)
                raise GateFailure(reason) from None
            event("success",decision=result,**facts)
            return result
