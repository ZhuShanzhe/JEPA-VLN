"""Fetch and verify only the authors' 50 smoke rows and their reachable RGB views."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import threading
import requests
from PIL import Image

REV = "076d076d8855c36ee7d925175e00bb89e7ac8466"
BASE = os.environ.get("HF_ENDPOINT","https://huggingface.co").rstrip("/") + "/datasets/Avalon-S/PInVerify/resolve/" + REV + "/"
INDEX = "val/pv_index_50.jsonl"
LOCAL = threading.local()


def fetch(root, rel):
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Invalid path in official manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        if not hasattr(LOCAL,"sessions"):
            LOCAL.sessions={"origin":requests.Session(),"mirror":requests.Session()}
            LOCAL.sessions["mirror"].trust_env=False
        origin="https://huggingface.co/datasets/Avalon-S/PInVerify/resolve/"+REV+"/"
        mirror="https://hf-mirror.com/datasets/Avalon-S/PInVerify/resolve/"+REV+"/"
        sites=[(BASE+rel,None),(origin+rel+"?download=true","origin"),(mirror+rel+"?download=true","mirror"),
               (origin+rel,"origin"),(mirror+rel+"?download=1","mirror"),(origin+rel+"?x=raw","origin")]
        for attempt,(url,endpoint) in enumerate(sites):
            try:
                endpoint=endpoint or ("mirror" if "hf-mirror" in url else "origin")
                response=LOCAL.sessions[endpoint].get(url,timeout=(7,18))
                response.raise_for_status()
                tmp.write_bytes(response.content)
                break
            except requests.RequestException:
                if attempt==len(sites)-1:
                    raise
                time.sleep(1+attempt)
        if path.suffix == ".png":
            with Image.open(tmp) as im:
                im.verify()
        elif path.suffix == ".json":
            json.loads(tmp.read_bytes())
        else:
            assert len([json.loads(x) for x in tmp.read_bytes().splitlines()]) == 50
        tmp.replace(path)
    return path


def prepare(root):
    root=root.resolve()
    index=fetch(root, INDEX)
    rows=[json.loads(x) for x in index.read_bytes().splitlines()]
    docs=sorted({r["meta_path"] for r in rows})
    views=set()
    errors=[]
    def attempt(rel):
        try:
            return fetch(root,rel)
        except (requests.RequestException,OSError,ValueError) as e:
            errors.append((rel,type(e).__name__))
            return None
    with ThreadPoolExecutor(max_workers=12) as pool:
        for path in pool.map(attempt, docs):
            if not path:
                continue
            doc=json.loads(path.read_bytes())
            for c in doc["captures"]:
                if c.get("navigable") and c.get("rgb"):
                    views.add(str(path.parent.relative_to(root) / c["rgb"]))
        list(pool.map(attempt, sorted(views)))
    if errors:
        raise RuntimeError(f"{len(errors)} files unavailable; safely resume the same manifest")
    signature=hashlib.sha256()
    for rel in sorted({INDEX,*docs,*views}):
        path=root/rel
        if path.suffix==".png":
            with Image.open(path) as im:
                im.verify()
        signature.update(rel.encode()+b"\0"+hashlib.sha256(path.read_bytes()).digest())
    receipt=dict(revision=REV,rows=len(rows),scenes=len({x["scene"] for x in rows}),
          source_episodes=len(docs),rgb_views=len(views),index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
          aggregate_source_sha256=signature.hexdigest())
    (root.parent/"pinverify_data_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("data",type=Path)
    prepare(parser.parse_args().data)
