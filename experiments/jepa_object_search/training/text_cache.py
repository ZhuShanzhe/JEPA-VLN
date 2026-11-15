"""Cache goal text only, using the locally installed BGE-M3 dense CLS encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import re
import tempfile

import numpy as np
import torch
from torch.nn import functional as F


def canonical_query(text):
    # Strip a WordNet synset suffix on category queries. Attribute phrases stay intact.
    match = re.fullmatch(r"([\w_]+)\.n\.\d+", text)
    return match.group(1).replace("_", " ") if match else text.strip()


class GoalEncoder:
    def __init__(self, model, device="cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.model = AutoModel.from_pretrained(model, local_files_only=True).eval().to(device)
        self.device = device
        if self.model.config.hidden_size != 1024:
            raise ValueError("BGE-M3 dense vectors must have dimension 1024")

    @torch.inference_mode()
    def encode(self, queries):
        batch = self.tokenizer([canonical_query(q) for q in queries], padding=True, truncation=True,
                               max_length=256, return_tensors="pt").to(self.device)
        dense = self.model(**batch).last_hidden_state[:,0,:].float()
        return F.normalize(dense, p=2, dim=-1).cpu().numpy()


def build(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    queries = sorted({json.loads(line)["query"] for line in (args.data/"index.jsonl").read_text().splitlines()})
    if not queries or not all(canonical_query(q) for q in queries):
        raise ValueError("Pack contains no usable goal text")
    torch.set_num_threads(args.threads)
    encoder = GoalEncoder(args.model, args.device)
    vec = np.concatenate([encoder.encode(queries[i:i+16]) for i in range(0,len(queries),16)])
    if not np.isfinite(vec).all() or not np.allclose(np.linalg.norm(vec,axis=-1),1.,atol=1.e-5):
        raise ValueError("Invalid goal embedding")
    protocol = dict(method="local BGE-M3 normalized final CLS, query-only", width=1024,
                    canonicalization="replace complete .n.NN suffixes with the noun, underscores with spaces",
                    model_config_sha256=hashlib.sha256((args.model/"config.json").read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        staged = Path(tmp)/"text.npz"
        np.savez(staged,queries=np.asarray(queries),vectors=vec,protocol=np.array(json.dumps(protocol)))
        # Publish only once, with no overwrite if another identical launch races.
        os.link(staged,args.output)
    print(json.dumps({"queries":len(queries), "width":vec.shape[1], **protocol}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    build(parser.parse_args())
