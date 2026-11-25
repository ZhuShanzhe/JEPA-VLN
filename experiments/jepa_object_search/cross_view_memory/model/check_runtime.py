"""Verify the frozen crop encoder, predictor and backward pass on rendered RGBs."""
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from reid_model import Encoder, ViewHead
from train import Crops


if __name__=="__main__":
    project=Path(sys.argv[1])
    root=project/"jepa_reid_v20/smoke"
    source=next((root/"houses").glob("*.json"))
    rows=json.loads(source.read_text())[:3]
    print("columns",sorted(rows[0]), flush=True)
    crops,contexts=next(iter(DataLoader(Crops(root,rows),batch_size=3)))
    enc=Encoder(project).cuda().eval()
    embedding=enc(crops.cuda(),contexts.cuda()).clone()
    head=ViewHead().cuda()
    logits,cos=head(embedding[:2],embedding[1:],torch.tensor([35.,-70.],device="cuda"))
    logits.sum().backward()
    print("embedding",list(embedding.shape),"unit norm",embedding.norm(dim=1).tolist(),"scores",logits.sigmoid().detach().cpu().tolist(),flush=True)
