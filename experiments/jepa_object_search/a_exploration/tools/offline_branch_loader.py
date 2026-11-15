"""Separate context-only branches from teacher observations used as predictive targets."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'paired_branches_v2'

def inputs(sample):
    sample=Path(sample)
    with np.load(sample/'context.npz') as x:
        return dict(history_rgb=x['rgb'],history_depth=x['depth_m'],
                    candidates=x['frontiers_xy'],proposed_actions=x['planned_actions'])

def supervised_future(sample, branch):
    sample=Path(sample)
    with np.load(sample/f'branch{branch:03d}.npz') as b:
        return dict(future_rgb=b['rgb'],future_depth=b['depth_m'],executed_actions=b['actions'])

def cached_features(sample):
    """Mean pooled V-JEPA embeddings returned separately for context and training targets."""
    with np.load(Path(sample)/'features_vjepa21_b384.npz') as z:
        return z['context_embedding'],z['future_targets']

def records(split='train'):
    assert split in ('train','dev')
    with (ROOT/'index.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['split']==split: yield r,ROOT/r['id']
