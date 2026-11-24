"""Goal-conditioned route selection from three observed views and candidate actions."""
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def previous(project):
    models = Path(project) / 'jepa_hidden_v11' / 'models'
    if str(models) not in sys.path:
        sys.path.insert(0, str(models))
    import train_compare
    return train_compare


class GoalRanker(nn.Module):
    """Condition predictions on the requested object before pooling route evidence."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.visual = nn.Sequential(nn.Linear(dim * 3 + 8, 128), nn.LayerNorm(128), nn.GELU())
        self.goal = nn.Linear(dim, 128)
        self.local = nn.Sequential(nn.Linear(128 * 2 + dim, 128), nn.GELU(), nn.Linear(128, dim))
        self.recognize = nn.Sequential(nn.Linear(128 * 2 + 5, 80), nn.LayerNorm(80),
                                       nn.GELU(), nn.Linear(80, 1))

    def forward(self, x, text):
        vision = self.visual(x)
        goal = self.goal(text)[:, None, None, :]
        joint = vision * goal
        words = text[:, None, None, :]
        departure = F.normalize(x[..., :self.dim], dim=-1)
        cues = x[..., :3*self.dim].reshape(*x.shape[:-1], 3, self.dim)
        cue_scores = (F.normalize(cues, dim=-1) * words.unsqueeze(-2)).sum(-1)
        local = F.normalize(self.local(torch.cat((vision, joint, departure * words), dim=-1)), dim=-1)
        alignment = (local * words).sum(-1, keepdim=True)
        time = x[..., -3:-2]
        hazard = self.recognize(torch.cat((vision, joint, 8*cue_scores, 8*alignment, time), dim=-1)).squeeze(-1)
        route = torch.logsumexp(hazard, dim=-1) - np.log(hazard.shape[-1])
        return route, hazard, local


def load_prior(old, project, seed, name, device):
    import json
    d = project/'jepa_hidden_v11'/'models'
    manifest = json.loads((d/'fit_receipt.json').read_text())['hssd']['runs']
    entry = next(x for x in manifest if x['arm']==name and x['seed']==seed)
    path = d/'weights'/'hssd'/f'{name}_{seed}.pt'
    if old.sha(path) != entry['checkpoint_sha256']:
        raise ValueError('prior weights differ from frozen V11')
    weights = torch.load(path, map_location=device, weights_only=True)
    m = old.Ranker(weights['goal.weight'].shape[1]).to(device)
    m.load_state_dict(weights)
    return m.eval()


def frozen_scores(entries, models, v11, mixing):
    """Post-encoder API. Only observed RGB features, target text, actions and memory enter."""
    x, text, memory = entries
    prior = {arm:head(x,text)[0].sigmoid() for arm,head in v11.items()}
    result = {arm:scores+memory for arm,scores in prior.items()}
    for arm,head in models.items():
        alpha = mixing[arm]
        own = head(x,text)[0].sigmoid()
        result[arm] = (1-alpha)*prior['v11_jepa'] + alpha*own + memory
    return result
