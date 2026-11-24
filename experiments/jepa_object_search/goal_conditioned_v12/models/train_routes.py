"""Train, calibrate, lock, and score goal-conditioned heads on HSSD route candidates."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.nn import functional as F
from goal_ranker import GoalRanker, previous, load_prior, frozen_scores


SEEDS = (11, 29, 47)
STEPS = (200, 400, 600)
MIX = (0., .25, .5, .75, 1.)
ARMS = ('v12_direct','v12_jepa')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,obj):
    t = path.with_suffix('.partial')
    t.write_text(json.dumps(obj, indent=2, allow_nan=False))
    t.replace(path)


def load_rows(project, split, include_truth=False, masks=False):
    root = project/'jepa_route_v12'
    old = previous(project)
    cache = (root if split=='fresh' else project/'jepa_hidden_v11')/'models'/'cache'/'hssd'/split
    rows,keys = [],[]
    mem = old.remem(project)
    for path in sorted((cache/'actor').glob('*.npz')):
        with np.load(path) as a:
            h,orig,words = str(a['house']),str(a['origin']),a['words'].tolist()
            plans, cue = a['actions'],a['cue']
            x = old.route_inputs(a['ctx'].astype('float32'),plans)
            text = a['text'].astype('float32')
        y,first,local = None,None,None
        if include_truth:
            with np.load(cache/'truth'/path.name) as gt:
                y = gt['y']
        if masks:
            with np.load(root/'data'/'local_targets'/split/f'{orig}.npz') as targets:
                if targets['words'].tolist()!=words or not np.array_equal(targets['y'],y):
                    raise ValueError('teacher labels mismatch')
                first,local = targets['first_tap'],targets['local_z'].astype('float32')
                assert np.array_equal(first>=0,y>0) and local.shape[:2]==y.shape
        for q,w in enumerate(words):
            old.memory_start(mem,cue[:,q])
            row = dict(x=x,text=text[q],memory=old.bonus(mem,plans,2).astype('float32'))
            if y is not None:
                row['y'] = y[:,q].astype('float32')
            if first is not None:
                row.update(t=first[:,q].astype('int64'),z=local[:,q])
            rows.append(row)
            keys.append((h,orig,w))
    if not rows or len(set(keys)) != len(rows):
        raise ValueError(f'empty or duplicate {split} records')
    table = {k:torch.as_tensor(np.stack([r[k] for r in rows]),device='cuda') for k in rows[0]}
    return keys,table


def pairs(keys, y):
    sites = defaultdict(list)
    for i,(h,o,_) in enumerate(keys):
        sites[h,o].append(i)
    return [(a,b) for ix in sites.values() for a,b in itertools.combinations(ix,2)
            if np.any(y[a]>0) and np.any(y[b]>0) and not np.any((y[a]>0)&(y[b]>0))]


def measure(keys,y,indices,opposite):
    def macro(rows):
        origins,houses = defaultdict(list),defaultdict(list)
        for i,value in rows:
            h,o,_ = keys[i]
            origins[h,o].append(value)
        for (h,_),v in origins.items():
            houses[h].append(np.mean(v))
        return float(np.mean([np.mean(v) for v in houses.values()]))
    hit = macro([(i, float(y[i,j]>0)) for i,j in enumerate(indices)])
    swapped = []
    for a,b in opposite:
        i,j = indices[a],indices[b]
        margin = (float(y[a,i]>0)+float(y[b,j]>0)-float(y[a,j]>0)-float(y[b,i]>0))/2
        swapped.append((a,margin))
    return hit, macro(swapped)


def train_once(tr, keys, opposite, seed, arm, out):
    y = tr['y'].cpu().numpy()
    table = defaultdict(list)
    for i,(h,_,_) in enumerate(keys):
        table[h].append(i)
    houses = sorted(table)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net = GoalRanker(tr['text'].shape[1]).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0007,weight_decay=.025)
    for step in range(1,max(STEPS)+1):
        regular = [int(rng.choice(table[houses[j]])) for j in rng.integers(len(houses),size=32)]
        packed = [opposite[i] for i in rng.integers(len(opposite),size=8)]
        ix = regular + [a for a,b in packed] + [b for a,b in packed]
        row,hazard,latent = net(tr['x'][ix],tr['text'][ix])
        counts,first,target = tr['y'][ix],tr['t'][ix],tr['z'][ix]
        positive = first>=0
        rank = counts.unsqueeze(-1)>counts.unsqueeze(-2)
        preference = F.softplus(row.unsqueeze(-2)-row.unsqueeze(-1))
        base = F.binary_cross_entropy_with_logits(row,positive.float()) + preference[rank].mean()
        ticks = torch.arange(4,device='cuda')
        eligible = (ticks<=first.unsqueeze(-1)) | ~positive.unsqueeze(-1)
        event = ticks==first.unsqueeze(-1)
        temporal = F.binary_cross_entropy_with_logits(hazard[eligible], event[eligible].float(),
                                                       pos_weight=torch.tensor(2.,device='cuda'))
        # With the same starting image, a query change should reverse route preference.
        differences = row[32:40]-row[40:]
        ab = positive[32:40].unsqueeze(-1) & positive[40:].unsqueeze(-2)
        swap = F.softplus(.4 - differences.unsqueeze(-1) + differences.unsqueeze(-2))[ab].mean()
        loss = base + .25*temporal + .6*swap
        if arm == 'v12_jepa':
            sampled,routes = torch.where(positive)
            when = first[sampled,routes]
            local = target[sampled,routes]
            pred = latent[sampled,routes,when]
            correct = (pred*local).sum(-1)
            alternatives = latent[sampled,:,when,:]
            false_routes = ~positive[sampled]
            competing = (alternatives*local[:,None,:]).sum(-1)
            contrast = F.relu(.12 + competing - correct[:,None])[false_routes].mean()
            loss = loss + .45*(1-correct).mean() + .3*contrast
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
        optimizer.step()
        if step in STEPS:
            dest = out/f'{arm}_{seed}_{step}.pt'
            torch.save(net.state_dict(), dest)
    return net


@torch.inference_mode()
def score_batches(net, d):
    out = []
    for i in range(0,len(d['text']),128):
        out.append(net(d['x'][i:i+128],d['text'][i:i+128])[0].sigmoid().cpu().numpy())
    return np.concatenate(out)


def fit(project):
    out = project/'jepa_route_v12'/'models'
    lock = out/'selection.json'
    if lock.exists():
        raise FileExistsError(lock)
    weights = out/'weights'
    weights.mkdir(parents=True,exist_ok=True)
    old = previous(project)
    tk,tr = load_rows(project,'train',True,True)
    ck,cal = load_rows(project,'calibration',True)
    assert set(h for h,_,_ in tk).isdisjoint(h for h,_,_ in ck)
    tp = pairs(tk,tr['y'].cpu().numpy())
    cy = cal['y'].cpu().numpy()
    cp = pairs(ck,cy)
    report = dict(procedure='select on calibration only: .7 balanced house hit + .3 goal swap margin; common choice over seeds',
                  counts={'train':len(tk),'calibration':len(ck),'train_pairs':len(tp),'calibration_pairs':len(cp)},
                  seeds=list(SEEDS), steps=list(STEPS), blending=list(MIX),arms={}, source={f:digest(Path(__file__).with_name(f)) for f in ('train_routes.py','goal_ranker.py')})
    memory = cal['memory'].cpu().numpy()
    for arm in ARMS:
        grid = defaultdict(list)
        for seed in SEEDS:
            prior = score_batches(load_prior(old,project,seed,'jepa','cuda'),cal)
            net = train_once(tr,tk,tp,seed,arm,weights)
            for step in STEPS:
                net.load_state_dict(torch.load(weights/f'{arm}_{seed}_{step}.pt',weights_only=True))
                pred = score_batches(net.eval(),cal)
                for alpha in MIX:
                    picks = np.argmax(alpha*pred+(1-alpha)*prior+memory,axis=1)
                    grid[step,alpha].append(measure(ck,cy,picks,cp))
        summary = {f'{step}/{alpha}':np.mean(v,0).tolist() for (step,alpha),v in grid.items()}
        def quality(k):
            hit,contrast = np.mean(grid[k],0)
            return (.7*hit+.3*contrast,hit,-k[0],-k[1])
        step,alpha = max(grid,key=quality)
        hashes={}
        for seed in SEEDS:
            dst = weights/f'{arm}_{seed}.pt'
            shutil.copyfile(weights/f'{arm}_{seed}_{step}.pt',dst)
            hashes[str(seed)] = digest(dst)
        report['arms'][arm]=dict(step=step,head_weight=alpha,calibration=summary,checkpoint_sha256=hashes)
        print(arm,step,alpha,summary[f'{step}/{alpha}'],flush=True)
    write(lock,report)


@torch.inference_mode()
def evaluate(project,split):
    out = project/'jepa_route_v12'/'models'
    conf = json.loads((out/'selection.json').read_text())
    old = previous(project)
    keys,tbl = load_rows(project,split)
    excluded = set(json.loads((project/'jepa_hidden_v11'/'hssd'/'cohort.json').read_text())['train'])
    assert not any(k[0] in excluded for k in keys)
    target = out/f'{split}_choices.jsonl'
    with target.with_suffix('.partial').open('w') as fp:
        for seed in SEEDS:
            priors = {'v11_'+a:load_prior(old,project,seed,a,'cuda') for a in ('direct','jepa')}
            heads,mixing = {},{}
            for arm,settings in conf['arms'].items():
                path = out/'weights'/f'{arm}_{seed}.pt'
                assert digest(path)==settings['checkpoint_sha256'][str(seed)]
                m = GoalRanker(tbl['text'].shape[1]).cuda()
                m.load_state_dict(torch.load(path,weights_only=True))
                heads[arm] = m.eval()
                mixing[arm] = settings['head_weight']
            for i in range(0,len(keys),128):
                inputs = [tbl[k][i:i+128] for k in ('x','text','memory')]
                scores = frozen_scores(inputs,heads,priors,mixing)
                mem = inputs[2].cpu().tolist()
                for arm,tt in scores.items():
                    for j,s in enumerate(tt.cpu().numpy()):
                        h,o,w = keys[i+j]
                        line = dict(split=split,house=h,origin=o,query=w,seed=seed,arm=arm,score=s.tolist(),
                                    first=int(s.argmax()),initial_memory=mem[j])
                        fp.write(json.dumps(line,allow_nan=False)+'\n')
    target.with_suffix('.partial').replace(target)
    print(split,'saved decisions',len(keys)*len(SEEDS)*4)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('project',type=Path)
    ap.add_argument('operation',choices=['fit','calibration','test','fresh'])
    args=ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.operation=='fit':
        fit(args.project)
    else:
        evaluate(args.project,args.operation)
