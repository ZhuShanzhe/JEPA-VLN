"""Within-origin and equal-house physical opportunity metrics."""
import numpy as np


def per_origin(scores, group, y):
    y = y.numpy()
    p = scores.detach().numpy()
    eligible = group["mask"].numpy()
    num, queries = p.shape
    # Alternate origins cycle uniformly. The actual MOST VLM is absent in these houses.
    ref = np.arange(num)[:,None]
    qi = np.arange(queries)[None,:]
    options = np.where(eligible[:,:,None], p[None,:,:], -np.inf)
    alt = options.argmax(1)
    switched = np.max(options,axis=1) - p > .01
    chosen = np.where(switched,alt,ref)
    picked = y[chosen,qi]
    advantage = (picked-y).mean()
    dy = y[:,None]-y[None,:]
    dp = p[:,None]-p[None,:]
    valid = (dy>0)&eligible[:,:,None]
    order = (np.sign(dp[valid])+1)/2
    return dict(instances=float(picked.mean()), gain=float(advantage),
                origin_instances=float(y.mean()), override=float(switched.mean()),
                pair_credit=float(order.sum()), strict_pairs=int(valid.sum()),
                no_opportunity=float((y.max(0)==0).mean()),
                no_distinction=float((y.max(0)==y.min(0)).mean()))


def aggregate(raw):
    per = {}
    for arm, houses in raw.items():
        home = {}
        for hid, starts in sorted(houses.items()):
            by_start = []
            for sid, vals in starts.items():
                # Each horizon/query/mask shares a physical start with all of its replicas.
                by_start.append({k:float(np.mean([x[k] for x in vals])) for k in vals[0]})
            mean = {k:float(np.mean([x[k] for x in by_start])) for k in by_start[0]}
            mean["pair_accuracy"] = mean["pair_credit"]/mean["strict_pairs"] if mean["strict_pairs"] else None
            mean["informative_starts"] = sum(x["strict_pairs"]>0 for x in by_start)
            mean["starts"] = len(by_start)
            home[hid] = mean
        metrics = {k:float(np.mean([v[k] for v in home.values()])) for k in
                   ("instances","gain","origin_instances","override","no_opportunity","no_distinction",
                    "already_eligible_at_start_rate")}
        valid = [v["pair_accuracy"] for v in home.values() if v["pair_accuracy"] is not None]
        metrics["house_mean_pair_accuracy_where_informative"] = float(np.mean(valid)) if valid else None
        per[arm] = dict(mean=metrics, houses=home)
    return per
