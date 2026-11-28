"""Post-training diagnostics conditional on a reached endpoint, independent of the navigation metrics."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from view_model import ViewPredictor, current_inputs, sketch, SKETCH

HERE=Path(__file__).resolve().parent


def main():
    head=ViewPredictor.load(HERE/"private_model/jepa.pt")
    truths=[]; probabilities={}; pairs={}; pair_group={}; latent={}
    nrequests=mixed=0
    for file in sorted((HERE/"private_data").glob("blind_*.npz")):
        with np.load(file,allow_pickle=False) as d, np.load(HERE/"private_predictions"/file.name) as p:
            group,view,kind,reach,length,labels=[d[k] for k in ("group","view_id","kind","reach","action","visible")]
            length=(length>0).sum(1)
            ctx,pose,mp=[d[k] for k in ("context","pose","map")]
            out=head.predict(ctx,pose,mp)
            future,roi=d["future"],d["future_roi"]
            scores={k.replace("p_visible_",""):p[k] for k in p.files if k.startswith("p_visible_")}
            scores["box_geometry_only"]=1/(1+np.exp(-current_inputs(ctx,pose,mp)[1]))
            assert np.array_equal(p["group"],group) and np.array_equal(p["view_id"],view)
            for name in np.unique(group):
                rows=np.flatnonzero((group==name)&(kind=="alt")&reach.astype(bool))
                if not len(rows):continue
                ix=[]
                for v in np.unique(view[rows]):
                    same=rows[view[rows]==v]
                    ix.append(same[np.argmin(length[same])])
                ix=np.asarray(ix)
                y=labels[ix].astype(bool)
                truths.extend(y);nrequests+=1;mixed+=int(y.any() and not y.all())
                for arm,v in scores.items():
                    s=v[ix]
                    probabilities.setdefault(arm,[]).extend(s)
                    if y.any() and not y.all():
                        diff=s[y,None]-s[~y][None,:]
                        order=(diff>0)+.5*(diff==0)
                        pairs.setdefault(arm,[]).extend(order.ravel())
                        pair_group.setdefault(arm,[]).append(order.mean())
                actual=sketch(future[ix].astype(np.float32)-ctx[ix,:1024].astype(np.float32))
                roi_used=(np.linalg.norm(roi[ix].astype(np.float32),axis=1)>.1)
                crop=sketch(roi[ix].astype(np.float32)-ctx[ix,1024:].astype(np.float32))
                fitted=out["future_delta"][ix]
                for k,target,pred in (("full",actual,fitted[:,:SKETCH]),
                                      ("roi",crop[roi_used],fitted[roi_used,SKETCH:])):
                    latent.setdefault(k,[]).extend(np.column_stack((np.square(target-pred).mean(1),
                                                np.square(target).mean(1))))
    assert nrequests and mixed
    report=dict(test="Blind unseen houses. Actual endpoint reached, new views only, shortest reached route per physical viewpoint, all cases. This offline diagnostic gives no evaluator information to the actor.",
                labelled_endpoints=len(truths),clear=int(np.sum(truths)),source_groups=nrequests,
                groups_with_both_clear_and_unclear=mixed,arms={},future_feature_mse={})
    for arm,v in probabilities.items():
        report["arms"][arm]=dict(conditional_auroc=float(roc_auc_score(truths,v)),
                brier=float(np.mean(np.square(np.asarray(truths)-v))),
                within_request_pair_accuracy=float(np.mean(pairs[arm])),
                mean_per_request_pair_accuracy=float(np.mean(pair_group[arm])),
                paired_candidates=len(pairs[arm]))
    for part,errors in latent.items():
        err=np.asarray(errors)
        report["future_feature_mse"][part]=dict(n=len(err),predicted=float(err[:,0].mean()),copy_source=float(err[:,1].mean()))
    (HERE/"conditional_visibility.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))

if __name__=="__main__":
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
