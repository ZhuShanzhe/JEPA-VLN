"""Train a small RGB-only patch readout on eight houses; assess on four other training houses."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def metrics(y, pred):
    confusion=np.zeros((11,11),dtype=int)
    np.add.at(confusion,(y,pred),1)
    n=confusion.sum(1)
    recall=np.divide(confusion.diagonal(), n, out=np.zeros(11),where=n>0)
    return dict(correct=int(np.trace(confusion)),patches=len(y),class_count=n.tolist(),
                accuracy=float(np.trace(confusion)/len(y)),
                balanced_accuracy=float(recall[n>=10].mean()),macro_classes_min_ten=np.flatnonzero(n>=10).tolist(),
                per_class_recall=recall.tolist(),confusion_rows_true=confusion.tolist())


def main(project):
    sys.path[:0]=[str(project/"jepa_env_v13/models"),str(project/"prior_approach_20261128")]
    from env_routes import EnvironmentRoutes
    from placement_prior import PlacementPrior

    here=Path(__file__).resolve().parent
    order=json.loads((project/"jepa_env_v13/simulator/training_cohort.json").read_text())["train"]
    assert len(order)==12 and len(set(order))==12
    model=EnvironmentRoutes(project)
    prior=PlacementPrior.load(project/"prior_approach_20261128/learned_prior.json")
    prompts=prior.visual_prompts(model)
    text=model.encoder.words(["chair"])
    X,Y,H,base=[],[],[],[]
    for i,house in enumerate(order):
        path=here/"training_private"/(house+".npz")
        with np.load(path,allow_pickle=False) as data:
            imgs,y=data["crops"],data["labels"]
        feat=[]
        for j in range(0,len(imgs),72):
            regions,_=model.encoder.encode(imgs[j:j+72],text)
            feat.append(regions.astype(np.float32))
            model.encoder.cache.clear()
        feat=np.concatenate(feat,axis=0)
        c=np.minimum((feat[:,0]@prompts.T).argmax(1),10)
        X.append(feat.reshape(len(feat),-1));Y.append(y);H.append(np.full(len(y),i));base.append(c)
    X=np.concatenate(X);Y=np.concatenate(Y).astype(int);H=np.concatenate(H);base=np.concatenate(base)
    train=H<8
    mean=X[train].mean(0);std=X[train].std(0).clip(.015);X=(X-mean)/std
    torch.manual_seed(321);torch.set_num_threads(2)
    x=torch.tensor(X[train],dtype=torch.float32);y=torch.tensor(Y[train],dtype=torch.int64)
    count=np.bincount(Y[train],minlength=11)
    balance=torch.tensor(1./np.sqrt(np.maximum(count,1)),dtype=torch.float32)
    net=torch.nn.Linear(x.shape[1],11)
    opt=torch.optim.AdamW(net.parameters(),lr=.012,weight_decay=.08)
    for _ in range(100):
        opt.zero_grad()
        loss=F.cross_entropy(net(x),y,weight=balance)+.15*net.weight.square().sum()/11.
        loss.backward();opt.step()
    pred=net(torch.tensor(X,dtype=torch.float32)).argmax(1).numpy()
    picked=metrics(Y[~train],pred[~train]);old=metrics(Y[~train],base[~train])
    print("supervised versus zero-shot", picked["balanced_accuracy"],old["balanced_accuracy"],picked["accuracy"],old["accuracy"])
    report=dict(schema=1,train_houses=order[:8],check_houses=order[8:],train_samples=int(train.sum()),
                classes=["wall","table","desk","counter","shelf","cabinet","sofa","bed","sink","floor","other"],
                baseline=old,trained=picked,training_count=count.tolist(),by_house={
                h:{"baseline":metrics(Y[H==i],base[H==i]),"trained":metrics(Y[H==i],pred[H==i])}
                for i,h in enumerate(order) if i>=8})
    out=here/"surface_reader.npz"
    if picked["balanced_accuracy"]>old["balanced_accuracy"] and picked["accuracy"]>old["accuracy"]:
        np.savez_compressed(out,weight=net.weight.detach().numpy(),bias=net.bias.detach().numpy(),mean=mean,std=std)
        report["selected_sha256"]=hashlib.sha256(out.read_bytes()).hexdigest()
    (here/"surface_validation.json").write_text(json.dumps(report,indent=2))
    if "selected_sha256" not in report:
        raise ValueError("new reader fails prespecified two-metric comparison")


if __name__=="__main__":
    main(Path(sys.argv[1]))
