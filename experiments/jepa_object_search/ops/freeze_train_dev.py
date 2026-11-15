"""House-disjoint development split with no model outcomes in the assignment."""
from pathlib import Path
import json, hashlib

p=Path("/root/autodl-tmp/jepa_object_search")
scenes=json.loads((p/"simulation/scene_partitions.json").read_text())
external={f.name[:-8] for f in (p/"a_hssdov_external/eval/hssdov/content").glob("*.json.gz")}
public=set(scenes["official_hssd_val_excluded"]) | external
for v in scenes["langnav_public_scenes"].values():public.update(v)
eligible=sorted(set(scenes["eligible_hssd_training_scene_ids"])-public)
salt="jepa-nav-houses-20261114-v1"
order=sorted(eligible,key=lambda s:hashlib.sha256((salt+"/"+s).encode()).digest())
dev=sorted(order[:max(1,round(len(order)*0.2))]);train=sorted(set(order)-set(dev))
assert not(set(train)&set(dev) or (set(train)|set(dev))&public)
target=p/"simulation/train_dev_v1.json"
data=json.dumps(dict(seed=salt,train=train,dev=dev,test_excluded=sorted(public),
    purpose="research development only; neither the official LangNav splits nor HSSD-OV scenes are eligible"),indent=2)+"\n"
if target.exists():assert target.read_text()==data, "frozen split differs"
else:target.write_text(data)
print(len(train),len(dev),len(public),hashlib.sha256(data.encode()).hexdigest())
