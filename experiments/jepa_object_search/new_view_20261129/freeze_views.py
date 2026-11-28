"""Select unused families from installed HSSD official training split using prior cohort manifests and run paths."""
import json, re, hashlib, os
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
DISK=PROJECT.parent
ASSETS=DISK/"jepa_vln_data_20261014/raw/hssd-hab"
PAT=re.compile(r"(?<!\d)(10\d{7})(?:_\d+)?(?!\d)")

def candidates():
    all_ids=[]
    for line in (ASSETS/"scene_splits.yaml").read_text().splitlines():
        if line.endswith(":"):part=line[:-1].strip()
        elif line.startswith("- ") and part=="train":
            h=line[2:].strip().strip("'\"")
            if (ASSETS/"scenes"/(h+".scene_instance.json")).is_file():all_ids.append(h)
    return set(all_ids)

def main():
    prior=json.loads((PROJECT/"path_feedback_20261128/natural_design.json").read_text())["cohorts"]
    families=lambda text:set(PAT.findall(text))
    used=families(str(prior)); manifests=set();run_families=set()
    pool=candidates()
    for base in [PROJECT,DISK/"jepa_vln_paper_bench_20261017",DISK/"jepa_instance_bench_20261016",DISK/"jepa_vln_baselines_20261015"]:
        for root,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if d not in (HERE.name,".git","__pycache__","data","datasets","models","models--openai","private_sources","cache","site-packages","bin","lib","owned_publication_20261128","raw","assets","policy_env","wheels","node_modules","conda-meta","hssd-hab") and not d.startswith(("hf-","models--"))]
            path=Path(root)
            for name in files:
                f=path/name
                if f.suffix.lower() in (".npz",".json",".jsonl",".log",".jpg",".png",".pth",".csv",".pt"):
                    run_families.update(families(str(f.relative_to(base))))
                if f.suffix in (".json",".jsonl") and any(k in name.lower() for k in ("cohort","panel","manifest","split","design","summary","result","episode","metric")) and f.stat().st_size<3500000:
                    if name in ("dataset_manifest.json","scene_splits.json","hssd-hab.scene_dataset_config.json"):continue
                    text=f.read_text(errors="ignore")
                    for k in families(text):
                        if any(v.split('_')[0]==k for v in pool):manifests.add(k)
    used|=manifests|run_families
    remain=sorted((h for h in pool if h.split('_')[0] not in used),key=lambda h:hashlib.sha256(("new-view-20261129:"+h).encode()).digest())
    chosen=[]
    for h in remain:
        if h.split('_')[0] in [x.split('_')[0] for x in chosen]:continue
        chosen.append(h)
        if len(chosen)==6:break
    print("installed_train",len(pool),"candidate_unseen",len(remain),"selected",chosen,flush=True)
    assert len(chosen)>=3
    report=dict(cohorts=dict(train=prior["train"],val=prior["val"],blind=chosen),blind_basis="Installed official train houses excluding family IDs identified in prior project cohorts, results, panels, experiment shard/trace names, benchmark output trees and previous step's train/val/test/blind. Downloaded asset files excluded. Sorted by salted hash before collection.",known_used_from_named_manifests=sorted((manifests|families(str(prior)))&{h.split('_')[0] for h in pool}),source_count_per_house=24,control_forwards=[4,8,12],scan_actions=[2,3],max_actions="control forward steps + 16 + 2 scan actions")
    f=HERE/"view_design.json"
    if f.exists():assert json.loads(f.read_text())==report
    else:f.write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":main()
