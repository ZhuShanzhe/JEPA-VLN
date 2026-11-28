"""Post-hoc inspection of images/poses selected on blind houses. Aggregate pixel-size and framing evidence."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    records=json.loads((ROOT/"private_evaluation/blind_choices.json").read_text())
    buckets=defaultdict(Counter)
    examples={}
    for house in sorted({r["house"] for r in records}):
        with np.load(ROOT/"private_render_blind60"/("blind_"+house)/"controls.npz") as z:
            d={k:z[k] for k in ("group","kind","pose","start_pixels","scan_pixels","future_box","pixels","category","reach","budget")}
        for r in (r for r in records if r["house"]==house):
            rows=np.flatnonzero(d["group"]==r["group"])
            i0=rows[0]; alt=rows[(d["kind"][rows]=="alt") & d["reach"][rows].astype(bool)]
            best=int(d["pixels"][alt].max()) if len(alt) else -1
            j=r["jepa"]
            if j is None:result="deferred"
            elif not d["reach"][j]:result="blocked"
            elif d["pixels"][j]>=369:result="clear"
            else:result="reached_unclear"
            stages=["all"]
            if r["trigger"]:stages+=["triggered"]
            if r["trigger"] and max(d["start_pixels"][i0],d["scan_pixels"][i0])<369:stages+=["triggered_originally_unclear"]
            for stage in stages:
                b=buckets[stage]
                if result!="reached_unclear":continue
                px=int(d["pixels"][j])
                b["n"]+=1
                b["none_visible" if not px else "visible_1_to_99" if px<100 else "visible_100_to_249" if px<250 else "visible_250_to_368"]+=1
                box=d["future_box"][j]
                if px and (min(box[:2])<=1 or max(box[2:])>=191):b["touches_image_edge"]+=1
                if best>=369:b["useful_alternative_existed"]+=1
            if r["trigger"] and r["reason"] not in examples:
                example=dict(class_name=str(d["category"][i0]),start_pixels=int(d["start_pixels"][i0]),initial_scan_pixels=int(d["scan_pixels"][i0]),
                             action_limit=int(d["budget"][i0]),best_reachable_alternative_pixels=best,
                             selected_pixels=int(d["pixels"][j]) if j is not None and d["reach"][j] else None)
                if j is not None:example["selected_horizontal_range_m"]=round(float(np.linalg.norm(d["pose"][j,:2])),2)
                examples[r["reason"]]=example
    report=dict(reached_unclear_pixels=buckets,illustrative_first_by_category=examples,
                scope="Offline same-instance pixel evidence. The first example in each diagnostic class is selected in fixed house/request order; counts, not these examples, describe frequencies. Pixel counts and edge contact alone cannot distinguish depth occlusion from unfavorable viewing scale.")
    (ROOT/"visual_failure_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)


if __name__=="__main__":main()
