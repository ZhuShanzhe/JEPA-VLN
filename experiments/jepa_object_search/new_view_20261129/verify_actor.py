"""Run the live RGB-D interface and compare its actions with the independent paired benchmark."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from observation_actor import ObservationActor
from long_data import FrozenImageEncoder

HERE=Path(__file__).resolve().parent


def main():
    actor=ObservationActor.load("cuda" if torch.cuda.is_available() else "cpu")
    encoder=FrozenImageEncoder(HERE.parent)
    choices=json.loads((HERE/"private_evaluation/blind_choices.json").read_text())
    blind=json.loads((HERE/"view_design.json").read_text())["cohorts"]["blind"]
    assert {r["house"] for r in choices}==set(blind)
    counts=Counter()
    for house in sorted({r["house"] for r in choices}):
        with np.load(HERE/"private_render_blind60"/("blind_"+house)/"controls.npz") as f:
            data={k:f[k] for k in ("source_index","action","kind","budget","reach","pixels","first_blocked","scan_pixels","start_pixels")}
        for record in (r for r in choices if r["house"]==house):
            source=record["source"]
            rows=np.flatnonzero(data["source_index"]==source)
            budget=int(data["budget"][rows[0]])
            n=budget-18
            with np.load(HERE/"private_live_inputs"/house/f"{source}.npz") as f:
                result=actor.from_images(encoder, f["rgb"], f["box"], f["depth"], f["left_depth"], n, budget-2)
            j=record["jepa"];base=record["old"]
            selected=None
            if result["status"]=="execute":
                action=result["actions"]
                assert len(action)+2<=budget
                matches=[r for r in rows if np.array_equal(data["action"][r][data["action"][r]>0],action)]
                if not matches:raise ValueError("unlabelled motion emitted")
                selected=matches[0]
                counts["issued"]+=1
                counts["reached"]+=int(data["reach"][selected])
                counts["clear"]+=int(data["pixels"][selected]>=369)
                counts["contact_or_stall"]+=int(data["first_blocked"][selected]>0)
                counts["actions"]+=int(data["first_blocked"][selected] or len(action))
            else:
                counts["deferred"]+=1
            same=(j is None and selected is None) or (selected is not None and j is not None and np.array_equal(data["action"][selected],data["action"][j]))
            counts["action_sequence_agrees"]+=int(same)
            counts["groups"]+=1
            counts["actions"]+=2
            old_clear=base is not None and data["pixels"][base]>=369
            new_clear=selected is not None and data["pixels"][selected]>=369
            unseen=max(data["start_pixels"][rows[0]],data["scan_pixels"][rows[0]])<369
            counts["rescued"]+=int(new_clear and not old_clear)
            counts["rescued_both_initially_unclear"]+=int(new_clear and not old_clear and unseen)
            counts["regressed"]+=int(old_clear and not new_clear)
    receipt=dict(scope="Benchmark action choices are hidden until after each live RGB-D call. The image/box crop are encoded independently from re-rendered original RGB. Only current detector box, two depth observations and finite remaining budget are provided.",counts=counts)
    (HERE/"live_actor_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(dict(counts))


if __name__=="__main__":
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
