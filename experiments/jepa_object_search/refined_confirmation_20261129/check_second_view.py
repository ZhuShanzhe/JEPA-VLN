"""Test physical re-acquisition after background padding on calibration houses, after policy freeze."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import numpy as np

from memory_refit import RefinedMemory
from train_memory import variants
from memory_diagnostics import classify

HERE=Path(__file__).resolve().parent


def check(path):
    with np.load(path) as f:
        rgb, depth, poses, masks, cue=(f[k] for k in ("rgb","depth","pose","mask","box"))
    masks=np.unpackbits(masks,axis=1).reshape(-1,192,192)
    stats=Counter()
    for box in list(variants(cue,int(path.stem)%2))[1:]:
        model=RefinedMemory(rgb[0],depth[0],box)
        first=model.observe(rgb[0],depth[0],poses[0])
        stats["padded_requests"]+=1
        if first["memory_state"]!="pending":
            continue
        stats["initially_delayed"]+=1
        assert not first["accept"]
        promoted=False
        physically_possible=False
        for i,pose in enumerate(poses[1:],start=1):
            translated=np.hypot(*pose[:2])>=.12
            if translated and masks[i].sum()>=369:
                physically_possible=True
            result=model.observe(rgb[i],depth[i],pose)
            if not promoted and result["memory_state"]=="reanchored":
                assert translated
                promoted=True
                stats["promoted_on_displaced_view"]+=1
                # The anchor is the newly registered frame's native reference rectangle.
                anchor_box=model.verifier.sourcebox
                rect=anchor_box
                l,t,r,b=rect
                cover=masks[i][t:b,l:r].sum()/max(1,masks[i].sum())
                stats["promoted_anchor_covers_at_least_half_visible_instance"]+=int(masks[i].any() and cover>=.5)
            if result["accept"]:
                assert promoted
                stats["correct_after_promotion" if classify(masks[i],result["box"])=="correct" else "incorrect_after_promotion"]+=1
                break
        stats["delayed_with_clear_and_displaced_view_before_stop"]+=int(physically_possible)
    return stats


if __name__=="__main__":
    os.umask(0o077)
    files=sorted((HERE/"private_memory_validation").glob("*/*.npz"))
    assert len(files)==144 and len({p.parent.name for p in files})==6
    with ProcessPoolExecutor(max_workers=6) as pool:
        total=sum(pool.map(check,files),Counter())
    assert total["padded_requests"]==288
    result={"scope":"Functional diagnostic on six calibration houses, not a test-set result. Two padded cue variants per physical source; only initially delayed instances are followed through their continuous 22-action observations.",
            "counts":dict(total)}
    (HERE/"second_view_check.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)
