"""Independently check copied appearance, camera alignment and physical identities."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import habitat_sim as hs
import numpy as np

from build_frames import open_sim,read_rows


def main(root,assets,previous):
    sys.path.insert(0,str(previous/"collect"))
    import collect_views
    streams=defaultdict(lambda:defaultdict(dict))
    for r in read_rows(root/"frames.jsonl"):
        streams[r["layout"]][r["stratum"]][r["view"]]=r
    errors=[]; scale_errors=[]; observed_gaps=[]; aligned=0; label_counts=0
    families=defaultdict(set)
    for layout,conditions in streams.items():
        single,double=conditions["single"],conditions["double"]
        for v in range(3):
            assert np.array_equal(single[v]["K"],double[v]["K"]) and np.array_equal(single[v]["T_world_from_camera"],double[v]["T_world_from_camera"])
            aligned+=1
        parent,=single[0]["boxes"]
        child,=[q for q in double[0]["boxes"] if ":synthetic:" in q["instance"]]
        assert parent["asset"]==child["asset"]
        a,b=parent["xyxy"],child["xyxy"]
        assert min(a[2],b[2])-max(a[0],b[0])<=1 or min(a[3],b[3])-max(a[1],b[1])<=1
        diff=np.array(child["world_center"])-parent["world_center"]
        observed_gaps.append(float(np.linalg.norm(diff)))
        families[single[0]["category"]].add(parent["asset"])
        original_id=int(parent["instance"].split(":")[-1]); house=single[0]["house"]
        sim=open_sim(assets,house)
        try:
            collect_views.inventory(sim,assets)
            manager=sim.get_rigid_object_manager()
            obj,=[o for o in manager.get_objects_by_handle_substring("").values() if o.semantic_id==original_id]
            duplicate=manager.duplicate_object_by_id(obj.object_id)
            duplicate.motion_type=hs.physics.MotionType.KINEMATIC
            errors.append(float(np.max(abs(np.array(obj.transformation)[:3,:3]-np.array(duplicate.transformation)[:3,:3]))))
            scale_errors.append(float(np.max(abs(np.array(obj.scale)-np.array(duplicate.scale)))))
        finally:
            sim.close()
        label_counts+=1
    out=dict(camera_pairs_exact=aligned,layouts_checked=label_counts,max_rotation_matrix_difference=max(errors),
             max_scale_difference=max(scale_errors),actual_center_distance_m=[min(observed_gaps),max(observed_gaps)],
             unique_asset_templates={k:len(v) for k,v in families.items()})
    (root/"controls_geometry_audit.json").write_text(json.dumps(out,indent=2))
    print(out)


if __name__=="__main__":
    main(*map(Path,sys.argv[1:]))
