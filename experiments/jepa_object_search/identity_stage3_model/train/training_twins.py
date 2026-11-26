"""Collect same-room lookalike/replacement controls in train and calibration houses."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import sys

import magnum as mn
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(sys.argv[1])/"jepa_reid_v21/data"))
from build_frames import CLOCKS,K,annotations,open_sim,read_rows,stable


def poses(sim,row):
    eye=np.asarray(row["camera_xyz"])
    old=(eye-[0,1.25,0]).astype(np.float32)
    heading=row["camera_heading_deg"]
    vec=np.array([-math.sin(math.radians(heading)),0.,math.cos(math.radians(heading))])
    for offset,turn in ((0.,0.),(-.24,-11.),(.24,11.)):
        ground=np.asarray(sim.pathfinder.try_step(old,(old+vec*offset).astype(np.float32))) if offset else old
        if not np.isfinite(ground).all() or np.linalg.norm(ground-old)>.38:
            ground=old
        yield dict(row,camera_xyz=(ground+[0,1.25,0]).tolist(),camera_heading_deg=heading+turn)


def render(sim,items,inventory,root,sequence,condition,layout):
    from replay_depth import observe
    from habitat_sim.utils.common import quat_rotate_vector
    frames=[]
    for i,row in enumerate(items):
        obs,state=observe(sim,row)
        cam=np.eye(4)
        cam[:3,:3]=np.column_stack([quat_rotate_vector(state.rotation,v) for v in np.eye(3)])@np.diag([1.,-1.,-1.])
        cam[:3,3]=state.position
        boxes=annotations(obs["ids"],inventory)
        frame_id=sequence+"_"+str(i)
        rgb=Path("images")/(frame_id+".jpg");depth=rgb.with_suffix(".npy")
        Image.fromarray(obs["rgb"][:,:,:3]).save(root/rgb,quality=93)
        np.save(root/depth,obs["depth"].astype(np.float32))
        frames.append(dict(frame_id=frame_id,split=row["split"],house=row["house"],category=row["category"],
                           view=i,sequence=sequence,layout=layout,stratum=condition,rgb=rgb.as_posix(),
                           depth=depth.as_posix(),camera_heading_deg=row["camera_heading_deg"],K=K,
                           T_world_from_camera=cam.tolist(),boxes=boxes))
    return frames


def main(project,assets,root,split,clock_only=False):
    if split not in ("train","val"): raise ValueError("development houses only")
    previous=project/"jepa_reid_v20"
    import habitat_sim as hs
    from habitat_sim.utils.common import quat_rotate_vector
    sys.path[:0]=[str(previous/"model"),str(previous/"collect")]
    import collect_views
    from replay_depth import observe
    goal={"clock":"wall_clock","wall_art":"wall_art"}
    collect_views.WORDS.update(CLOCKS)
    candidate=defaultdict(lambda:defaultdict(list))
    manifests=[root.parent/(split+"_clock_seeds.jsonl")] if clock_only else [previous/name/"manifest.jsonl" for name in ("data","clocks")]
    for manifest in manifests:
        for r in read_rows(manifest):
            if r["split"]==split and r["category"] in goal:
                candidate[r["category"]][r["house"]].append(r)
    (root/"images").mkdir(parents=True,exist_ok=True)
    rows=[]; distances=[]; rejected=Counter(); diagnostics=Counter()
    for cat,houses in candidate.items():
        for house in sorted(houses,key=lambda h:stable("train-twins-stage3",h))[:5]:
            sim=open_sim(assets,house)
            try:
                objects=sim.get_rigid_object_manager()
                registry=collect_views.inventory(sim,assets)
                group=defaultdict(dict)
                for subtype,obs in registry.items():
                    for sid,asset,center in obs:
                        group["clock" if subtype in CLOCKS else subtype][sid]=dict(instance=f"{house}:{sid}",
                                                                                 subtype=subtype,asset=asset,world_center=center.tolist())
                entries=group[cat]
                by_sid={o.semantic_id:o for o in objects.get_objects_by_handle_substring("").values()}
                success=False
                for row in sorted(houses[house],key=lambda r:stable(r["image"],"control")):
                    sid=int(row["instance"].split(":")[-1])
                    if entries[sid]["subtype"]!=goal[cat]: continue
                    diagnostics["eligible_subtype"]+=1
                    obj=by_sid[sid]
                    anchor=annotations(observe(sim,row)[0]["ids"],entries)
                    if len(anchor)!=1 or anchor[0]["instance"]!=row["instance"]:continue
                    diagnostics["single_anchor"]+=1
                    bbox=anchor[0]["xyxy"]
                    if min(bbox[2]-bbox[0],bbox[3]-bbox[1])<20:continue
                    diagnostics["large_enough"]+=1
                    # Pick the broad locally horizontal object axis. Translate along its
                    # horizontal projection so that an installed wall item stays at its height.
                    dims=np.array(obj.aabb.max)-np.array(obj.aabb.min)
                    candidates=[]
                    for axis,length in zip(np.eye(3),dims):
                        vec=np.array(obj.rotation.transform_vector(mn.Vector3(*axis)))
                        if abs(vec[1])>.35:continue
                        vec[1]=0;vec/=np.linalg.norm(vec)
                        candidates.append((length,vec))
                    if not candidates:continue
                    width,axis=max(candidates,key=lambda l:l[0])
                    gap=float(width+.12)
                    if not .22<=gap<=1.4:continue
                    diagnostics["gap_pass"]+=1
                    layout="ctrl_"+cat+"_"+house
                    cam=list(poses(sim,row))
                    for sign in (-1,1):
                        twin=objects.duplicate_object_by_id(obj.object_id)
                        twin.motion_type=hs.physics.MotionType.KINEMATIC
                        # duplicate_object_by_id duplicates all local transforms and scaling.
                        twin.translation=obj.translation+mn.Vector3(*(sign*gap*axis))
                        gift=max(by_sid)+1
                        twin.semantic_id=gift
                        center=np.array(twin.transformation.transform_point(twin.aabb.center()))
                        augmented=dict(entries)
                        augmented[gift]=dict(**{k:v for k,v in entries[sid].items() if k not in ("instance","world_center")},
                                             instance=f"{house}:synthetic:{sid}",world_center=center.tolist())
                        pairs=annotations(observe(sim,row)[0]["ids"],augmented)
                        diagnostics["pair_count_"+str(len(pairs))]+=1
                        if diagnostics["gap_pass"]<=2:
                            print("clone",sid,gift,gap,obj.translation,twin.translation,[(p["instance"],p["xyxy"]) for p in pairs],flush=True)
                        good=len(pairs)==2 and {p["instance"] for p in pairs}=={q["instance"] for q in (entries[sid],augmented[gift])}
                        if good:
                            a,b=[q["xyxy"] for q in pairs]
                            good=(min(a[2],b[2])-max(a[0],b[0])<=1 or min(a[3],b[3])-max(a[1],b[1])<=1)
                        if good:
                            doubles=render(sim,cam,augmented,root,layout+"_double","double",layout)
                        objects.remove_object_by_id(twin.object_id)
                        if good:
                            singles=render(sim,cam,entries,root,layout+"_single","single",layout)
                            rows.extend(singles+doubles)
                            distances.append(gap);success=True;break
                    if success:break
                if not success:rejected[cat]+=1
            finally:
                sim.close()
    if not rows:
        raise RuntimeError("no controlled layout: "+str(diagnostics))
    path=root/"frames.jsonl"
    path.write_text("".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows))
    result=dict(frames=len(rows),houses=len({r["house"] for r in rows}),layouts=dict(Counter(r["category"] for r in rows if r["view"]==0 and r["stratum"]=="double")),
                chosen_houses_without_eligible_layout=dict(rejected),min_requested_separation_m=min(distances),max_requested_separation_m=max(distances),
                strata_box_counts={s:dict(Counter(len(r["boxes"]) for r in rows if r["stratum"]==s)) for s in ("single","double")})
    (root/"controls_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=="__main__":
    main(*map(Path,sys.argv[1:4]),sys.argv[4],len(sys.argv)>5 and sys.argv[5]=="clocks")

