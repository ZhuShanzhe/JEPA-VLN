"""Re-run the frozen visual identity check on every later weak capture with its acknowledged reference."""
import ast
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from appearance import _rgb, build_reference
from audit_clock import PROJECT, STAGE8, accepted, digest, lines, load
from local_match import match_prepared, prepare_reference
from proposal_worker import ProposalWorker
from trusted_reference import assess_reference, MIN_DETECTOR_SCORE
from verify_rank import deny_ip_sockets


def archived_frame_reader(lock,files):
    # Execute the frozen mapper's own file-based accessor in isolation. The
    # solver is unnecessary for stored keyframes; depth and poses stay absent.
    filename, = [Path(p) for p in lock["files"] if p.endswith("/mapping/server.py")]
    assert digest(filename)==lock["files"][str(filename)]
    tree=ast.parse(filename.read_text())
    cls,=[c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=="MappingServer"]
    wanted={"_vggt_frame_grid","_frame_rgb","get_frame_image"}
    cls.body=[f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name in wanted]
    assert len(cls.body)==len(wanted) and not cls.bases
    code=ast.Module(body=[cls],type_ignores=[])
    context={"np":np,"cv2":cv2,"os":os,"Image":Image}
    exec(compile(code,str(filename),"exec"),context)
    reader=context["MappingServer"]()
    reader._locate_frame=lambda fid:None
    reader._keyframe_files=files
    def read(fid):
        meta,raw=reader.get_frame_image(fid)
        assert meta["found"] and meta["source"]=="keyframe_file"
        return _rgb(raw)
    return read


def main():
    deny_ip_sockets()
    assert (STAGE8/"closed.json").exists()
    row,=load(STAGE8/"panel.json")["rows"]
    name=row["episode_id"]+".json"
    path,doc,ep=accepted(STAGE8/"comparison/accepted/second"/name)
    assert digest(path)==load(STAGE8/"score_receipts"/name)["source_result_sha256"]
    lock=load(STAGE8/"comparison/lock.json")
    matcher=PROJECT/"identity_stage7_transfer/local_match.py"
    assert digest(matcher)==lock["files"][str(matcher)]
    agent=path.parent/"agent"
    notes=lines(agent/"confirm_events.jsonl")
    decisions=lines(agent/"decision_trace.jsonl")
    refs=[r for r in notes if r["event"]=="confirmed_reference_pinned"]
    confirmed,=refs
    receipt,=[r for r in notes if r["event"]=="report_action_acknowledged"
              and r["instance"]==confirmed["instance"] and r["step"]==confirmed["step"]]
    report,=[r for r in ep["report_events"] if r["step"]==receipt["source_step"]]
    assert len(report["new_targets"])==1
    src,=[r for r in notes if r["event"]=="reference_quality" and r["trusted"] and r["frame"]==confirmed["frame"]]
    obs,=[o for o in ep["paper"]["agent_trace"]["observations"] if o["observation_id"]==src["observation"]]
    capture=[r for r in notes if r["event"]=="weak_relook_capture" and r["step"]>confirmed["step"]]
    category,={q["category"] for r in decisions if (q:=r["world_state"].get("local_rgb_proposals")) and q["category"]}
    service=STAGE8.parents[1]/"runs"/load(STAGE8/"mapper_process.json")["run"]
    keyframes=service/"mapping/keyframes"
    files={f:keyframes/("frame_%06d.png"%f) for f in [src["frame"],*[c["frame"] for c in capture]]}
    read=archived_frame_reader(lock,files)
    source=read(src["frame"])
    reference=build_reference(source,obs["bbox"],point_1000=obs["pixel"])
    source_box=np.asarray(reference.roi)*np.tile(source.shape[1::-1],2)
    model=ProposalWorker(PROJECT/"b_verification/env/bin/python",PROJECT,"cuda")
    results=[]
    try:
        # Re-establish source fidelity using the online grader and identical class text.
        from source_quality import coarse_category
        det=model.detect(source,coarse_category(category))
        test=assess_reference(source,source_box,np.array(obs["pixel"])*source.shape[1::-1]/1000.,det)
        assert test.eligible() and abs(test.detection_score-src["detector"])<.001
        assert abs(test.occupancy-src["occupancy"])<.02
        template=prepare_reference(source,source_box)
        for c in capture:
            live=_rgb(np.asarray(Image.open(files[c["frame"]]).convert("RGB")))
            rr={}
            for tag,rgb in [("live_lossless_rgb",live),("frozen_mapper_rgb",read(c["frame"]))]:
                boxes=[b["bbox"] for b in model.detect(rgb,category) if b["score"]>=MIN_DETECTOR_SCORE]
                # Applying no semantic or geometry filter, verify each high-confidence
                # proposal on its own, as in the real _arrival_matches(other,...,[box]).
                n=sum(match_prepared(template,rgb,[box])==0 for box in boxes)
                unique=match_prepared(template,rgb,boxes)
                assert (unique is not None)==(n==1)
                rr[tag]=dict(high_confidence_candidates=len(boxes),individually_match_pinned_reference=n,
                             unique_correspondence=unique is not None)
            results.append(dict(step=c["step"],**rr))
    finally:
        model.close()
    result=dict(scope="Counterfactual visual subgate only, using the officially credited report's pinned source, the frozen mapper's recorded keyframe files, all subsequent weak captures and every high-confidence proposal. No geometry eligibility, reports or actions simulated; no semantic labels filter any proposal.",
                result_sha256=digest(path),matcher_sha256=digest(matcher),report_step=report["step"],
                pinned_at_step=confirmed["step"],reference_source_step=obs["step"],
                source_keyframe_sha256=digest(files[src["frame"]]),online_source_foreground_fraction=src["occupancy"],
                reconstructed_source_foreground_fraction=test.occupancy,online_source_detector_score=src["detector"],
                reconstructed_source_detector_score=test.detection_score,detector_query=category,
                threshold=MIN_DETECTOR_SCORE,paid_calls=0,results=results)
    dest=Path(__file__).with_name("matcher_after_delay.json")
    if dest.exists():
        assert load(dest)==result
    else:
        dest.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,allow_nan=False))


if __name__=="__main__":
    main()
