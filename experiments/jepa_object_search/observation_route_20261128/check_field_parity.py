"""Recover the 32 preregistered decisions and compare geometry with the saved branch experiment."""
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import probe_branches as probe
from gateway import ObservationGateway
from local_runtime import propose_sites
from multi_policy import located_cue
from routes import observation_routes
from terminal_prefix import with_terminal_prefixes


def run():
    counts = {"discover":0, "verify":0, "strict_endpoint_count":0, "compared_options":0}
    signatures=[]
    for index,spec in enumerate(probe.read(probe.FAMILY/"panel.json")):
        original={r["step"]:r for r in probe.read(probe.HERE/"probe_private"/f"branches_{index}.json")}
        def compare(sim,scan,grid,seen,pose,event,mode,goal,channel,blocked):
            step=event["step"]
            budget=300-step
            previous=original[step]
            assert mode==previous["mode"]
            frames=list(scan.frames)
            views=[frames[i]["rgb"] for i in (1,0,3)]
            geometry,receipt=scan.packet(views,frames[0]["depth"],pose)
            assert receipt==event["physical_scan"]
            box=previous["box"]
            anchors=located_cue(box,views,grid,pose,1.,geometry,90.)[0] if box else []
            gate=ObservationGateway()
            admitted,_=gate.admit(upper_mode=event["gpt_mode"],accepted_cue=box is not None,
                anchors=anchors,blocked=blocked,budget=budget,region=event["executed_region_id"],
                occupied=event.get("commit_previous_id") is not None)
            assert admitted==previous["approved"]
            base=list(event["actions"][:4])
            options=[]
            if admitted:
                p=observation_routes(grid,pose,1.,mode=mode,region_actions=event["actions"],budget=budget,
                    observed_surfaces=seen,anchor_xy=anchors[0]["anchor_xy"] if anchors else None)
                if mode=="verify":
                    assert np.allclose(anchors[0]["anchor_xy"],event["cue_anchor_xy"],atol=1e-6)
                    strict=propose_sites(grid,pose,1.,anchors,budget=min(budget,24))
                    counts["strict_endpoint_count"]+=len(strict)
                    assert not strict and with_terminal_prefixes(p,strict,grid,pose,1.,anchors[0]["anchor_xy"],budget) is p
                if mode=="discover":
                    assert gate.discoverable(p,pose,1.,budget)[1]==previous["gate"]
                options=p["candidates"]
            if not options:
                options=[dict(id="continue",actions=base,geometry_score=None,new_surfaces=None)]
            signature=[(r["id"],r["actions"],r["geometry_score"],r["new_surfaces"]) for r in options]
            saved=[(r["id"],r["actions"],r["geometry_score"],r["map_forecast"]) for r in previous["outcomes"]]
            assert signature==saved,(index,step,mode)
            counts[mode]+=1
            counts["compared_options"]+=len(signature)
            signatures.append((index,step,mode,signature))
            return dict(gate="unchanged",outcomes=options,selected=previous["selected"])
        with patch.object(probe,"snapshot",compare):
            probe.run_house(index,spec,None)
    assert counts["discover"]==24 and counts["verify"]==8
    return dict(**counts,ordered_geometry_and_actions_identical=True,model_inference=False,paid_calls=0,
                ordered_signature_sha256=sha256(json.dumps(signatures,separators=(",",":")).encode()).hexdigest())


if __name__=="__main__":
    result=run()
    output=Path(__file__).with_name("field_parity.json")
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
