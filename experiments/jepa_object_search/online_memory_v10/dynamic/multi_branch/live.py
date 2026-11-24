"""Physically execute each policy in a single three-bay Habitat environment."""
import argparse
from itertools import permutations
import json
from multiprocessing.connection import Connection
from pathlib import Path
import socket
import subprocess
import sys
import numpy as np
from world import World,PATTERNS


ARMS=("latent_update","direct_update","uniform_visit","latent_visit","direct_visit")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dst",type=Path);ap.add_argument("objects",type=Path);ap.add_argument("policy_python")
    ap.add_argument("house",type=int);a=ap.parse_args();d=a.dst.resolve()
    w=World(d,a.objects.resolve(),a.house)
    local,remote=socket.socketpair()
    proc=subprocess.Popen([a.policy_python,str(Path(__file__).resolve().with_name("heads.py")),str(d),
        str(d.parents[1]),"serve",str(remote.fileno())],pass_fds=[remote.fileno()])
    remote.close();conn=Connection(local.detach())
    def call(typ,images):
        conn.send((typ,images));return conn.recv()
    out=d/"metrics";out.mkdir(exist_ok=True)
    result=[]
    try:
        for mode in PATTERNS:
            for order in permutations(range(3)):
                w.setup(mode,order)
                for arm in ARMS:
                    for target in range(2):
                        # Each per-agent world is reset only here. Its return, second choice
                        # and stop follow the first choice's exact physical image history.
                        w.reset();views=w.pano();values,affinity=call("entrance",views)
                        k=0 if arm.startswith("latent") else 1
                        start_prior=np.array(values[k])[:,target] if arm!="uniform_visit" else np.zeros(3)
                        scores=start_prior.copy();order_vis=[];seen=[];said=None;steps=12;truth_pixels=[]
                        signals=[]
                        for attempt in range(2):
                            branch=int(np.argmax(scores));order_vis.append(branch)
                            rgb,cls,pixels,_=w.go(branch);steps+=28
                            # Policy receives only RGB. Independent instance truth stays in evaluator.
                            detected=np.array(call("terminal",[rgb])[0]);rivals=np.delete(detected,target)
                            support=float(detected[target]-np.max(rivals));signals.append(support)
                            seen.append(cls==target);truth_pixels.append(pixels)
                            if support>=.10:
                                said=branch;break
                            if attempt==0:
                                w.back(branch);steps+=40
                                if arm.endswith("update"):
                                    residual=np.clip(support-start_prior[branch],-1,1)
                                    scores+=.10*.85*residual*np.array(affinity[branch])
                                scores[branch]=-np.inf
                        result.append(dict(mode=mode,order=order,arm=arm,q=target,actions=steps,
                            visited=order_vis,target_count=sum(c==target for c in w.classes),
                            score=signals,report=said,correct_report=bool(said is not None and seen[-1]),
                            physically_found=any(seen),physical_at_first=seen[0],pixels=truth_pixels))
            print("live",a.house,mode,len(result),flush=True)
        (out/f"tri_{a.house:02}.json").write_text(json.dumps(result,allow_nan=False))
    finally:
        try:conn.send(None)
        finally:conn.close();proc.wait();w.close()


if __name__=="__main__":main()
