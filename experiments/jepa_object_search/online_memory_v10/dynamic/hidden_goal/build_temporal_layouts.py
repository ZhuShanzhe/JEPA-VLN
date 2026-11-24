"""Create the six unseen rooms for a runtime relocation test, with a fixed protocol."""
import json
from pathlib import Path
import random
import sys
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_layouts import cube


def main(out):
    out.mkdir(exist_ok=True,parents=True)
    protocol=dict(rooms=[f"dynamic_{h:02}" for h in range(21,27)],seed=65479,
        events=["unchanged","physical_target_swap"],cues=["congruent","conflicting","hidden"],
        categories=["cup","sofa"],poses="both target orders, live Habitat; two 24-action candidate routes",
        memory="after sufficient route-local inspection, EWMA signed visual evidence (85% newest) with a 96-step half life",
        score="0.18 x max-normalized prior + upcoming route evidence. Equal starts split across target queries. Ties half.",
        hypotheses=["same prediction before a differentiating observation","improve after physical re-observation",
                    "compare same memory with direct supervised and constant priors"])
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2))
    for i,name in enumerate(protocol["rooms"]):
        rng=random.Random(65479+i)
        w=rng.uniform(4.4,5.6);l=rng.uniform(4.6,5.7)
        bpy.ops.object.select_all(action="SELECT");bpy.ops.object.delete()
        wall=rng.choice(["b4a7a1","899bb7","b4b594"]);wood=rng.choice(["8a716b","84774d","747d76"])
        cube("floor",(0,0,-.1),(2*w,2*l,.20),wood)
        for s in (-1,1):
            cube("side",(s*w,0,1.37),(.13,2*l,2.74),wall)
            cube("end",(0,s*l,1.37),(2*w,.13,2.74),wall)
            cube("alcove",(s*w*.56,2.52,1.32),(w*.78,.17,2.64),wall)
            for k in range(3):
                cube("furniture",(s*(w-.48),rng.uniform(-l+1,l-1),rng.uniform(.4,.65)),
                     (.64,rng.uniform(.3,1.4),rng.uniform(.75,1.2)),rng.choice(["8b614b","618286","82766d"]))
        cube("fork",(0,-1.82,1.43),(rng.uniform(2.49,2.55),.15,2.86),wall)
        cube("rear partition",(rng.uniform(-.19,.19),3.37,1.32),(.17,2.4,2.64),wall)
        bpy.ops.export_scene.gltf(filepath=str(out/(name+".glb")),export_format="GLB",export_yup=True)
        (out/(name+".stage_config.json")).write_text(json.dumps(dict(render_asset=name+".glb",
            collision_asset=name+".glb",orient_up=[0,1,0],orient_front=[0,0,-1])))


if __name__=="__main__":main(Path(sys.argv[sys.argv.index("--")+1]).resolve())
