"""Blender asset construction for a fixed paired unseen-object benchmark."""
import json
from pathlib import Path
import random
import sys
import bpy


def cube(name,position,scale,color):
    bpy.ops.mesh.primitive_cube_add(size=1,location=position)
    m=bpy.context.object
    m.name=name
    m.dimensions=scale
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    mat=bpy.data.materials.get(color) or bpy.data.materials.new(color)
    mat.diffuse_color=tuple(int(color[i:i+2],16)/255. for i in (0,2,4))+(1.,)
    m.data.materials.append(mat)


def main(out):
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(split={"train":list(range(1,9)),"test":list(range(9,15))},
        variants=3,modes=["aligned","contradictory","no_cues"],swap=[0,1],
        actions="symmetric 24 actions, 20 forward at 0.25m",queries=["cup","sofa"],
        train_weight={"aligned":3,"contradictory":1,"no_cues":1},
        outcomes=["full factorial","aligned","contradictory","no cues","75/25 cue-conditioned weighted mixture"],
        seed=23111)
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2))
    for h in range(1,15):
        rng=random.Random(23111+h)
        w=rng.uniform(4.4,5.8);l=rng.uniform(4.5,5.7)
        bpy.ops.object.select_all(action="SELECT");bpy.ops.object.delete()
        wall=rng.choice(["9da9a2","baab94","afbbcb","a9b79f"])
        wood=rng.choice(["926a41","7f875e","8f7867","927c52"])
        cube("floor",(0,0,-.10),(2*w,2*l,.20),wood)
        for s in (-1,1):
            cube("outer side",(s*w,0,1.4),(.14,2*l,2.8),wall)
            cube("end",(0,s*l,1.4),(2*w,.14,2.8),wall)
            for k in range(1+rng.randrange(3)):
                cube("side furniture",(s*(w-.48),rng.uniform(-l+1,l-1),.48),
                     (.63,rng.uniform(.50,1.33),.96),rng.choice(["817477","568483","996b45"]))
            cube("recess wall",(s*w*.55,2.8,1.35),(w*.82,.18,2.7),wall)
        # Wide enough to occlude the target from the complete start panorama, while
        # both instructed paths pass unobstructed around its left and right edges.
        cube("entry divider",(0,-1.82,1.35),(rng.uniform(2.46,2.56),.15,2.7),wall)
        cube("rear divider",(rng.uniform(-.22,.22),3.25,1.32),(.18,2.4,2.64),wall)
        name=f"house_{h:02}"
        bpy.ops.export_scene.gltf(filepath=str(out/(name+".glb")),export_format="GLB",export_yup=True)
        (out/(name+".stage_config.json")).write_text(json.dumps(dict(render_asset=name+".glb",
            collision_asset=name+".glb",orient_up=[0,1,0],orient_front=[0,0,-1])))


if __name__=="__main__":
    main(Path(sys.argv[sys.argv.index("--")+1]).resolve())
