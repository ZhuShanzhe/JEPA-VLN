"""Blender geometry for genuine three-way object-search forks."""
import json
import math
from pathlib import Path
import random
import sys
import bpy


def cube(name,pos,shape,color,angle=0.):
    bpy.ops.mesh.primitive_cube_add(size=1,location=pos)
    obj=bpy.context.object;obj.name=name;obj.dimensions=shape
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    obj.rotation_euler[2]=angle
    mat=bpy.data.materials.get(color) or bpy.data.materials.new(color)
    mat.diffuse_color=tuple(int(color[i:i+2],16)/255. for i in (0,2,4))+(1.,)
    obj.data.materials.append(mat)


def main(d):
    d.mkdir(parents=True,exist_ok=True)
    protocol=dict(train_houses=[1,2,3,4,5,6],test_houses=[7,8,9,10],
        seed=7461,angles_deg=[0,120,-120],plan_steps=28,return_steps=40,
        budget_branches=2,initial_panorama_steps=12,modes=["aligned_single","rotated_single","absent_cup",
         "absent_sofa","multi_cup","multi_sofa","no_cues_single"],orders="all cyclic shifts of both cue orders",
        targets=["cup","sofa"],minimum_visible_pixels=64,minimum_stop_confidence_margin=.10,
        weak_prior=.18,feedback=.85,visited_gate=-1,
        score="terminal target opportunity if real instance >=64px and <=2m. If both inspections yield no estimated target: output not_found. Count false absence when any unreported target existed.")
    (d/"protocol.json").write_text(json.dumps(protocol,indent=2))
    for h in range(1,11):
        rng=random.Random(7461+h)
        bpy.ops.object.select_all(action="SELECT");bpy.ops.object.delete()
        width=rng.uniform(6.1,7.2);wall=rng.choice(["a1b6b3","b9ac9a","99ac8f","ab9fac"])
        cube("floor",(0,0,-.1),(width*2,width*2,.2),rng.choice(["877566","997e63","937f70"]))
        for sign in (-1,1):
            cube("outside",(sign*width,0,1.4),(.14,2*width,2.8),wall)
            cube("outside",(0,sign*width,1.4),(2*width,.14,2.8),wall)
        for direction in (0,120,-120):
            theta=math.radians(direction);co,si=math.cos(theta),math.sin(theta)
            def at(x,y):return (co*x-si*y,si*x+co*y)
            cube("occluder",(*at(0,1.9),1.4),(rng.uniform(2.48,2.56),.15,2.8),wall,theta)
            cube("recess back",(*at(-2.1,5.95),1.32),(2.55,.16,2.64),wall,theta)
            cube("outer cabinet",(*at(1.3,4.45),.55),(.65,.90,1.1),rng.choice(["746557","637a70","8a716c"]),theta)
        name=f"tri_{h:02}";bpy.ops.export_scene.gltf(filepath=str(d/(name+".glb")),export_format="GLB",export_yup=True)
        (d/(name+".stage_config.json")).write_text(json.dumps(dict(render_asset=name+".glb",
             collision_asset=name+".glb",orient_up=[0,1,0],orient_front=[0,0,-1])))


if __name__=="__main__":main(Path(sys.argv[sys.argv.index("--")+1]).resolve())
