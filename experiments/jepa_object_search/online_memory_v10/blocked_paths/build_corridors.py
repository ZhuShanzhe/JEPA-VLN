"""Generate geometry for fourteen disjoint, seeded corridor layouts with movable full-width gates."""
import bpy
import json
import math
import random
import sys
from pathlib import Path

out=Path(sys.argv[-1]);out.mkdir(parents=True,exist_ok=True)

def clear():
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def box(name,xyz,size,color):
    x,y,z=xyz;a,b,c=size
    bpy.ops.mesh.primitive_cube_add(size=1,location=(x,-z,y))
    o=bpy.context.object;o.name=name;o.dimensions=(a,c,b)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);o.data.materials.append(m)
def export(name,typ):
    bpy.ops.export_scene.gltf(filepath=str(out/(name+'.glb')),export_format='GLB',export_yup=True)
    cfg=dict(render_asset=name+'.glb',collision_asset=name+'.glb')
    if typ=='stage':cfg.update(orient_up=[0,1,0],orient_front=[0,0,-1])
    else:cfg.update(mass=1.,COM=[0,0,0],use_bounding_box_for_collision=False)
    (out/f'{name}.{typ}_config.json').write_text(json.dumps(cfg))
def color(r,lo=.16,hi=.75):
    return tuple(r.uniform(lo,hi) for _ in range(3))
spec=[]
for i in range(14):
    name=f'h{i:02d}';r=random.Random(91107+263*i);width=r.uniform(1.45,2.22);height=r.uniform(2.4,3.2)
    length=r.uniform(4.0,4.65);dist=[r.uniform(1.65,2.4),r.uniform(1.65,2.4)]
    floor=color(r);wall=color(r);far=color(r);gate=color(r)
    clear()
    box('floor',(0,-.07,0),(2*length,.14,width+.5),floor)
    for s in (-1,1):
        box('side_wall',(0,height/2,s*(width/2+.08)),(2*length,height,.16),wall)
        box('end_wall',(s*(length+.08),height/2,0),(.16,height,width+.45),far)
        for k in range(6):
            x=r.uniform(-length+.2,length-.2);y=r.uniform(.4,1.9)
            box('wall_detail',(x,y,s*(width/2-.02)),(r.uniform(.14,.62),r.uniform(.12,.42),.025),color(r))
    # Individual floor inlays add depth texture and make RGB-only collision sensing possible.
    for x in range(-7,8):
        box('floor_inlay',(x*.50,.004,0),(.016,.008,width-.05),color(r,.12,.63))
    export(name,'stage')
    clear();box('gate',(0,1.07,0),(.16,2.14,width+.25),gate)
    for t in (-.64,-.05,.61):
        box('crossbar',(0,1.05+t,0),(.185,.035,width+.2),tuple(min(1.,.16+.7*c) for c in gate))
    export(name+'_gate','object')
    spec.append(dict(id=name,width=width,dist=dist,near_target_x=[1.32,1.32],near_target_z=width*.27))
clear();box('model_target', (0,.63,0),(.14,.55,.16),(.75,.13,.07));export('small_goal','object')
(out/'spec.json').write_text(json.dumps(spec,indent=2))
