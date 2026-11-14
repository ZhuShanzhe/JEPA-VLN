"""blender -b -t 2 --python tools/build_room.py -- room"""
import json
import sys
from pathlib import Path

import bpy

out = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
out.mkdir(parents=True, exist_ok=True)
materials = {}

def empty():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

def block(name, center, dimensions, color):
    if color not in materials:
        mat = bpy.data.materials.new(color)
        rgb = tuple(int(color[i:i+2], 16) / 255 for i in (0, 2, 4))
        mat.diffuse_color = (*rgb, 1.)
        mat.use_nodes = True
        mat.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (*rgb, 1.)
        materials[color] = mat
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(materials[color])

def export(name):
    bpy.ops.export_scene.gltf(filepath=str(out / name), export_format="GLB", export_yup=True)

empty()
block("floor", (0, 0, -.10), (8.4, 9, .20), "a99b80")
for x in (-4.2,4.2):
    block("side wall", (x,0,1.5), (.12,9,3), "c5c9c0")
for y in (-4.5,4.5):
    block("end wall", (0,y,1.5), (8.4,.12,3), "c2ba9e")
block("occluder", (0,-1.65,1.1), (2.45,.15,2.2), "626e76")
for side in (-1,1):
    for i in range(5):
        block("wall panel", (side*(1.3+.42*i),4.41,1.4), (.29,.035,1.10),
              ["374958","6d7966","94765c"][i%3])
export("room.glb")

empty()
block("seat",(0,0,.52),(.72,.74,.11),"b11615")
block("back",(0,.31,.91),(.72,.12,.68),"be1916")
for x in (-.26,.26):
    for y in (-.26,.26):
        block("leg",(x,y,.24),(.08,.08,.48),"3f3022")
export("red_chair.glb")
(out/"chair.object_config.json").write_text(json.dumps(dict(
    render_asset="red_chair.glb", collision_asset="red_chair.glb",
    mass=1.0, COM=[0,0,0], use_bounding_box_for_collision=False)))
(out/"room.stage_config.json").write_text(json.dumps(dict(render_asset="room.glb",
    collision_asset="room.glb", orient_up=[0,1,0], orient_front=[0,0,-1])))
