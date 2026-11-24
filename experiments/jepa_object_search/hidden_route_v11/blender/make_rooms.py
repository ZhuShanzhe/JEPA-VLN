"""Generate new three-entrance Blender scenes and separately varied target/landmark assets."""
import bpy
import colorsys
import json
import math
from pathlib import Path
import random
import sys


def clear():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge(do_recursive=True)


def material(name, hsv):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*colorsys.hsv_to_rgb(*hsv), 1)
    return m


def box(name, location, size, mat, yaw=0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    ob = bpy.context.object
    ob.name, ob.dimensions = name, size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    ob.rotation_euler[2] = yaw
    ob.data.materials.append(mat)


def make(dst, source, house):
    folder = dst / "assets" / f"tri11_{house:02}"
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(113003 + house * 14011)
    clear()
    def color(name, sat, light):
        return material(name, (rng.random(), sat, light))
    wall = color("wall", rng.uniform(.04,.25), rng.uniform(.62,.85))
    accent = color("trim", rng.uniform(.12,.38), rng.uniform(.36,.65))
    floor = color("floor", rng.uniform(.12,.35), rng.uniform(.35,.59))
    width = rng.uniform(6.25,7.45)
    box("floor", (0,0,-.11), (2*width,2*width,.22), floor)
    for sign in (-1,1):
        box("boundary", (sign*width,0,1.5), (.14,2*width,3), wall)
        box("boundary", (0,sign*width,1.5), (2*width,.14,3), wall)
    for direction in (0,120,-120):
        t = math.radians(direction)
        def place(name,x,y,z,w,h,d,mat):
            co, si = math.cos(t), math.sin(t)
            box(name,(co*x-si*y,si*x+co*y,z),(w,h,d),mat,t)
        screen_y=rng.uniform(1.86,1.96)
        place("screen",0,screen_y,1.4,rng.uniform(2.47,2.55),.15,2.8,wall)
        place("screen trim",0,screen_y-.08,2.20,2.45,.01,rng.uniform(.04,.15),accent)
        place("alcove back",-2.05,rng.uniform(5.9,6.1),1.4,rng.uniform(2.4,2.65),.15,2.8,wall)
        place("cabinet",rng.uniform(1.08,1.65),rng.uniform(4.1,4.8),.55,rng.uniform(.50,.85),.8,1.1,accent)
        place("high shelf",rng.uniform(-.7,.7),5.8,2.1,rng.uniform(.5,1.3),.20,.08,accent)
    bpy.ops.export_scene.gltf(filepath=str(folder / "room.glb"), export_format="GLB", export_yup=True)
    (folder / "room.stage_config.json").write_text(json.dumps(dict(render_asset="room.glb",collision_asset="room.glb",orient_up=[0,1,0],orient_front=[0,0,-1])))
    for name in ("cup","sofa","red_chair","coffee_station","lamp"):
        clear()
        bpy.ops.import_scene.gltf(filepath=str(source/(name+".glb")))
        for obj in list(bpy.context.scene.objects):
            if obj.type!="MESH": continue
            for slot in obj.material_slots:
                mat=slot.material
                if not mat: continue
                hsv=(rng.random(),rng.uniform(.14,.65),rng.uniform(.45,.92))
                rgba=(*colorsys.hsv_to_rgb(*hsv),1.)
                mat.diffuse_color=rgba
                if mat.use_nodes:
                    for n in mat.node_tree.nodes:
                        if n.type=="BSDF_PRINCIPLED": n.inputs["Base Color"].default_value=rgba
            obj.scale.x *= rng.uniform(.94,1.06)
            obj.scale.z *= rng.uniform(.92,1.08)
        bpy.ops.export_scene.gltf(filepath=str(folder/(name+".glb")),export_format="GLB",export_yup=True)
        (folder/(name+".object_config.json")).write_text((source/(name+".object_config.json")).read_text())


if __name__=="__main__":
    dst,source=sys.argv[sys.argv.index("--")+1:]
    for house in range(1,17):
        make(Path(dst),Path(source),house)
