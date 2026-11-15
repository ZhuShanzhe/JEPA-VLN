"""Export six controlled indoor layouts and four target meshes with Blender (CPU)."""
import json
from pathlib import Path
import sys

import bpy


def block(name, position, shape, rgb):
    bpy.ops.mesh.primitive_cube_add(size=1, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = shape
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mat = bpy.data.materials.get(rgb) or bpy.data.materials.new(rgb)
    mat.use_nodes = True
    color = tuple(int(rgb[i:i+2],16)/255 for i in (0,2,4)) + (1.,)
    mat.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


def clear():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def save(out, name, stage=False):
    mesh = name + ".glb"
    bpy.ops.export_scene.gltf(filepath=str(out/mesh), export_format="GLB", export_yup=True)
    cfg = dict(render_asset=mesh, collision_asset=mesh)
    if stage:
        cfg.update(orient_up=[0,1,0],orient_front=[0,0,-1])
    else:
        cfg.update(mass=1., COM=[0,0,0], use_bounding_box_for_collision=False)
    ext = ".stage_config.json" if stage else ".object_config.json"
    (out/(name+ext)).write_text(json.dumps(cfg))


def main(out):
    out.mkdir(parents=True,exist_ok=True)
    # x is lateral and Blender +y becomes Habitat -z after GLB conversion.
    layouts = [("open_table","train",4.2,4.8,0),("closed_left","train",4.5,5.,1),
               ("closed_right","train",4.8,4.6,2),("gallery","train",4.2,5.6,3),
               ("offset_gallery","dev",5.2,5.3,4),("split_rear","dev",4.65,5.9,5)]
    for name, split, w, length, kind in layouts:
        clear()
        block("floor",(0,0,-.09),(2*w,2*length,.18),"a58d70")
        for side in (-1,1):
            block("side wall",(side*w,0,1.35),(.14,2*length,2.7),"c8c4b2")
            block("end wall",(0,side*length,1.35),(2*w,.14,2.7),"b3c4be")
            # Distinct visual cues in the two flanks. All query permutations share this geometry.
            x=side*(w-1.)
            if side==(-1 if kind%2==0 else 1):
                block("table top",(x,-.8,.78),(1.15,.75,.10),"955d37")
                for dx in (-.43,.43):
                    block("table leg",(x+dx,-.8,.39),(.11,.56,.78),"72513c")
            else:
                for h in (.38,.89,1.4):
                    block("shelf",(x,-.45,h),(.64,.80,.09),"556b76")
                block("shelf back",(x+side*.31,-.45,.75),(.08,.8,1.5),"536470")
        block("entry screen",(0,-1.82,1.12),(2.03+.07*(kind%3),.14,2.24),"727969")
        if kind in (1,2,4,5):
            side = -1 if kind in (1,4) else 1
            block("alcove back",(side*w/2,2.5,1.35),(w,.16,2.7),"beb494")
        if kind in (3,4,5):
            block("rear divider",(.32 if kind==4 else 0,length/2,1.32),(.16,length-.35,2.64),"a9b0aa")
        if kind==5:
            block("gallery end",(-w+.85,2.15,1.35),(1.7,.16,2.7),"b5ab91")
        save(out,name,True)
    for item,color in (("red_chair","a91e20"),("blue_chair","23479c"),("sofa","357567"),("cup","c7a53a")):
        clear()
        if item=="cup":
            bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=.11,depth=.22,location=(0,0,.31))
            cup=bpy.context.object
            block("display stand",(0,0,.095),(.43,.38,.19),"897969")
            block("handle",(.105,0,.32),(.07,.035,.11),color)
            cup.data.materials.append(bpy.data.materials.get(color))
        else:
            width=1.62 if item=="sofa" else .65
            block("seat",(0,0,.43),(width,.63,.14),color)
            block("back",(0,.24,.73),(width,.13,.55),color)
            if item=="sofa":
                for side in (-1,1):
                    block("arm",(side*.76,0,.58),(.12,.64,.24),color)
            for dx in (-.38,.38):
                for dy in (-.21,.21):
                    block("leg",(dx*width,dy,.19),(.08,.08,.38),"51493c")
        save(out,item)
    (out/"layouts.json").write_text(json.dumps([dict(house=x[0],split=x[1]) for x in layouts]))


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--")+1]).resolve())
