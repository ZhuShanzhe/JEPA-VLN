"""Author materials and colour-marker assets once for the counterbalanced fixture."""
from pathlib import Path
import bpy

root=Path(__file__).resolve().parents[1]
out=root/"attribute_fixture/assets"
out.mkdir(parents=True, exist_ok=True)
def clear():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
def export(name):
    bpy.ops.export_scene.gltf(filepath=str(out/(name+".glb")),export_format="GLB",export_yup=True)
def color(mat, rgb):
    mat.use_nodes=True
    mat.diffuse_color=(*rgb,1)
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value=(*rgb,1)
    mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value=1
for name,rgb in (("red",(.85,.025,.025)),("blue",(.025,.025,.85))):
    clear()
    bpy.ops.import_scene.gltf(filepath=str(root/"room/red_chair.glb"))
    mats={m for obj in bpy.context.scene.objects if obj.type=="MESH" for m in obj.data.materials}
    for mat in mats:
        v=mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value
        if v[0]>2.5*v[1]:
            color(mat,rgb)
    export("chair_"+name)
    clear()
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj=bpy.context.object
    obj.dimensions=(.54,.06,.48)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    mat=bpy.data.materials.new("marker_"+name)
    color(mat,rgb)
    obj.data.materials.append(mat)
    export("marker_"+name)
