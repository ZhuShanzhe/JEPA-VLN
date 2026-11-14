"""Run with Blender 3.6.23; combine the original room and chairs into a static GLB."""
import sys
from pathlib import Path
import bpy

root=Path(__file__).resolve().parents[1]
out=root/"most_cli_fixture/scenes/00000-BlenderRoom/BlenderRoom.basis.glb"
out.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
bpy.ops.import_scene.gltf(filepath=str(root/"room/room.glb"))
for x in (-2.25,2.25):
    existing=set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(root/"room/red_chair.glb"))
    for obj in set(bpy.data.objects)-existing:
        if obj.parent is None:
            obj.location.x+=x
            obj.location.y+=1.65
bpy.ops.export_scene.gltf(filepath=str(out),export_format="GLB",export_yup=True)
