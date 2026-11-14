"""Make public-interface test inputs for one Blender room. No official HM3D inputs used."""
import json
import math
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[1]
test=root/"most_cli_fixture"
ann=test/"annotations/BlenderRoom"
ann.mkdir(parents=True,exist_ok=True)
scenes=test/"scenes"
scenes.mkdir(exist_ok=True)
scene_dir=scenes/"00000-BlenderRoom"
assert (scene_dir/"BlenderRoom.basis.glb").exists()
shutil.copyfile(root/"room/room.navmesh",scene_dir/"BlenderRoom.basis.navmesh")
# The stage loader looks for this sibling descriptor when fed a raw .basis.glb.
# IDs and colours describe this fixture only; official RGB-D config does not render semantics.
(scene_dir/"BlenderRoom.basis.scn").write_text('HM3D Semantic Annotations\n101,FF0000,"chair",0\n102,FF0001,"chair",0\n')
def save(path,data):
    path.write_text(json.dumps(data,indent=2))
save(scenes/"hm3d_annotated_val_basis.scene_dataset_config.json",
     {"stages":{"default_attributes":{"up":[0,1,0],"front":[0,0,-1],"orient_up":[0,1,0],"orient_front":[0,0,-1]},
                "paths":{".glb":["00000-BlenderRoom/BlenderRoom.basis.glb"]}}})
points={101:[-2.25,.20,-.70],102:[2.25,.20,-.70]}
save(ann/"goals.json",{"goals":{str(k):dict(object_id=k,category="chair",position=[v[0],.6,-1.65],
     navigation_view_points=[{"agent_state":{"position":v}}]) for k,v in points.items()}})
instr="Find both red chairs."
save(ann/"queries.json",{"queries":[dict(query_id="chairs",target_count=2,natural_language=[instr],
      query_program={"select":"objects","where":[{"field":"category","op":"==","value":"chair"}]})]})
save(ann/"episodes.json",{"episodes":[dict(episode_id="fixture_"+suffix,scene_id="BlenderRoom",query_id="chairs",
     query_type="many_category",track="category",instruction=instr,target_mode="many",target_count=2,
     target_object_ids=[101,102],target_cluster_sizes={"101":1,"102":1},
     start_position=[-2.75,.20,-.70],start_rotation=[0,-math.sqrt(.5),0,math.sqrt(.5)],
     target_nav_positions=list(points.values()),shortest_path_distance=5.0)
     for suffix in ("tworeports","repeat")]})
