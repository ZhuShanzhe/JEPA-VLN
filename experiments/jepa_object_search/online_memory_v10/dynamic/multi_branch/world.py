"""Three physical routes through a single Habitat world. Training and live use the same API."""
import argparse
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import sys
import habitat_sim as hs
from habitat_sim.utils.common import quat_from_angle_axis
import magnum as mn
import numpy as np
from PIL import Image

src=Path(__file__).resolve()
sys.path[:0]=[str(src.parents[1]),str(src.parents[2]/"code")]
from collect_streams import ACTIONS,PLANS,configure

ANGLES=(0.,2*math.pi/3,-2*math.pi/3)
PLANS3=([1,2,1,2]+PLANS[0],[1]*4+PLANS[0],[2]*4+PLANS[0])
CATEGORIES=("cup","sofa","red_chair")
PATTERNS={"aligned_single":[0,1,2],"rotated_single":[2,0,1],"absent_cup":[2,1,2],
          "absent_sofa":[0,2,2],"multi_cup":[0,1,0],"multi_sofa":[0,1,1],"no_cues_single":[0,1,2]}


class World:
    def __init__(self,d,assets,h):
        self.d=d;self.h=h;self.data=d/"raw"/f"tri_{h:02}";self.data.mkdir(parents=True,exist_ok=True)
        self.sim=hs.Simulator(configure(d/"assets"/f"tri_{h:02}.stage_config.json"))
        self.om=self.sim.get_rigid_object_manager();tm=self.sim.get_object_template_manager()
        self.ids={s:tm.load_configs(str(assets/(s+".object_config.json")))[0] for s in
                  CATEGORIES+("coffee_station","lamp")}
        self.objects=[];self.agent=self.sim.get_agent(0);self.cache=set()

    def setup(self,mode,order):
        for obj in self.objects:self.om.remove_object_by_id(obj.object_id)
        self.objects=[];self.targets=[];self.classes=[];self.mode=mode;self.order=order
        def xyz(branch,x,distance):
            co,si=np.cos(ANGLES[branch]),np.sin(ANGLES[branch])
            return [co*x-si*distance,0.,-(si*x+co*distance)]
        def add(template,p,sem):
            obj=self.om.add_object_by_template_id(self.ids[template])
            obj.translation=mn.Vector3(*p);obj.motion_type=hs.physics.MotionType.STATIC;obj.semantic_id=sem
            self.objects.append(obj)
            return obj
        for i,rank in enumerate(order):
            cat=PATTERNS[mode][rank]
            self.targets.append(add(CATEGORIES[cat],xyz(i,-2.03,5.4),101+i))
            self.classes.append(cat)
            if mode!="no_cues_single" and rank!=2:
                add(("coffee_station","lamp")[rank],xyz(i,.58,1.40),301+i)
        nav=hs.NavMeshSettings();nav.set_defaults();nav.agent_height,nav.agent_radius,nav.include_static_objects=1.45,.12,True
        assert self.sim.recompute_navmesh(self.sim.pathfinder,nav)
        self.origin=np.asarray(self.sim.pathfinder.snap_point(np.array([0.,0.,0.],np.float32)))
        self.reset()

    def reset(self):
        st=hs.AgentState();st.position=self.origin;st.rotation=quat_from_angle_axis(0.,np.asarray([0.,1.,0.]))
        self.agent.set_state(st)

    def observe(self):
        obs=self.sim.get_sensor_observations()
        rgb=np.ascontiguousarray(obs["rgb"][:,:,:3]);seg=obs["instance"]
        return rgb,np.array([np.count_nonzero(seg==101+i) for i in range(3)]),np.array([np.count_nonzero(seg==301+i) for i in range(3)])

    def image_key(self,im):
        digest=hashlib.sha256(im).hexdigest()
        if digest not in self.cache:
            path=self.data/(digest+".png")
            if not path.exists():Image.fromarray(im).save(path,compress_level=1)
            self.cache.add(digest)
        return digest

    def pano(self):
        views=[];px=[];cues=[]
        for i in range(12):
            rgb,area,context=self.observe()
            if i in (0,4,8):views.append(rgb)
            px.append(area);cues.append(context)
            self.agent.act("turn_left")
        if np.max(px)>0:raise RuntimeError(f"Leaked goal or distractor {self.h} {self.mode} {self.order} {np.max(px,axis=0)}")
        if self.mode!="no_cues_single" and sum(np.max(cues,axis=0)>=64)!=2:
            raise RuntimeError("Insufficient visible entrance cues")
        return views

    def go(self,side):
        gain=np.zeros(3,bool);length=0.;hits=0
        for a in PLANS3[side]:
            start=self.agent.get_state().position.copy()
            hits+=int(self.agent.act(ACTIONS[a]));pos=self.agent.get_state().position.copy()
            length+=float(np.linalg.norm(pos-start))
            rgb,px,_=self.observe()
            near=np.array([np.linalg.norm((np.array(o.translation)-pos)[[0,2]])<=2. for o in self.targets])
            gain|=(px>=64)&near
        if abs(length-5.)>.015 or gain.tolist()!=[j==side for j in range(3)] or not (px[side]>=64 and near[side]):
            raise RuntimeError(f"Goal geometry {self.h} {self.mode} {side} {length} {gain} {px}")
        return rgb,int(self.classes[side]),int(px[side]),hits

    def back(self,side):
        for a in [1]*6+[{0:0,1:2,2:1}[b] for b in reversed(PLANS3[side])]+[1]*6:
            self.agent.act(ACTIONS[a])
        if np.linalg.norm(self.agent.get_state().position-self.origin)>.035:
            raise RuntimeError(f"Failed physical return {self.h} {side}")

    def close(self):
        self.sim.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("dst",type=Path);ap.add_argument("objects",type=Path)
    ap.add_argument("house",type=int);a=ap.parse_args()
    w=World(a.dst.resolve(),a.objects.resolve(),a.house);manifest=[]
    try:
        for mode in PATTERNS:
            for p in permutations(range(3)):
                w.setup(mode,p);start=[w.image_key(img) for img in w.pano()];future=[]
                for branch in range(3):
                    w.reset();rgb,cls,px,hit=w.go(branch);w.back(branch)
                    future.append(dict(frame=w.image_key(rgb),category=cls,pixels=px,collisions=hit))
                manifest.append(dict(mode=mode,order=p,start=start,future=future))
        (w.data/"manifest.json").write_text(json.dumps(manifest))
        print("VERIFIED",a.house,len(manifest),flush=True)
    finally:w.close()


if __name__=="__main__":main()
