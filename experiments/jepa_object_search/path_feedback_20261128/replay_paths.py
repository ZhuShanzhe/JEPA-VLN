"""Independent replay of deterministic source states and sibling plans on train, calibration and test."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from collect_paths import HERE,SOURCE_H,ASSETS,simulator,houses,inventory,choose_sources,SAMPLES_PER_HOUSE,camera,options


def confirm(split,house):
    path=HERE/"private_data"/(split+"_"+house+".npz")
    raw=HERE/"private_render"/path.stem
    seed=int(hashlib.sha256(("path_feedback:paired:v1:"+house).encode()).hexdigest()[:8],16)%(2**31-1)
    sim=simulator(house);sim.seed(seed);sim.pathfinder.seed(seed)
    result=dict(source_rgb_jpeg_sha_matched=0,plans=0,first_collision_index_mismatches=0,
                same_instance_pixel_mismatches=0,issued_actions=0,reached=0)
    try:
        start=choose_sources(sim,inventory(sim,ASSETS),np.random.default_rng(seed),SAMPLES_PER_HOUSE)
        with np.load(path,allow_pickle=False) as z:
            ag=sim.get_agent(0)
            for index in (0,len(start)-1):
                cat,sid,p,yaw,cue=start[index]
                origin=camera(sim,p,yaw,SOURCE_H,0)
                state=ag.get_state()
                mem=io.BytesIO()
                Image.fromarray(origin["rgb"][...,:3]).save(mem,format="JPEG",quality=92)
                assert mem.getvalue()==(raw/f"{index}_source.jpg").read_bytes()
                result["source_rgb_jpeg_sha_matched"]+=1
                for distance in (2,8,20):
                    ii=np.flatnonzero(z["group"]==f"{house}/{index}/d{distance}")
                    assert len(ii)==7
                    for q in ii:
                        ag.set_state(state,reset_sensors=False,infer_sensor_states=False)
                        blocked=0
                        for step,a in enumerate(z["action"][q][z["action"][q]>0],1):
                            prior=ag.state.position.copy()
                            contact=ag.act({1:"move_forward",2:"turn_left",3:"turn_right"}[int(a)])
                            result["issued_actions"]+=1
                            delta=ag.state.position-prior
                            if a==1 and (np.linalg.norm(delta[[0,2]])<.23 or contact):
                                blocked=step
                                break
                        assert blocked==z["first_blocked"][q]
                        assert np.allclose(ag.state.position,[z["end_x"][q],z["end_y"][q],z["end_z"][q]],atol=.001)
                        if z["reach"][q]:
                            obs=sim.get_sensor_observations()
                            pixels=int((obs["ids"]==sid).sum())
                            assert pixels==z["pixels"][q]
                            result["reached"]+=1
                        result["plans"]+=1
    finally:sim.close()
    return result


if __name__=="__main__":
    report={s:confirm(s,sorted(hh)[0]) for s,hh in houses().items()}
    (HERE/"physical_replay.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)
