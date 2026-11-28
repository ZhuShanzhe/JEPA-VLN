"""Measure repeated and partially successful actions marked as contact/stall in the local route runs."""
from collections import Counter
import json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def collect(folder):
    result=Counter();distances=[]
    for p in folder.glob('*/*.json'):
        r=json.loads(p.read_text())
        if not r['focus']:continue
        failed=[]
        for t in range(r['baseline_length']+1,len(r['events'])):
            before,after=r['events'][t-1],r['events'][t]
            prev=np.array(before['pose'][:2]);next_xy=np.array(after['pose'][:2])
            if not after['bump']:continue
            h=before['pose'][2]
            length=float(np.linalg.norm(prev-next_xy));distances.append(length)
            result['under_2cm']+=length<.02
            result['moved_at_least_10cm']+=length>=.1
            result['moved_at_least_23cm_but_contact']+=length>=.23
            result['previous_contact_same_pose_20cm_heading_15deg']+=any(np.linalg.norm(prev-u)<.20 and math.cos(h-v)>math.cos(math.pi/12) for u,v in failed)
            failed.append((prev,h))
        result['groups']+=1
    return dict(result,contact_or_stall=len(distances),median_displacement_m=round(float(np.median(distances)),4))
if __name__=='__main__':
    out={k:collect(p) for k,p in {'prior':HERE.parent/'reactive_view_20261129/private_roll','curve':HERE.parent/'layout_detour_20261129/private_motion','grid':HERE/'private_motion'}.items()}
    print(out);(HERE/'contact_diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
