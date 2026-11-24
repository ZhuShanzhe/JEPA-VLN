"""Locate navigable poses and object anchors in HSSD semantic-region polygons."""
from pathlib import Path
import json
import numpy as np


GROUPS = {
    'bedroom':['bedroom'],
    'bathroom':['bathroom','toilet','spa/sauna'],
    'kitchen':['kitchen'],
    'dining room':['dining room','dining booth'],
    'living room':['living room','familyroom','tv','lounge'],
    'hallway or entrance':['hallway','entryway/foyer/lobby'],
    'office or study':['office','library','classroom','meetingroom/conferenceroom'],
    'closet':['closet'],
    'laundry or utility room':['laundryroom/mudroom','utilityroom/toolroom'],
    'outdoor area':['outdoor','porch/terrace/deck/driveway','balcony'],
    'garage':['garage'],
    'other room':['other room','workout/gym/exercise','rec/game','bar']
}
ROOMS=list(GROUPS)
MAP={name:i for i,room in enumerate(ROOMS) for name in GROUPS[room]}


class Atlas:
    def __init__(self,assets,house):
        source = Path(assets)/'semantics/scenes'/(house+'.semantic_config.json')
        self.regions = json.loads(source.read_text())['region_annotations']
        self.names = np.array([MAP[s['label']] for s in self.regions],int)
        self.bounds = np.array([[s['min_bounds'][0],s['max_bounds'][0],
                                 s['min_bounds'][2],s['max_bounds'][2],
                                 s['floor_height'],s['floor_height']+s['extrusion_height']]
                                for s in self.regions])
        self.edges=[]
        for s in self.regions:
            p=np.array(s['poly_loop'])[:,[0,2]]
            q=np.roll(p,-1,axis=0)
            keep=abs(q[:,1]-p[:,1])>1e-8
            self.edges.append((p[keep],q[keep]))

    def locate(self,xyz,instance=False):
        """For repeated/overlapping annotations keep only an unambiguous label."""
        x,y,z=np.asarray(xyz,dtype=float).reshape(-1,3).T
        out=np.full(len(x),-1,int)
        ambiguous=np.zeros(len(x),bool)
        for i, (x0,x1,z0,z1,y0,y1) in enumerate(self.bounds):
            subset=np.flatnonzero((x>=x0)&(x<=x1)&(z>=z0)&(z<=z1)&(y>=y0-.12)&(y<=y1+.02))
            if not len(subset):
                continue
            p,q=self.edges[i]
            xx=x[subset,None];zz=z[subset,None]
            crosses=((p[:,1]>zz)!=(q[:,1]>zz)) & (xx<(q[:,0]-p[:,0])*(zz-p[:,1])/(q[:,1]-p[:,1])+p[:,0])
            hit=subset[(crosses.sum(1)%2).astype(bool)]
            label=i if instance else self.names[i]
            ambiguous[hit] |= (out[hit]>=0)&(out[hit]!=label)
            out[hit]=label
        out[ambiguous]=-1
        return out
