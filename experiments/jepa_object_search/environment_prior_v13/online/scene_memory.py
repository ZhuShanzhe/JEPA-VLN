"""Query-specific, episode-local room evidence and spatial exploration for route selection."""
import math
import numpy as np


def axis(yaw):
    return np.array([math.cos(yaw),math.sin(yaw)])


def normalize(row):
    row=np.asarray(row,float)
    if not np.isfinite(row).all() or (row<0).any() or (row.sum(axis=-1)<=0).any():
        raise ValueError('invalid room posterior')
    return row/row.sum(axis=-1,keepdims=True)


class SceneMemory:
    def __init__(self, route_memory, cue_model):
        self.basic=route_memory
        self.basic.reset()
        self.cue_model=cue_model
        self.xy=np.zeros(2);self.yaw=0.;self.time=0
        self.cells={};self.visited={(0,0)}

    @staticmethod
    def cell(xy):
        return tuple(np.floor(xy/.75+.5).astype(int))

    def observe(self,action,blocked,environment=None,cue=None):
        if action==0 and not blocked:self.xy=self.xy+.25*axis(self.yaw)
        if action==1:self.yaw+=math.pi/6
        if action==2:self.yaw-=math.pi/6
        self.time+=1
        loc=self.cell(self.xy);self.visited.add(loc)
        self.basic.observe(self.xy,self.time,axis(self.yaw),cue,bool(blocked))
        if environment is not None:
            if cue is None:raise ValueError('a sensed frame must supply both environment and target evidence')
            e=normalize(environment)
            strength=float(np.interp(cue,self.cue_model['raw'],self.cue_model['evidence']))
            # Re-observation replaces the previous reading at this spatial cell and bearing.
            heading=int(round(self.yaw/(math.pi/3)))%6
            self.cells[(loc,heading)]=(self.time,e,strength)

    def components(self,plans):
        plans=np.asarray(plans)
        travel=self.basic.features(plans,self.xy,axis(self.yaw),self.time)
        weights=np.array([self.basic.weights[k] for k in ('novelty','clearance','target_trace')])
        base=travel@weights
        if not self.cells:raise ValueError('scan before selecting a route')
        count=np.zeros_like(next(iter(self.cells.values()))[1]);total=count.copy()
        # Several rays from the same location contribute at most one full observation.
        rays={}
        for (loc,_),record in self.cells.items():rays.setdefault(loc,[]).append(record)
        for records in rays.values():
            for t,e,signal in records:
                amount=math.exp(-(self.time-t)/90.)/len(records)
                count+=amount*e;total+=amount*e*signal
        reference=self.cue_model['train_first_appearance_frequency']
        # A weak negative sighting should not exclude the rest of the room type.
        difference=(total-count*reference)/(2.+count)
        difference=np.clip(difference,-.06,.22)
        supply=[]
        for seq in plans:
            xy=self.xy.copy();yaw=self.yaw;hits=[]
            for act in seq:
                if act==0:
                    xy+=.25*axis(yaw);hits.append(self.cell(xy) not in self.visited)
                elif act==1:yaw+=math.pi/6
                elif act==2:yaw-=math.pi/6
            supply.append(np.mean(hits) if hits else 0.)
        return base,difference,1./np.sqrt(1.+count),np.asarray(supply)

    @staticmethod
    def apply(scores,room_predictions,components,*,gain,exploration):
        base,difference,uncertainty,supply=components
        room=normalize(room_predictions)
        delta=room@difference
        # Room uncertainty alone should not encourage repeatedly checking the same place.
        novelty=supply*(.4+.6*(room@uncertainty))
        return scores+base+gain*(delta-np.mean(delta))+exploration*(novelty-np.mean(novelty))

    def rerank(self,scores,plans,room_predictions,*,gain,exploration):
        return self.apply(scores,room_predictions,self.components(plans),gain=gain,exploration=exploration)
