"""Causal, episode-local correction of room forecasts after observing an arrival."""
import math
import numpy as np


def endpoint(xy,yaw,plan,collision=None):
    pos=np.asarray(xy,float).copy()
    if collision is None:collision=np.zeros(len(plan),bool)
    for cmd,bump in zip(plan,collision):
        if cmd==0 and not bump:pos+=.25*np.asarray([math.cos(yaw),math.sin(yaw)])
        elif cmd==1:yaw+=math.pi/6
        elif cmd==2:yaw-=math.pi/6
    return pos,yaw


class EpisodeLayout:
    """Use observe() only for a chosen route. No ground-truth room names are required."""
    def __init__(self):
        self.events=[]
        self.xy=np.zeros(2)
        self.yaw=0.

    def observe(self,previous_forecast,actual_actions,blocked,arrival_room):
        self.xy,self.yaw=endpoint(self.xy,self.yaw,actual_actions,blocked)
        actual=np.asarray(arrival_room,float)
        if actual.ndim==2:actual=actual.mean(0)
        forecast=np.atleast_2d(previous_forecast)
        self.events.append((self.xy.copy(),self.yaw,actual,actual-forecast))

    def features(self,prediction,plans):
        p=np.asarray(prediction,float)
        if p.ndim==2:p=p[None]
        if p.shape[1]!=3:raise ValueError('expected [seeds,3,rooms]')
        x=np.zeros(p.shape+(7,),float)
        if not self.events:return x
        ends,headings=zip(*(endpoint(self.xy,self.yaw,a) for a in plans))
        ends=np.array(ends);headings=np.array(headings)
        total=0.
        for age,(pos,yaw,seen,error) in enumerate(reversed(self.events)):
            decay=.65**age;total+=decay
            distance=np.linalg.norm(ends-pos,axis=1)
            local=np.exp(-distance/2.)
            orient=(1+np.cos(headings-yaw))*.5
            # Hellinger affinity of the forecast with the previously observed environment.
            similar=np.sqrt(p.clip(0)*seen.clip(0)).sum(-1).clip(0,1)**3
            across=np.broadcast_to(local,similar.shape)
            angle=np.broadcast_to(orient,similar.shape)
            kernels=np.stack([np.ones_like(similar),across,similar,across*angle,
                              across*similar,similar*angle],axis=-1)
            x[...,:6]+=decay*error[:,None,:,None]*kernels[:,:,None,:]
        x[...,:6]/=total
        x[...,6]=x[...,4]-x[...,4].mean(axis=1,keepdims=True)
        return x

    def predict(self,prediction,plans,weights,strength):
        corrected=np.asarray(prediction,float)+strength*(self.features(prediction,plans)@weights)
        corrected=np.clip(corrected,1e-6,None)
        return corrected/corrected.sum(-1,keepdims=True)
