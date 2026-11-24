"""Stateful controller for the four physically observed junctions in one search task."""
import json
from pathlib import Path
import sys
import numpy as np
from run_search import pick_route, SETTINGS
from scene_memory import SceneMemory


class EnvironmentSearch:
    def __init__(self,project,target,seed=11,*,horizon=True,layout=False):
        project=Path(project);root=project/'jepa_env_v13'
        sys.path.insert(0,str(root/'models'))
        from env_routes import EnvironmentRoutes
        self.model=EnvironmentRoutes(project,seed)
        self.project=project;self.root=root
        self.calibrator=json.loads((root/'online/cue_calibration.json').read_text())
        choice=json.loads((root/'online/online_selection.json').read_text())['setting']
        self.config=SETTINGS[choice]
        self.long_head=None;self.layout_settings=None
        if horizon:
            sys.path.insert(0,str(root/'horizon'))
            from train_value import ValueResidual
            self.long_head=ValueResidual.load(project,'jepa')
        if layout:
            sys.path.insert(0,str(root/'layout_update'))
            self.layout_settings=json.loads((root/'layout_update/frozen.json').read_text())
        self.reset(target)

    def reset(self,target):
        self.target=target;self.turn=0;self.pending=None;self.arrival=None
        self.past_rooms=[];self.past_cues=[]
        self.memory=SceneMemory(self.model.old.remem(self.project),self.calibrator)
        if self.layout_settings:
            from scene_layout import EpisodeLayout
            self.layout=EpisodeLayout()

    def choose_after_scan(self,views,plans):
        if self.pending is not None or self.turn>=4:
            raise ValueError('consume the selected path before the next decision; at most four junctions')
        views=np.asarray(views); plans=np.asarray(plans)
        if views.shape[0]!=3 or views.dtype!=np.uint8 or plans.shape!=(3,24) or not np.isin(plans,[0,1,2]).all():
            raise ValueError('expected three observed RGBs and three executable 24-action plans')
        sensed,cues=self.model.observe_batch(views,self.target)
        for t,action in enumerate((2,1,1,2)):
            self.memory.observe(action,False,sensed[t] if t<3 else None,float(cues[t]) if t<3 else None)
        out=self.model.live(views,self.target,plans)
        room=out['env_pred_jepa']
        score=np.asarray(out['jepa'],float).copy()
        if self.long_head is not None:
            past_room=np.mean(self.past_rooms,axis=(0,1)) if self.past_rooms else sensed.mean(0)
            past_cue=float(np.mean(self.past_cues)) if self.past_cues else float(np.mean(cues))
            score+=self.long_head.route_delta(out['jepa'],room,out['prior'],sensed,cues,plans,
                                              4-self.turn,past_room,past_cue)
        proposed=room
        if self.layout_settings:
            if self.arrival is not None:
                self.layout.observe(*self.arrival,sensed)
                self.arrival=None
            proposed=self.layout.predict(room[None],plans,self.layout_settings['weights']['jepa'],
                                         self.layout_settings['strength']['jepa'])[0]
            score+=(proposed-room)@out['prior']
        self.past_rooms.append(sensed);self.past_cues.append(cues)
        k,scout=pick_route(score,room,out['prior'],self.memory.components(plans),self.turn,self.config)
        self.pending=plans[k].copy()
        self.forecast=room[k].copy()
        return dict(index=k,expected_room=self.model.rooms[int(np.argmax(proposed[k]))],
                    prior_room=self.model.rooms[int(np.argmax(out['prior']))],scout_changed=scout)

    def consume_walk(self,actions,collisions,frames):
        actions=np.asarray(actions)
        if self.pending is None or not np.array_equal(self.pending,actions):
            raise ValueError('walk does not match current selected plan')
        if len(collisions)!=24 or len(frames)!=4:raise ValueError('missing physical feedback')
        sensed,cues=self.model.observe_batch(np.asarray(frames),self.target)
        for t,(a,c) in enumerate(zip(actions,collisions),1):
            i=t//6-1 if t%6==0 else None
            self.memory.observe(a,c,None if i is None else sensed[i],None if i is None else float(cues[i]))
        if self.layout_settings:
            self.arrival=(self.forecast.copy(),actions.copy(),np.asarray(collisions).copy())
        self.turn+=1;self.pending=None
