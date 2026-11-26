"""Supply inspectable past views to the native decision loop. No autonomous registration."""
import json
import os
from pathlib import Path
import cv2
import numpy as np
from source_quality import best,coarse_category
from attribute_screen import AttributeWorker,local_crop


class CandidateScoutMixin:
    def _init_report_memory(self):
        super()._init_report_memory()
        self._scout_spec=None
        self._scout_goal=None
        self._scout_checked=set()
        self._scout_views={}
        self._scout_calls={}
        self._scout_done=set()
        self._scout_step=None
        self._scout_last_state=None
        self._scout_images={}
        self._scout_offline=False

    def _scout_remember(self,observation):
        info=getattr(self,"_last_feed_info",None) or {}
        if info.get("is_keyframe") and info.get("frame_id"):
            self._scout_images[int(info["frame_id"])]=cv2.resize(np.asarray(observation.rgb)[...,:3],
                (16,16),interpolation=cv2.INTER_AREA)

    def act(self, observation):
        action=super().act(observation)
        self._scout_remember(observation)
        return action

    def _scout_observe(self,fid,noun):
        meta,jpeg=self.client.get_frame_image(fid)
        if not meta.get("found") or not jpeg:
            return
        bgr=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
        if bgr is None:
            return
        image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        thumb=cv2.resize(image,(16,16),interpolation=cv2.INTER_AREA)
        if np.abs(thumb.astype(float)-self._scout_images[fid]).mean()>10:
            return
        h,w=image.shape[:2]
        rows=self._boxes(("scout",fid),image,noun)
        self._scout_checked.add(fid)
        score,ix=best(self._scout_spec,rows,w,h)
        if ix is None or score<self._scout_spec["threshold"]:
            return
        box=rows[ix]["bbox"]
        if os.environ.get("NAV_SOURCE_SCOUT")!="category_only":
            if not hasattr(self,"_scout_attribute_worker"):
                project=Path(os.environ["NAV_ROOT"])
                self._scout_attribute_worker=AttributeWorker(project,os.environ.get("NAV_BOX_DEVICE","cpu"))
            crop=local_crop(image,box)
            if min(crop.shape[:2])<8 or self._scout_attribute_worker.margin(crop,self._scout_goal)<=0:
                return
        normalized=np.clip(np.asarray(box,float)*[1000/w,1000/h,1000/w,1000/h],0,1000).round().astype(int)
        # Keep a labeled source preview; normal tool images can take precedence.
        p0,p1=tuple(map(int,box[:2])),tuple(map(int,box[2:]))
        cv2.rectangle(bgr,p0,p1,(0,210,255),max(2,w//150))
        small=cv2.resize(bgr,(360,max(1,round(360*h/w))),interpolation=cv2.INTER_AREA)
        ok,encoded=cv2.imencode(".jpg",small,[int(cv2.IMWRITE_JPEG_QUALITY),81])
        if ok:
            self._scout_views[fid]=(score,normalized.tolist(),encoded.tobytes())

    def _build_decider_input(self,observation,local_map=False):
        state,panel=super()._build_decider_input(observation,local_map)
        if self._scout_offline or os.environ.get("NAV_LOCAL_BOXES")!="1" or os.environ.get("NAV_SOURCE_SCOUT","0") not in ("1","category_only") or not self._ev_deciding:
            return state,panel
        goal=str(getattr(observation,"goal_text","") or self._category(observation)).strip()
        self._scout_remember(observation)
        if not goal:
            return state,panel
        if self._scout_spec is None:
            path=Path(os.environ.get("NAV_SCOUT_SELECTOR") or Path(__file__).with_name("selector.json"))
            try:
                self._scout_spec=json.loads(path.read_text(encoding="utf-8"))
            except (OSError,ValueError) as error:
                self._scout_offline=True
                self._note("source_selector_unavailable",None,reason=type(error).__name__)
                return state,panel
        if self._scout_goal!=goal:
            self._scout_checked.clear(); self._scout_done.clear()
            self._scout_views.clear(); self._scout_calls.clear()
            self._scout_last_state=None; self._scout_step=None; self._scout_goal=goal
        step=int(observation.step_count)
        if self._scout_step==step:
            if self._scout_last_state:
                state=dict(state,source_frame_scout=self._scout_last_state)
            return state,panel
        self._scout_step=step
        self._scout_last_state=None
        current,viewable=self._frame_ids_from_feed()
        if not current or not viewable:
            return state,panel
        noun=coarse_category(goal)
        try:
            _,ids=self.client.get_all_poses()
            mapped={int(i) for i in ids if 0<int(i)<=min(current,viewable)}
            unseen=sorted((mapped & self._scout_images.keys()) - self._scout_checked)
            # Sample both backlog and recently mapped views under a fixed detector budget.
            idx=np.linspace(0,len(unseen)-1,min(4,len(unseen))).round().astype(int)
            for j in idx:
                self._scout_observe(unseen[j],noun)
        except (ValueError,TypeError,RuntimeError,OSError,EOFError) as error:
            self._note("scout_temporarily_unavailable",None,reason=type(error).__name__)
            if isinstance(error,(RuntimeError,EOFError)):
                self._scout_offline=True
                return state,panel
        usable=[fid for fid in self._scout_views if fid not in self._scout_done and self._scout_calls.get(fid,0)<2]
        usable.sort(key=lambda f:(self._scout_calls.get(f,0),-self._scout_views[f][0],-f))
        rows=[]
        for i,fid in enumerate(usable[:2]):
            score,box,preview=self._scout_views[fid]
            self._scout_calls[fid]=self._scout_calls.get(fid,0)+1
            rows.append(dict(frame_id=fid,box_xyxy_0_to_1000=box,localization_score=round(score,3)))
            if i==0:
                self._pending_decision_images.append(("scout_frame_"+str(fid),preview))
        if rows:
            self._scout_last_state=dict(category_proposal=noun,views=rows,
                guidance="These are mapped source views with unverified category-level regions. Use view_frame or propose_candidates on the listed frame. Verify the full goal including its attributes and review the resulting native candidate IDs before registration.")
            state=dict(state,source_frame_scout=self._scout_last_state)
            self._note("source_frame_scout",None,mapped_sources=[r["frame_id"] for r in rows])
        return state,panel

    def _tool_propose_candidates(self,frame_id,query=""):
        out=super()._tool_propose_candidates(frame_id,query)
        if (out.get("masks") or out.get("local_pending_proposals")) and str(frame_id).isdigit():
            self._scout_done.add(int(frame_id))
        return out
