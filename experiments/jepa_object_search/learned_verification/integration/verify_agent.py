"""Treat a better camera observation as an action; keep instance reporting independent."""
import math
import os
from pathlib import Path

import cv2
import numpy as np

from agents import navigator
from appearance import _rgb, build_reference, match_reference, WORK_WIDTH
from confirm_view import MAX_ATTEMPTS, MAX_ACTIONS, FORWARD, clear_move, pose_after
from learned_map_views import map_views, relative_action
from report_evidence import current_support, eligible_boxes


class VerifyActionMixin:
    def _init_report_memory(self):
        super()._init_report_memory()
        self._probe_context={}
        self._weak_candidate=None
        self._weak_waiting=[]
        self._weak_handled=set()
        self._recent_inspection=None
        self._verification_scorer=None
        self._uncertain_scorer=None
        self._blocked_probe_edges=[]
        self._probe_retry=None

    def _verification_model(self,uncertain=False):
        cache="_uncertain_scorer" if uncertain else "_verification_scorer"
        model=getattr(self,cache,None)
        if model is None:
            variant=os.environ.get("JEPA_UNCERTAIN_HEAD","local") if uncertain else os.environ.get("JEPA_VIEW_HEAD","local")
            if variant=="context":
                from weak_view_score import WeakViewScorer as Scorer
                folder="active_weak"
            elif variant=="local":
                from active_view_score import ActiveViewScorer as Scorer
                folder="active"
            else:
                raise ValueError("Verification head must be local or context")
            default=Path(__file__).resolve().parents[1]/folder/"weights/jepa.npz"
            checkpoint="JEPA_UNCERTAIN_CHECKPOINT" if uncertain else "JEPA_VIEW_CHECKPOINT"
            model=Scorer(os.environ.get(checkpoint,default))
            setattr(self,cache,model)
        return model

    def _source(self, fid, roi):
        key=(int(fid),tuple(roi))
        if key not in self._probe_context:
            meta,jpeg=self.client.get_frame_image(int(fid))
            if not meta.get("found") or not jpeg:
                return None
            rgb=_rgb(jpeg)
            if rgb is None:
                return None
            h,w=rgb.shape[:2]
            box=tuple(np.array(roi)*[w,h,w,h])
            if not np.all(np.isfinite(box)) or min(box[2]-box[0],box[3]-box[1])<2:
                return None
            if len(self._probe_context)>=8:
                del self._probe_context[next(iter(self._probe_context))]
            self._probe_context[key]=(rgb,box)
        return self._probe_context[key]

    def _rank_probe(self,fid,roi,pose,point,scale,excluded,budget=MAX_ACTIONS,uncertain=False):
        if self.align_R is None or self._frontier_grid is None or not scale or scale<=0:
            return None
        matrix=self.client.get_frame_pose(int(fid))
        if matrix is None:
            return None
        anchor=tuple(navigator.pose_to_yaw_2d(matrix,self.align_R))
        self._blocked_probe_edges=[(t,p) for t,p in self._blocked_probe_edges if self.now-t<=24]
        options=map_views(self._frontier_grid,pose,point,anchor,scale,excluded,self.report_near_dist_m,
                          [p for t,p in self._blocked_probe_edges],budget)
        source=self._source(fid,roi) if options else None
        if source is None:
            return None
        offsets=[relative_action(anchor,q,scale) for q,_,_ in options]
        scores=np.asarray(self._verification_model(uncertain).score(*source,offsets),dtype=float).reshape(-1)
        if len(scores)!=len(options) or not np.isfinite(scores).all():
            raise ValueError("Inspection predictor returned invalid candidate scores")
        # Prefer the shorter route on a score tie. Every planned primitive debits
        # the existing episode budget, including reorientation.
        index=max(range(len(options)),key=lambda i:(scores[i],-len(options[i][1])))
        waypoint,commands,source_type=options[index]
        return dict(waypoint=waypoint,sequence=commands,source=source_type,
                    source_displacement_m=math.hypot(*offsets[index][:2]),
                    count=len(options),score=float(scores[index]),anchor=anchor)

    def _begin_confirm_view(self,node,obs,pose,scale):
        step=int(obs.step_count)
        hist=self._view_history.setdefault(node.iid,dict(attempts=0,end=-100,excluded=[]))
        retry=self._probe_retry if self._probe_retry and self._probe_retry[:2]==(node.iid,step) else None
        if hist["attempts"]>=MAX_ATTEMPTS or (not retry and step-hist["end"]<12):
            return False
        refs=self._references(node)
        if not refs or not scale or math.dist(pose[:2],node.point[:2])*scale>1.8:
            return False
        # A registered bounded object patch carries the source location and extent.
        fid,ref=refs[-1]
        budget=retry[2] if retry else MAX_ACTIONS
        result=self._rank_probe(fid,ref.roi,pose,tuple(node.point[:2]),scale,hist["excluded"],budget)
        if result is None:
            return False if retry or self._blocked_probe_edges else super()._begin_confirm_view(node,obs,pose,scale)
        hist["attempts"]+=1
        self._confirm_view_state=dict(node_id=node.iid,waypoint=result["waypoint"],fid=int(fid),
            target=tuple(map(float,node.point[:2])),start=step,sequence=result["sequence"],
            cursor=0,last_step=step-1,expected=None,learned=True,budget=budget)
        self._note("confirm_view_start",node.iid,frame=int(fid),view_source="learned_"+result["source"],
                   options=result["count"],predicted_score=result["score"],attempt=hist["attempts"],
                   source_displacement_m=result["source_displacement_m"],
                   planned_actions=list(result["sequence"]),waypoint=list(result["waypoint"]))
        return True

    def _end_confirm_view(self,reason,step):
        state=self._confirm_view_state
        super()._end_confirm_view(reason,step)
        if not state or not state.get("learned"):
            return
        if reason=="forward_stalled":
            pose=self._estimated_current_pose()
            if pose is not None:
                self._blocked_probe_edges.append((step,tuple(pose)))
                # This approach failed; the camera has not inspected its destination.
                self._view_history[state["node_id"]]["excluded"].pop()
        remaining=state["budget"]-state["cursor"]
        if remaining>0 and reason in ("forward_stalled","route_blocked_on_updated_map","need_metric_near_view",
                                      "no_local_target_box","no_current_instance_match","ambiguous_live_boxes"):
            self._probe_retry=(state["node_id"],step,remaining)

    def _confirm_view_step(self,node,obs):
        state=self._confirm_view_state
        if (state and state.get("learned") and state["node_id"]==node.iid
                and 0<state["cursor"]<len(state["sequence"]) and not self._pending_report
                and self._known_owner(node) is None and int(obs.step_count)==state["last_step"]+1
                and obs.previous_action==state["sequence"][state["cursor"]-1]
                and not (obs.previous_action==FORWARD and getattr(self,"_last_motion_failed",False))):
            ok,reason=self._report_gate(node,obs)
            if ok:
                self._note("confirm_view_recheck",node.iid,frame=state["fid"],live_step=int(obs.step_count),
                           accepted=True,reason=reason,intermediate=True)
                self._end_confirm_view("observed_before_waypoint",int(obs.step_count))
                reply=self._stage_report(node,obs,reason),"confirm_view_report"
                self._view_last_reply=int(obs.step_count),node.iid,reply
                return reply
        result=super()._confirm_view_step(node,obs)
        if result is None and self._probe_retry and self._probe_retry[:2]==(node.iid,int(obs.step_count)):
            result=super()._confirm_view_step(node,obs)
            self._probe_retry=None
        return result

    def _tool_commit_candidates(self,reviews,label=""):
        out=super()._tool_commit_candidates(reviews,label)
        # The native VLM judged the selected mask uncertain. It remains outside
        # the registry throughout this probe, including a successful arrival.
        for cid in out.get("uncertain",[]):
            cid=str(cid)
            p=self._proposal_queue.get(cid,{})
            if (cid not in self._weak_handled and p.get("status")=="uncertain"
                    and p.get("bbox") and p.get("frame_id") is not None):
                self._weak_handled.add(cid)
                self._weak_waiting.append(dict(cid=cid,fid=p["frame_id"],box=p["bbox"],step=self.now))
        self._next_weak()
        return out

    def _next_weak(self):
        if self._weak_candidate is not None:
            return
        while self._weak_waiting:
            item=self._weak_waiting.pop(0)
            if (self.now-item["step"]<=24 and
                    self._proposal_queue.get(item["cid"],{}).get("status")=="uncertain"):
                self._weak_candidate=item
                return
            self._note("uncertain_view_end",0,reason="expired_or_reviewed",candidate=item["cid"])

    def _finish_weak(self,obs,reason,capture=False):
        state=self._weak_candidate
        self._weak_candidate=None
        self._note("uncertain_view_end",0,reason=reason,candidate=state["cid"])
        if capture:
            exact=(self._last_feed_info or {})
            fid=exact.get("frame_id") if exact.get("is_keyframe") else None
            if fid is None:
                fid=self._tool_capture_frame().get("frame_id")
            rgb=np.asarray(obs.rgb)[...,:3]
            width=min(640,rgb.shape[1])
            small=cv2.resize(rgb,(width,max(1,round(rgb.shape[0]*width/rgb.shape[1]))),interpolation=cv2.INTER_AREA)
            ok,b=cv2.imencode(".jpg",cv2.cvtColor(small,cv2.COLOR_RGB2BGR))
            if ok:
                self._recent_inspection=dict(candidate=state["cid"],step=int(obs.step_count),
                    jpeg=b.tobytes(),frame_id=fid,offer_steps=set())

    def _weak_action(self,obs):
        self._next_weak()
        s=self._weak_candidate
        if s is None:
            return None
        step=int(obs.step_count)
        if (step-s["step"]>24 or self._proposal_queue.get(s["cid"],{}).get("status")!="uncertain"):
            self._finish_weak(obs,"expired_or_reviewed")
            return None
        pose,scale=self._estimated_current_pose(),self._metric_scale_value()
        if pose is None or not scale or scale<=0 or self.align_R is None or self._frontier_grid is None:
            return None
        if "sequence" not in s:
            raw=self.client.resolve_candidates([s["cid"]])
            row=raw.get("candidates",raw).get(s["cid"],{})
            source,jpg=self.client.get_frame_image(int(s["fid"]))
            img=_rgb(jpg) if source.get("found") and jpg else None
            if not row.get("found") or row.get("point") is None or img is None:
                self._finish_weak(obs,"geometry_or_source_unavailable")
                return None
            point=tuple(self._aligned_point(row["point"])[:2])
            if math.dist(point,pose[:2])*scale>1.8:
                self._finish_weak(obs,"outside_local_map_range")
                return None
            h,w=img.shape[:2]
            roi=np.asarray(s["box"],dtype=float)/[w,h,w,h]
            s.update(roi=roi,point=point,budget=MAX_ACTIONS,attempts=1,reference=build_reference(img,s["box"]))
            route=self._rank_probe(s["fid"],roi,pose,point,scale,[],s["budget"],uncertain=True)
            if route is None:
                self._finish_weak(obs,"no_route")
                return None
            s.update(sequence=route["sequence"],cursor=0,last=step-1)
            self._note("uncertain_view_start",0,candidate=s["cid"],options=route["count"],
                       predicted_score=route["score"],actions=len(s["sequence"]))
        if s["last"]==step:
            return s["sent"]
        if s["cursor"]:
            if step!=s["last"]+1 or obs.previous_action!=s["sent"]:
                self._finish_weak(obs,"interrupted_or_pose_changed")
                return None
            if s["sent"]==FORWARD and getattr(self,"_last_motion_failed",False):
                return self._retry_weak(obs,pose,scale)
            if (math.dist(pose[:2],s["expected"][:2])*scale>.22
                    or abs(math.remainder(pose[2]-s["expected"][2],2*math.pi))>math.radians(19)):
                self._finish_weak(obs,"interrupted_or_pose_changed")
                return None
        if s["cursor"]==len(s["sequence"]):
            self._finish_weak(obs,"fresh_view",True)
            return None
        if s["cursor"] and self._clearer_weak_view(s,obs):
            self._finish_weak(obs,"clearer_live_region",True)
            return None
        action=s["sequence"][s["cursor"]]
        nxt=pose_after(pose,action,scale)
        if action==FORWARD and not clear_move(self._frontier_grid,pose,nxt):
            return self._retry_weak(obs,pose,scale)
        s.update(cursor=s["cursor"]+1,last=step,sent=action,expected=nxt)
        # The inherited action receipt keeps the ordinary exploration follower
        # synchronized when our movement replaces one of its primitives.
        self._view_last_reply=step,("uncertain",s["cid"]),(action,"confirm_view_motion")
        return action

    def _clearer_weak_view(self,state,obs):
        # A larger, independently re-detected matching patch can trigger an
        # early visual review. The reporter still needs the native verdict.
        ref=state["reference"]
        if ref is None:
            return False
        image=np.asarray(obs.rgb)
        rows=self._boxes(("live",int(obs.step_count)),image,self._category(obs))
        boxes=eligible_boxes(rows,image.shape)
        if not boxes:
            return False
        box,_=current_support(ref,match_reference(ref,image),image,boxes)
        if box is None:
            return False
        x,y,u,v=state["roi"]
        live_area=(box[2]-box[0])*(box[3]-box[1])
        return live_area > (u-x)*(v-y)*WORK_WIDTH**2*image.shape[0]/image.shape[1]

    def _retry_weak(self,obs,pose,scale):
        s=self._weak_candidate
        remaining=s["budget"]-s["cursor"]
        self._blocked_probe_edges.append((int(obs.step_count),tuple(pose)))
        if remaining>0 and s["attempts"]<MAX_ATTEMPTS:
            route=self._rank_probe(s["fid"],s["roi"],pose,s["point"],scale,[],remaining,uncertain=True)
            if route is not None:
                s.update(sequence=route["sequence"],cursor=0,last=int(obs.step_count)-1,
                         budget=remaining,attempts=s["attempts"]+1)
                self._note("uncertain_view_replan",0,candidate=s["cid"],remaining=remaining)
                return self._weak_action(obs)
        self._finish_weak(obs,"blocked_or_no_alternate")
        return None

    def _auto_report(self,obs):
        issued=super()._auto_report(obs)
        if issued is not None or self._pending_report or self._confirm_view_state:
            return issued
        return self._weak_action(obs)

    def _build_decider_input(self,obs,local_map=False):
        state,picture=super()._build_decider_input(obs,local_map)
        rec=self._recent_inspection
        step=int(obs.step_count)
        if (rec and self._ev_deciding and 0<=step-rec["step"]<=24
                and (step in rec["offer_steps"] or len(rec["offer_steps"])<2)):
            label="inspection_"+rec["candidate"]
            rec["offer_steps"].add(step)
            state["verification_observation"]=dict(original_candidate=rec["candidate"],captured_step=rec["step"],
                attachment=label,new_frame_id=rec["frame_id"],guidance=(
                    "Inspect this fresh image. Use propose_candidates on its new_frame_id and review a NEW candidate; "
                    "the original candidate remains uncertain." if rec["frame_id"] is not None else
                    "Only the image is available. Capture a new frame if another view supports the target."))
            if not any(name==label for name,_ in self._pending_decision_images):
                self._pending_decision_images.insert(0,(label,rec["jpeg"]))
        return state,picture
