"""MOST entry point: return to corroborated observation viewpoints before resuming frontier search."""
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

PROJECT=Path(os.environ.get("NAV_ROOT",Path(__file__).resolve().parents[2]))
sys.path[:0]=[str(PROJECT/"jepa_variable_v16/adapter")]
from agents import navigator
from benchmark_api import Action
from variable_agent import VariableFrontier
from view_explore import ReachableViewMixin
from view_memory import evidence_patch,live_match


class ConfirmFrontier(ReachableViewMixin,VariableFrontier):
    _confirm_pending=None
    def init_episode(self,observation):
        self._views={}
        self._retry_after={}
        self._endorsed=set()
        self._attempts={}
        self._confirm_pending=None
        self._confirm_frame=None
        self._confirm_orient=0
        super().init_episode(observation)

    def _note(self,event,instance=None,**values):
        with (Path(self.output_dir)/"confirm_events.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(dict(step=self.now,event=event,instance=instance,**values),allow_nan=False)+"\n")

    def _collect_views(self):
        for node in self.instance_store.available():
            for oid in node.observation_ids:
                if oid in self._views or self.now < self._retry_after.get(oid,-1):
                    continue
                record=self.instance_store.get_observation(oid)
                if record is None or record.frame_id is None:
                    continue
                point=record.evidence.get("pixel_norm")
                if point is None:
                    continue
                fid=int(record.frame_id)
                pose=self.client.get_frame_pose(fid)
                if pose is None:
                    self._retry_after[oid]=self.now+4
                    continue
                meta,jpeg=self.client.get_frame_image(fid)
                patch=evidence_patch(jpeg,point) if meta.get("found") and jpeg else None
                if patch is None:
                    self._retry_after[oid]=self.now+32
                    continue
                self._views[oid]=(fid,patch)
                # Only the native VLM's accepted registrations reach the canonical
                # store. Pending duplicates and rejected proposals lack an owner here.
                self._endorsed.add(node.iid)
                self._note("view_saved",node.iid,frame=fid,observation=int(oid),
                           center_1000=[float(v) for v in point])

    def _references(self,node):
        return [self._views[i] for i in node.observation_ids if i in self._views]

    def _geometry(self,node,fid):
        scale=self._metric_scale_value()
        if not scale or self.align_R is None:
            return None
        camera=self.client.get_frame_pose(fid)
        here=self._estimated_current_pose()
        if camera is None or here is None:
            return None
        x,y,yaw=navigator.pose_to_yaw_2d(camera,self.align_R)
        values=(math.dist((x,y),here[:2])*scale,math.dist((x,y),node.point[:2])*scale,
                math.dist(here[:2],node.point[:2])*scale,math.remainder(yaw-here[2],2*math.pi))
        return values if all(map(math.isfinite,values)) else None

    def _release(self,reason):
        if self._confirm_pending is not None:
            self._note("return_cancelled",self._confirm_pending,reason=reason)
        self._confirm_pending=None
        self._confirm_frame=None
        self._confirm_orient=0

    def _return(self,observation,nodes):
        options=[]
        for node in nodes:
            if node is None or node.reported or node.iid not in self._endorsed or node.iid in self._unreachable_instance_ids:
                continue
            for fid,_ in self._references(node):
                count,last=self._attempts.get((node.iid,fid),(0,-100))
                if count>=2 or self.now-last<24:
                    continue
                geo=self._geometry(node,fid)
                if geo is not None and geo[1]<=2.25:
                    options.append((geo[1]+.22*geo[0],node.iid,fid))
        for _,iid,fid in sorted(options):
            count,_=self._attempts.get((iid,fid),(0,0))
            self._attempts[iid,fid]=(count+1,self.now)
            response=self._schedule_revisit(fid)
            if not response.get("navigating"):
                self._note("return_unavailable",iid,frame=fid,reason="native_map_or_route_unavailable")
                continue
            self._close_commit("return_to_instance")
            self._end_scan("return_to_instance")
            self._end_route("return_to_instance")
            self._cancel_delay("return_to_instance")
            self._confirm_pending=iid
            self._confirm_frame=fid
            self._confirm_orient=0
            self.target_instance_id=iid
            self._selected_evidence=self._tool_view_instance(iid)
            self._note("return_started",iid,frame=fid)
            action,arrived,stuck=self._nav_action(observation)
            if arrived:
                return self._handle_arrival_transition(observation)
            if action is not None and not stuck:
                return action
            self._release("no_executable_action")
            self._clear_current_target()
            self.mode="explore"
        return None

    def _execute_decider_result(self,observation,result,event,images=None):
        self._collect_views()
        node=self.instance_store.get(result.target_id) if result.action in ("GOTO_INSTANCE","REPORT_FOUND") else None
        if node is not None and not node.reported:
            self._endorsed.add(node.iid)
        if self._confirm_pending is not None and result.action in ("FINISH","EXPLORE","GOTO_INSTANCE","GOTO_FRONTIER"):
            self._release("native_decision_"+result.action)
        if event not in ("arrival","adjustment","nav_failed","finish_check") and result.action in ("EXPLORE","GOTO_FRONTIER","GOTO_INSTANCE"):
            candidates=[node] if result.action=="GOTO_INSTANCE" else self.instance_store.available()
            action=self._return(observation,candidates)
            if action is not None:
                return action
        return super()._execute_decider_result(observation,result,event,images)

    def _plan_to_target(self,observation):
        if self._confirm_pending is not None and self.target_instance_id==self._confirm_pending:
            camera=self.client.get_frame_pose(self._confirm_frame)
            if camera is None:
                self._release("keyframe_unavailable")
                return False
            self.target_candidate_id=None
            self.target_point=np.asarray(camera[:3,3],dtype=float)
        return super()._plan_to_target(observation)

    def _nav_action(self,observation):
        if self._confirm_pending is None or self.target_instance_id!=self._confirm_pending or self.follower is None:
            return super()._nav_action(observation)
        # The saved target is a camera location in previously occupied free space.
        old,follower=self.reach_m,self.follower
        old_follower=follower.reach_m
        try:
            self.reach_m=follower.reach_m=min(old,.32)
            return super()._nav_action(observation)
        finally:
            self.reach_m=old
            follower.reach_m=old_follower

    def _nav_failed_recovery(self,observation):
        self._release("native_navigation_failed")
        return super()._nav_failed_recovery(observation)

    def _handle_arrival_transition(self,observation):
        iid=self._confirm_pending
        if iid is not None:
            node=self.instance_store.get(iid)
            geo=self._geometry(node,self._confirm_frame) if node is not None and not node.reported else None
            if geo is None:
                self._release("lost_identity_or_pose")
            elif geo[0]>.85:
                self._release("arrival_viewpoint_mismatch")
            elif abs(geo[3])>math.radians(19) and self._confirm_orient<6:
                self._confirm_orient+=1
                return int(Action.TURN_LEFT if geo[3]>0 else Action.TURN_RIGHT)
            else:
                self._note("return_arrived",iid,frame=self._confirm_frame,view_distance_m=round(geo[0],3))
        return super()._handle_arrival_transition(observation)

    def _build_decider_input(self,observation,local_map=False):
        state,picture=super()._build_decider_input(observation,local_map)
        if self._confirm_pending is not None:
            state=dict(state)
            state["viewpoint_revisit"]={"instance_id":self._confirm_pending,"evidence_frame":self._confirm_frame,
                "instruction":"Compare the named instance and its attributes in the current image with the saved evidence. Report only when the same target is visible and close; otherwise adjust or explore."}
        return state,picture

    def _report_found(self,instance_id=None):
        node=self.instance_store.get(instance_id)
        if node is None or node.reported:
            self._note("already_reported" if node is not None else "unknown_instance",instance_id)
            return super()._report_found(instance_id)
        refs=self._references(node)
        obs=self._last_observation
        best=0.
        for fid,template in refs:
            ok,score,gap=live_match(template,obs.rgb)
            best=max(best,score)
            geo=self._geometry(node,fid)
            if ok and geo is not None and (geo[2]<=self.report_near_dist_m or (geo[0]<=.65 and geo[1]<=2.0)):
                self._note("reidentified",node.iid,frame=fid,score=score,gap=gap,view_distance_m=round(geo[0],3))
                action=super()._report_found(instance_id)
                if action==int(Action.TARGET_FOUND):
                    self._note("report_issued",node.iid,frame=fid,mode="current_rgb_and_native_vlm")
                    self._release("reported")
                return action
        if not refs:
            # A native VLM decision with no registered appearance template follows its original interface.
            self._note("native_report_without_template",node.iid)
            return super()._report_found(instance_id)
        self._note("report_deferred",node.iid,score=best,reason="no_close_current_view_match")
        self._stamp_pending_outcome("blocked:need_current_view","Revisit or adjust to a current, close view matching the registered evidence.")
        if self._confirm_pending is None and not self._arrival_transition_active:
            action=self._return(obs,[node])
            if action is not None:
                return action
        self._release("report_unconfirmed")
        self._clear_current_target()
        self.mode="explore"
        return int(Action.TURN_LEFT)

    def _clear_current_target(self):
        self._release("native_clear")
        return super()._clear_current_target()
