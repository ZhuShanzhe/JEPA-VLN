"""Actively acquire an identifiable view for weak, already registered target clues."""
import math
from dataclasses import dataclass, field

import numpy as np

from appearance import _rgb, build_reference
from benchmark_api import Action
from confirm_view import FORWARD, LEFT, RIGHT, TURN, clear_move, pose_after
from identity import relation
from map_views import point_in_box, rgb_aligned
from optics import skew_view
from transfer_agent import ArrivalTransferAgent
from trusted_reference import assess_reference, MIN_DETECTOR_SCORE


@dataclass
class Relook:
    iid: int
    start: int
    pose: tuple
    target: tuple
    route: tuple
    cursor: int = 0
    last: int = -1
    expected: tuple = ()
    captures: dict = field(default_factory=dict)
    checked: set = field(default_factory=set)


def grounded_region(view, rgb, proposals, point, rotation, scale):
    """Require exactly one high-quality detection with depth support at the registered location."""
    h, w = rgb.shape[:2]
    extent = np.array(view['grid'].shape[1::-1]) * view['stride']
    warp = np.tile(extent / [w, h], 2)
    choices = []
    for row in proposals:
        box = np.asarray(row.get('bbox'), dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or float(row.get('score',0)) < MIN_DETECTOR_SCORE:
            continue
        world = point_in_box(view, box*warp)
        if world is None:
            continue
        aligned = rotation @ np.asarray(world) / scale
        error = float(np.linalg.norm(aligned-point)*scale)
        if error <= .48:
            choices.append((error, box, aligned))
    if len(choices) != 1 or choices[0][0] > .32:
        return None
    return choices[0]


class WeakRouteAgent(ArrivalTransferAgent):
    def _init_report_memory(self):
        super()._init_report_memory()
        self._relook = None
        self._relook_history = {}
        self._relook_reply = None
        self._relook_return = {}
        self._duplicate_cooldown = {}
        self._relook_attempt_step = -1
        self._fresh_frame_offer = None
        self._departure_keys = {}

    def _weak_source(self,node):
        with self._reference_purpose(False):
            if self._references(node):
                return None
        # A weak rectangle locates a possible object for the view predictor only.
        for oid in reversed(node.observation_ids):
            if oid in self._views:
                return self._views[oid]
        return None

    def _end_relook(self,reason):
        state=self._relook
        if state is not None:
            self._note("weak_relook_end",state.iid,reason=reason,actions=state.cursor)
        self._relook=None

    def _begin_relook(self,obs,node,pose,scale):
        step=int(obs.step_count)
        source=self._weak_source(node)
        if source is None or self._frontier_grid is None or self.align_R is None:
            return False
        if getattr(self,"_goal_images",None) and self._instance_goal_index.get(node.iid) is None:
            return False
        point=tuple(float(x) for x in node.point[:2])
        distance=math.dist(point,pose[:2])*scale
        if not .10 <= distance <= 1.8:
            return False
        key=(node.iid,tuple(node.observation_ids))
        seen=self._relook_history.setdefault(key,[])
        if len(seen)>=2 or (seen and step-seen[-1][0]<20):
            return False
        budget=min(12,int(getattr(obs,"max_steps",300))-step-2)
        if budget<2:
            return False
        # Exclude the current camera and earlier endpoints so a high-scoring
        # change in yaw alone cannot consume the verification opportunity.
        excluded=[pose[:2],*[end for _,end in seen]]
        fid,ref=source
        ranked=self._rank_probe(fid,ref.roi,pose,point,scale,excluded,budget,uncertain=True)
        if not ranked or not ranked['sequence'] or len(ranked['sequence'])>budget:
            seen.append((step,pose[:2]))
            self._note('weak_relook_evidence',node.iid,reason='no_reachable_ranked_view')
            return False
        route=tuple(ranked['sequence'])
        if FORWARD not in route or any(cmd not in (FORWARD,LEFT,RIGHT) for cmd in route):
            return False
        seen.append((step,ranked['waypoint'][:2]))
        self._relook=Relook(node.iid,step,pose,point,route)
        self._release_pursuit(step,'weak_reference_probe',pause=False)
        if self._confirm_view_state:
            self._end_confirm_view('weak_reference_probe',step)
        self._note('weak_relook_start',node.iid,options=ranked['count'],source=ranked['source'],
                   score=ranked['score'],actions=len(route))
        return True

    def _capture_relook(self,obs,node,pose,scale):
        state=self._relook
        step=int(obs.step_count)
        if (math.dist(pose[:2],state.pose[:2])*scale<.17 or len(state.captures)>=4
                or any(step-t<3 for t in state.captures.values())):
            return
        info=self._last_feed_info or {}
        rows=self._boxes(('live',step),obs.rgb,self._category(obs))
        if not any(float(r.get('score',0))>=MIN_DETECTOR_SCORE for r in rows):
            return
        fid=info.get('frame_id') if info.get('is_keyframe') else None
        if fid is None:
            fid=self._tool_capture_frame().get('frame_id')
        if fid is not None and fid not in state.captures:
            state.captures[int(fid)]=step
            self._fresh_frame_offer=(int(fid),node.iid,step)
            self._note('weak_relook_capture',node.iid,frame=int(fid))

    def _inspect_capture(self,node,fid,snapshot,scale):
        meta,data=self.client.get_frame_image(fid)
        rgb=_rgb(data) if data and meta.get('found') else None
        if rgb is None or not rgb_aligned(snapshot,rgb,8):
            return 'map_rgb_unverified',None
        view=skew_view(snapshot,scale)
        if view is None:
            return 'map_projection_unavailable',None
        rows=self._boxes(('map',fid),rgb,self._category(self._last_observation))
        match=grounded_region(view,rgb,rows,node.point,self.align_R,scale)
        if match is None:
            return 'no_unique_spatial_box',None
        error,box,aligned=match
        center=(box[:2]+box[2:])/2
        evidence=assess_reference(rgb,box,center,rows)
        if not evidence.eligible():
            return 'weak_foreground_'+evidence.reason,None
        owners=[]
        for other in self.instance_store.nodes:
            if other.iid==node.iid:
                continue
            sep=float(np.linalg.norm(np.asarray(other.point)-aligned)*scale)
            if other.reported and sep<=.60:
                if self._arrival_matches(other,rgb,np.asarray([box])):
                    owners.append(other)
                    continue
            if sep <= error+.18:
                return 'competing_spatial_registration',None
        if owners:
            if len(owners)!=1 or relation(self._records(node),self._records(owners[0]),
                    self._instance_goal_index.get(node.iid),self._instance_goal_index.get(owners[0].iid))=='distinct':
                return 'conflicting_instances',None
            return 'duplicate',owners[0]
        # Require aligned depth and an isolated class box. No target IDs or
        # benchmark report locations are available to this decision.
        pixel=(center*1000/np.array(rgb.shape[1::-1])).tolist()
        ref=build_reference(rgb,box,point_1000=pixel)
        if ref is None or any(r.frame_id==fid for r in self._records(node)):
            return 'unusable_or_existing_image',None
        item={'frame_id':fid,'bbox':box.tolist(),'pixel_norm':pixel,'source':'active_reobservation'}
        rec=self.instance_store.new_observation(aligned.tolist(),node.text,evidence=item,
                frame_id=fid,step=self._relook.captures[fid],pixel=pixel,bbox=box.tolist())
        self.instance_store.attach_observation(node,rec)
        self._views[rec.oid]=(fid,ref)
        self._reference_evidence[rec.oid]=evidence
        self._trusted_pixels[(fid,ref.roi)]=(rgb.copy(),tuple(box))
        self._endorsed.add(node.iid)
        self._relook_return[node.iid]=(int(self.now),fid)
        self._note('weak_reference_registered',node.iid,frame=fid,occupancy=evidence.occupancy,
                   score=evidence.detection_score,spatial_error_m=error)
        return 'registered',None

    def _report_gate(self,node,obs):
        # Do not count a frame as independent confirmation of the trusted
        # reference that was just made from that very view.
        entry=self._relook_return.get(node.iid)
        if entry:
            step,fid=entry
            if int(obs.step_count)<=step or ((self._last_feed_info or {}).get('is_keyframe')
                    and self._last_feed_info.get('frame_id')==fid):
                return False,'await_independent_reobservation'
        return super()._report_gate(node,obs)

    def _depart(self,obs,node):
        step=int(obs.step_count)
        previous=self._departure_keys.get(node.iid)
        if (previous and step-previous[0]<=16 and self._explore_follower is not None
                and self.mode=='explore' and self._active_frontier_key==previous[1]):
            action=self._explore_follow(obs)
            if action is not None:
                return action
        if self._confirm_view_state:
            self._end_confirm_view('reported_object_revisited',step)
        self._release_pursuit(step,'reported_object_revisited',pause=False)
        if self._confirm_pending is not None:
            self._release('reported_object_revisited')
        self._clear_current_target()
        self.mode='explore'
        self._plan_exploration(obs,select=False)
        scale=self._metric_scale_value()
        for cluster in self._last_frontier_clusters if scale is not None and scale>0 else ():
            dest=cluster.get('world')
            if dest is None:
                continue
            dist=math.dist(dest[:2],node.point[:2])*scale
            if dist>=1.8 and self._activate_frontier(obs,cluster,source='search away from confirmed report'):
                self._departure_keys[node.iid]=(step,self._active_frontier_key)
                self._note('weak_region_departure',node.iid,distinct_frontier=True,distance_m=float(dist))
                action=self._explore_follow(obs)
                if action is not None:
                    return action
        self._note('weak_region_departure',node.iid,distinct_frontier=False)
        return self._autonomous_explore_action(obs)

    def _step_relook(self,obs):
        state=self._relook
        node=self.instance_store.get(state.iid)
        step=int(obs.step_count)
        pose=self._estimated_current_pose()
        scale=self._metric_scale_value()
        if (node is None or self._known_owner(node) is not None or self._pending_report or
                pose is None or not scale or not math.isfinite(scale) or self._frontier_grid is None):
            self._end_relook('unavailable_or_already_resolved')
            return None
        if step-state.start>len(state.route)+5 or math.dist(node.point[:2],state.target)*scale>.32:
            self._end_relook('expired_or_target_changed')
            return None
        if state.expected and (step!=state.last+1 or obs.previous_action!=state.route[state.cursor-1]
                or math.dist(state.expected[:2],pose[:2])*scale>.25
                or abs(math.remainder(state.expected[2]-pose[2],math.tau))>TURN*.65
                or (obs.previous_action==FORWARD and getattr(self,'_last_motion_failed',False))):
            self._end_relook('motion_interrupted')
            return None
        self._capture_relook(obs,node,pose,scale)
        overdue=[fid for fid in state.captures if fid not in state.checked]
        if overdue:
            snap=self._map_snapshot()
            for fid in overdue:
                if fid not in snap:
                    continue
                state.checked.add(fid)
                result,owner=self._inspect_capture(node,fid,snap[fid],scale)
                self._note('weak_relook_evidence',node.iid,frame=fid,reason=result)
                if result=='registered':
                    self._end_relook('trusted_reference_acquired')
                    return None
                if owner is not None:
                    self._duplicate_cooldown[node.iid]=(tuple(node.observation_ids),step+32)
                    self._note('weak_duplicate_confirmed',node.iid,owner=owner.iid)
                    self._end_relook('reported_instance')
                    return self._depart(obs,node)
        if state.cursor==len(state.route):
            if overdue and step<state.start+len(state.route)+3:
                state.expected=()
                return LEFT
            self._end_relook('no_mapped_identifiable_view')
            return None
        cmd=state.route[state.cursor]
        nxt=pose_after(pose,cmd,scale)
        if cmd==FORWARD and not clear_move(self._frontier_grid,pose,nxt):
            self._end_relook('updated_map_blocked')
            return None
        state.cursor+=1
        state.expected,state.last=nxt,step
        self._view_last_reply=(step,node.iid,(cmd,'confirm_view_motion'))
        self._note('weak_relook_action',node.iid,action=cmd,index=state.cursor)
        return cmd

    def _cooling(self,node,step):
        record=self._duplicate_cooldown.get(node.iid)
        return record is not None and record[0]==tuple(node.observation_ids) and step<record[1]

    def _try_relook(self,obs,requested=None):
        step=int(obs.step_count)
        if self._relook_reply and self._relook_reply[0]==step:
            return self._relook_reply[1]
        quota=getattr(obs,'target_count',0)
        if self._pending_report or self._routing_native_goto or (quota and self._reported_count>=quota):
            return None
        if self._relook is None:
            if self._relook_attempt_step==step or self._pursuit or self._native_leg is not None:
                return None
            self._collect_views()
            pose,scale=self._estimated_current_pose(),self._metric_scale_value()
            if pose is None or not scale or scale<=0 or not np.isfinite((*pose,scale)).all():
                return None
            choices=[requested] if requested is not None else sorted(self.instance_store.available(),
                    key=lambda n:math.dist(pose[:2],n.point[:2]))
            for node in choices:
                if not self._cooling(node,step) and self._known_owner(node) is None and self._begin_relook(
                        obs,node,tuple(pose),scale):
                    break
            else:
                return None
            self._relook_attempt_step=step
        action=self._step_relook(obs)
        self._relook_reply=(step,action)
        return action

    def _pursue(self,obs,requested=None):
        action=self._try_relook(obs,requested)
        return action if action is not None else super()._pursue(obs,requested)

    def _execute_decider_result(self,obs,result,event,images=None):
        if result.action in {'GOTO_INSTANCE','REPORT_FOUND'} and self._pending_report is None:
            node=self.instance_store.get(result.target_id)
            if node is not None:
                if self._known_owner(node) is not None or self._cooling(node,int(obs.step_count)):
                    if self._relook:
                        self._end_relook('new_decision')
                    return self._depart(obs,node)
                if self._relook and self._relook.iid!=node.iid:
                    self._end_relook('different_requested_instance')
                if self._relook is not None or self._weak_source(node):
                    action=self._try_relook(obs,node)
                    if action is not None:
                        return action
        return super()._execute_decider_result(obs,result,event,images)

    def _build_decider_input(self,observation,local_map=False):
        state,image=super()._build_decider_input(observation,local_map)
        offer=self._fresh_frame_offer
        if offer and 0<=int(observation.step_count)-offer[2]<=15:
            state=dict(state,weak_clue_reobservation={'frame_id':offer[0],'instance':offer[1],
                  'guidance':'Inspect this closer frame and localize the object before reporting. If it is a previously reported instance, explore a different frontier.'})
        return state,image
