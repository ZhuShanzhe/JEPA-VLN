"""Seek a view that separates possible owners of a geometric place in the episode map."""
import math

import numpy as np

from agents import navigator
from appearance import _rgb, build_reference
from confirm_view import FORWARD, MAX_ACTIONS
from delayed_relook_agent import DelayedRelookAgent
from identity import relation
from learned_map_views import map_views, relative_action
from map_views import rgb_aligned
from optics import skew_view
from spatial_identity import associate, make_anchor
from trusted_reference import assess_reference
from visual_witness import compare_regions, describe, strong_object
from weak_route_agent import WeakRouteAgent, grounded_region


def target_relative_motion(anchor, waypoint, point):
    """Clockwise azimuth around the object (the navigator uses positive left yaw)."""
    source=np.asarray(anchor[:2],float)-point
    destination=np.asarray(waypoint[:2],float)-point
    r0,r1=np.linalg.norm(source),np.linalg.norm(destination)
    if min(r0,r1)<1.e-6:
        return None
    angle=math.remainder(math.atan2(source[1],source[0])-math.atan2(destination[1],destination[0]),math.tau)
    return angle,r1/r0


class _SpatialCapture(WeakRouteAgent):
    """A cooperative base: immediate AND deferred captures enter the same classifier."""
    def _init_report_memory(self):
        super()._init_report_memory()
        self._native_captures={}
        self._spatial_hold={}
        self._spatial_alias={}
        self._needs_view={}
        self._place_checked={}
        self._spatial_probe=None
        self._shape_view_model=None

    def _capture_relook(self,obs,node,pose,scale):
        super()._capture_relook(obs,node,pose,scale)
        for fid,step in self._relook.captures.items():
            if step==obs.step_count and fid not in self._native_captures:
                rgb=_rgb(obs.rgb)
                if rgb is not None:
                    self._native_captures[fid]=(rgb.copy(),self._boxes(('live',step),rgb,self._category(obs)))
        # Retain only pending views and the active two short maneuvers.
        live=set(self._waiting_for_map)|set(self._relook.captures)
        self._native_captures={fid:v for fid,v in self._native_captures.items() if fid in live}

    def _known_owner(self,node):
        native=super()._known_owner(node)
        if native is not None:
            return native
        item=self._spatial_alias.get(node.iid)
        if item and tuple(node.observation_ids)==item[0]:
            owner=self.instance_store.get(item[1])
            if owner and owner.reported and tuple(owner.observation_ids)==item[2]:
                return owner
        return None

    def _memory_anchors(self,node,query_rgb,box,query,scale):
        snapshots=self._map_snapshot()
        memories,looks,excluded={},{},set()
        query_descriptor=describe(query_rgb,box)
        for other in self.instance_store.nodes:
            if other.iid==node.iid:
                continue
            records=relation(self._records(node),self._records(other),self._instance_goal_index.get(node.iid),
                             self._instance_goal_index.get(other.iid))
            if records=='distinct':
                excluded.add(other.iid)
            # A registry centroid is only a broad candidate selector. A close
            # record without a calibrated reference remains an unresolved rival.
            if np.linalg.norm(self.align_R@query.center-np.asarray(other.point)*scale)>.95:
                continue
            refs=memories[other.iid]=[]
            with self._reference_purpose(False):
                trusted=self._references(other)
            for fid,ref in trusted:
                source=self._trusted_pixels.get((fid,tuple(ref.roi)))
                snap=snapshots.get(fid)
                if fid==query.frame_id or source is None or snap is None or not rgb_aligned(snap,source[0],8):
                    continue
                view=skew_view(snap,scale)
                anchor=make_anchor(view,*source,frame_id=fid) if view is not None else None
                if anchor is not None:
                    refs.append(anchor)
                    if strong_object(compare_regions(describe(*source),query_descriptor)):
                        looks[other.iid]='strong'
        return memories,looks,excluded

    def _inspect_capture(self,node,fid,snapshot,scale):
        info,data=self.client.get_frame_image(fid)
        mapped=_rgb(data) if data and info.get('found') else None
        if mapped is None or not rgb_aligned(snapshot,mapped,8):
            return 'map_rgb_unverified',None
        view=skew_view(snapshot,scale)
        if view is None:
            return 'map_projection_unavailable',None
        native=self._native_captures.get(fid)
        if native is not None and rgb_aligned(snapshot,native[0],8):
            rgb,rows=native
        else:
            rgb=mapped
            rows=self._boxes(('map',fid),rgb,self._category(self._last_observation))
        hit=grounded_region(view,rgb,rows,node.point,self.align_R,scale)
        if hit is None:
            return 'no_unique_spatial_box',None
        _,box,point=hit
        query=make_anchor(view,rgb,box,frame_id=fid)
        if query is None:
            return 'unreliable_core_depth',None
        memories,looks,excluded=self._memory_anchors(node,rgb,box,query,scale)
        if memories:
            verdict=associate(query,memories,appearances=looks,excluded=excluded)
            self._note('mapped_identity_decision',node.iid,frame=int(fid),status=verdict.status,reason=verdict.reason,
                       candidates=len(memories),supported_owners=sum(bool(r) for r in memories.values()),
                       used_native_rgb=rgb is not mapped)
            if verdict.status=='same':
                owner=self.instance_store.get(verdict.owner)
                if owner is not None and owner.reported:
                    self._spatial_alias[node.iid]=(tuple(node.observation_ids),owner.iid,tuple(owner.observation_ids))
                    self._spatial_hold.pop(node.iid,None)
                    return 'duplicate',owner
                # A separate UNREPORTED record never suppresses an eligible target.
                return 'unreported_owner_still_under_review',None
            if verdict.status=='uncertain':
                self._spatial_hold[node.iid]=tuple(node.observation_ids)
                self._needs_view[node.iid]=tuple(node.observation_ids)
                return verdict.reason,None
        self._spatial_hold.pop(node.iid,None)
        # Distinct/isolated only creates a reference, never a report action.
        center=(box[:2]+box[2:])/2
        quality=assess_reference(rgb,box,center,rows)
        if not quality.eligible():
            return 'weak_foreground_'+quality.reason,None
        click=center*1000/np.array(rgb.shape[1::-1])
        ref=build_reference(rgb,box,point_1000=click)
        if ref is None or any(r.frame_id==fid for r in self._records(node)):
            return 'unusable_or_existing_image',None
        # Stored observation boxes use the mapper image grid, even when the
        # visual verifier keeps the original-resolution crop separately.
        map_box=(np.array(ref.roi)*np.tile(mapped.shape[1::-1],2)).tolist()
        item=dict(frame_id=fid,bbox=map_box,pixel_norm=click.tolist(),source='place_checked_reobservation')
        rec=self.instance_store.new_observation(point.tolist(),node.text,evidence=item,frame_id=fid,
                step=self._relook.captures[fid],pixel=click.tolist(),bbox=map_box)
        self.instance_store.attach_observation(node,rec)
        self._place_checked[node.iid]=tuple(node.observation_ids)
        self._needs_view.pop(node.iid,None)
        self._views[rec.oid]=(fid,ref)
        self._reference_evidence[rec.oid]=quality
        self._trusted_pixels[(fid,ref.roi)]=(rgb.copy(),tuple(box))
        self._endorsed.add(node.iid)
        self._relook_return[node.iid]=(int(self.now),fid)
        self._note('weak_reference_registered',node.iid,frame=fid,occupancy=quality.occupancy,score=quality.detection_score)
        return 'registered',None

    def _weak_source(self,node):
        signature=tuple(node.observation_ids)
        request=self._needs_view.get(node.iid)==signature or self._spatial_hold.get(node.iid)==signature
        if not request and self._place_checked.get(node.iid)!=signature and not node.reported:
            scale=self._metric_scale_value()
            if scale and np.isfinite(scale) and scale>0:
                request=any(other.reported and other.iid!=node.iid
                    and np.linalg.norm(np.asarray(node.point)-other.point)*scale<=.95
                    and relation(self._records(node),self._records(other),self._instance_goal_index.get(node.iid),
                                 self._instance_goal_index.get(other.iid))=='unknown' for other in self.instance_store.nodes)
        if request:
            for oid in reversed(node.observation_ids):
                if oid in self._views:
                    self._needs_view[node.iid]=signature
                    return self._views[oid]
        return super()._weak_source(node)

    def _report_gate(self,node,obs):
        if self._spatial_hold.get(node.iid)==tuple(node.observation_ids):
            return False,'need_independent_spatial_confirmation'
        if self._known_owner(node) is not None:
            return False,'already_reported_identity'
        ok,reason=super()._report_gate(node,obs)
        if not ok and reason in {'arrival_no_unique_correspondence','arrival_other_identity_matches',
                'other_instance_explains_same_live_region','no_current_instance_match','ambiguous_live_boxes'}:
            self._needs_view[node.iid]=tuple(node.observation_ids)
        return ok,reason

    def _begin_relook(self,obs,node,pose,scale):
        prior=self._spatial_probe
        self._spatial_probe=node
        try:
            signature=tuple(node.observation_ids)
            history=self._relook_history.get((node.iid,signature),[])
            # A previously inconclusive place gets a fresh attempt only after
            # exploration actually brought the camera elsewhere, not in a loop.
            if (len(history)>=2 and self._needs_view.get(node.iid)==signature
                    and obs.step_count-history[-1][0]>=40
                    and all(math.dist(pose[:2],p)*scale>.45 for _,p in history)):
                history.clear()
            return super()._begin_relook(obs,node,pose,scale)
        finally:
            self._spatial_probe=prior

    def _rank_probe(self,fid,roi,pose,point,scale,excluded,budget=MAX_ACTIONS,uncertain=False):
        if self._spatial_probe is None or not uncertain:
            return super()._rank_probe(fid,roi,pose,point,scale,excluded,budget,uncertain)
        matrix=self.client.get_frame_pose(int(fid))
        if matrix is None:
            return None
        anchor=tuple(navigator.pose_to_yaw_2d(matrix,self.align_R))
        self._blocked_probe_edges=[(t,p) for t,p in self._blocked_probe_edges if self.now-t<=24]
        options=map_views(self._frontier_grid,pose,point,anchor,scale,excluded,self.report_near_dist_m,
                          [p for _,p in self._blocked_probe_edges],budget)
        # Include a translation; rotating in place cannot resolve depth ambiguity.
        options=[o for o in options if FORWARD in o[1] and target_relative_motion(anchor,o[0],point) is not None]
        source=self._source(fid,roi) if options else None
        if source is None:
            return None
        moves=[relative_action(anchor,p,scale) for p,_,_ in options]
        learned=np.asarray(self._verification_model(True).score(*source,moves),float).reshape(-1)
        motions=[target_relative_motion(anchor,p,point) for p,_,_ in options]
        angles,ratios=map(np.array,zip(*motions))
        if self._shape_view_model is None:
            from view_jepa import load_predictor
            self._shape_view_model=load_predictor()
        utility=np.asarray(self._shape_view_model.score_views(*source,angles,ratios),float).reshape(-1)
        if (learned.shape!=utility.shape or learned.shape!=(len(options),) or
                not np.isfinite([learned,utility]).all() or np.any((utility<0)|(utility>1))):
            raise ValueError('Invalid active-view prediction')
        # Also seek a place where two nearby map positions separate in bearing.
        rivals=[other.point[:2] for other in self.instance_store.nodes if other.iid!=self._spatial_probe.iid
                and math.dist(point,other.point[:2])*scale<1.]
        def separation(candidate):
            place=np.asarray(point)-candidate[:2]
            def contrast(rival):
                diff=np.asarray(rival)-candidate[:2]
                return min(1.,abs(math.remainder(math.atan2(*diff[::-1])-math.atan2(*place[::-1]),math.tau))/.35)
            return min(map(contrast,rivals),default=.5)
        context=np.array([separation(p) for p,_,_ in options])
        # Normalize the inherited predictor by ranks, since its units depend on
        # the loaded head. Equal scores receive equal ranks.
        ranks=np.array([np.mean(learned<x)+.5*np.mean(learned==x) for x in learned])
        scores=.30*ranks+.50*utility+.20*context
        i=max(range(len(options)),key=lambda j:(scores[j],-len(options[j][1])))
        end,commands,kind=options[i]
        self._note('spatial_view_selection',self._spatial_probe.iid,options=len(options),
                   learned=float(learned[i]),shape_jepa=float(utility[i]),place_separation=float(context[i]),
                   range_ratio=float(ratios[i]),azimuth_change_rad=float(angles[i]))
        return dict(waypoint=end,sequence=commands,source='shape_place_'+kind,source_displacement_m=math.hypot(*moves[i][:2]),
                    count=len(options),score=float(scores[i]),anchor=anchor)


class SpatialAgent(DelayedRelookAgent,_SpatialCapture):
    """Preserve Stage 8's delayed-map lifecycle with the spatial capture handler."""

    def _poll_pending_views(self,obs):
        # Stage 8 normally drops pending pictures if the node has since gained a
        # trusted crop. A good crop alone does not resolve a known place conflict.
        self._polling_place_evidence=True
        try:
            return super()._poll_pending_views(obs)
        finally:
            self._polling_place_evidence=False

    def _references(self,node):
        if (getattr(self,'_polling_place_evidence',False) and not node.reported
                and self._needs_view.get(node.iid)==tuple(node.observation_ids)):
            return []
        return super()._references(node)

    def _execute_decider_result(self,obs,result,event,images=None):
        if result.action in {'GOTO_INSTANCE','REPORT_FOUND'} and self._pending_report is None:
            node=self.instance_store.get(result.target_id)
            if node is not None:
                sig=tuple(node.observation_ids)
                attempts=self._relook_history.get((node.iid,sig),[])
                if (self._spatial_hold.get(node.iid)==sig and len(attempts)>=2 and
                        self._relook is None and 0<=obs.step_count-attempts[-1][0]<40):
                    self._poll_pending_views(obs)
                    if self._spatial_hold.get(node.iid)==sig:
                        self._note('unresolved_place_resume_exploration',node.iid)
                        return self._depart(obs,node)
        return super()._execute_decider_result(obs,result,event,images)
