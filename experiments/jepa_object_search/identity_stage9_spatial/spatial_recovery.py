"""Optional audit/runtime variant. Recognize an old object in an off-track detection.

A result belongs to a particular detection. It never aliases the pursued clue.
"""
from dataclasses import dataclass
import math

import numpy as np

from agents import navigator
from appearance import _rgb
from confirm_view import MAX_ACTIONS
from local_match import match_prepared, prepare_reference
from map_views import rgb_aligned
from optics import skew_view
from spatial_agent import SpatialAgent, _SpatialCapture
from spatial_identity import Anchor, Decision, _compare, _consistent_surface, make_anchor
from trusted_reference import MIN_DETECTOR_SCORE
from visual_witness import compare_regions, describe, strong_object


@dataclass(frozen=True)
class ReferenceView:
    anchor: Anchor
    body: bool
    joint: bool  # frozen visual verifier requires object content AND surroundings


@dataclass(frozen=True)
class OwnerView:
    center: tuple
    reported: bool
    references: tuple


def _joint_place(query,ref):
    # If the local surrounding surface is unavailable, allow the frozen joint
    # visual test to supply contextual support; a measured disagreement vetoes it.
    anchor=ref.anchor
    if (not ref.joint or query.frame_id==anchor.frame_id
            or math.dist(query.camera,anchor.camera)<.10):
        return False
    delta=np.asarray(query.center)-anchor.center
    distance=np.linalg.norm(delta)
    uncertainty=math.hypot(query.sigma,anchor.sigma)
    if uncertainty>.18 or distance>.19 or distance+uncertainty>=.32:
        return False
    if query.surface is not None and anchor.surface is not None:
        if not _consistent_surface(query,anchor,delta):
            return False
    # Reliable intrinsic conflicts must veto even when the missing plane
    # prevented the original spatial classifier from reaching its color gate.
    if query.hue is not None and anchor.hue is not None:
        if abs((query.hue-anchor.hue+180)%360-180)>68:
            return False
    return True


def recognize_seen(query,track_center,owners):
    """Recognize this off-track detection without inferring the tracked node's identity."""
    if query is None or track_center is None or not np.isfinite(track_center).all():
        return Decision('uncertain',None,'missing_tracking_geometry')
    # Use the existing broad target-box radius plus center error. A plausible
    # detection of the tracked node belongs to the normal association path.
    if math.dist(query.center,track_center)<=.48+query.sigma:
        return Decision('uncertain',None,'possible_tracked_object')
    plausible=[]
    accepted=[]
    for iid,owner in owners.items():
        if not np.isfinite(owner.center).all():
            plausible.append(iid)
            continue
        gap=math.dist(query.center,owner.center)
        if gap>.95:  # same broad registry lookup used by SpatialAgent
            continue
        pairs=[_compare(query,ref.anchor,'strong' if ref.body else 'unknown')
               for ref in owner.references]
        if pairs and all(p=='far' for p in pairs):
            if gap<=.19:  # inconsistent registry and trusted references
                plausible.append(iid)
            continue
        if not pairs and gap>.48+query.sigma:
            continue
        plausible.append(iid)
        if (not owner.reported or not pairs or gap>.48
                or {'far','intrinsic_conflict'}.intersection(pairs)):
            continue
        if 'match' in pairs:
            accepted.append((iid,'observed_old_mapped_place'))
        elif any(_joint_place(query,ref) for ref in owner.references):
            accepted.append((iid,'observed_old_joint_visual_place'))
    if len(plausible)==len(accepted)==1:
        return Decision('same',accepted[0][0],accepted[0][1])
    return Decision('uncertain',None,'ambiguous_owners' if len(plausible)>1 else 'insufficient_old_instance_evidence')


class _RecoveryCapture(_SpatialCapture):
    def _init_report_memory(self):
        super()._init_report_memory()
        self._old_sightings={}
        self._off_track_views={}

    def _recovery_owners(self,node,query,rgb,box,snapshots,scale):
        owners={}
        descriptor=describe(rgb,box)
        for other in self.instance_store.nodes:
            if other.iid==node.iid:
                continue
            # All centers and reference grids compared here use metric raw-map coordinates.
            center=self.align_R.T@(np.asarray(other.point,dtype=float)*scale)
            if np.linalg.norm(np.asarray(query.center)-center)>.95:
                continue
            refs=[]
            with self._reference_purpose(False):
                saved=self._references(other)
            for fid,ref in saved:
                src=self._trusted_pixels.get((fid,tuple(ref.roi)))
                snap=snapshots.get(fid)
                if src is None or snap is None or fid==query.frame_id or not rgb_aligned(snap,src[0],8):
                    continue
                prior=skew_view(snap,scale)
                anc=make_anchor(prior,*src,frame_id=fid) if prior is not None else None
                if anc is None:
                    continue
                body=strong_object(compare_regions(describe(*src),descriptor))
                joint=match_prepared(prepare_reference(*src),rgb,[box])==0
                refs.append(ReferenceView(anc,body,joint))
            owners[other.iid]=OwnerView(tuple(center),bool(other.reported),tuple(refs))
        return owners

    def _remember_offtrack(self,node,fid,verdict):
        owner=self.instance_store.get(verdict.owner)
        if verdict.status!='same' or owner is None or not owner.reported or owner.iid==node.iid:
            return False
        # Keep separately from verified registration views: a sighting cannot
        # propagate its own identity to another view or suppress a new target.
        sightings=self._old_sightings.setdefault(owner.iid,{})
        if fid in sightings:
            return False
        sightings[fid]=verdict.reason
        sig=tuple(node.observation_ids)
        self._off_track_views.setdefault((node.iid,sig),set()).add(fid)
        self._needs_view[node.iid]=sig
        self._note('incidental_reported_sighting',owner.iid,frame=int(fid),pursued=node.iid,
                   basis=verdict.reason)
        return True

    def _inspect_capture(self,node,fid,snapshot,scale):
        reason,result=super()._inspect_capture(node,fid,snapshot,scale)
        if reason!='no_unique_spatial_box':
            return reason,result
        info,data=self.client.get_frame_image(fid)
        mapped=_rgb(data) if data and info.get('found') else None
        if mapped is None or not rgb_aligned(snapshot,mapped,8):
            return reason,result
        view=skew_view(snapshot,scale)
        if view is None:
            return reason,result
        native=self._native_captures.get(fid)
        if native is not None and rgb_aligned(snapshot,native[0],8):
            rgb,rows=native
        else:
            rgb=mapped
            rows=self._boxes(('map',fid),rgb,self._category(self._last_observation))
        snapshots=self._map_snapshot()
        center=self.align_R.T@(np.asarray(node.point,dtype=float)*scale)
        by_owner={}
        for row in rows:
            if row.get('score',0)<MIN_DETECTOR_SCORE:
                continue
            box=row.get('bbox')
            query=make_anchor(view,rgb,box,frame_id=fid)
            if query is None:
                continue
            owners=self._recovery_owners(node,query,rgb,box,snapshots,scale)
            verdict=recognize_seen(query,center,owners)
            if verdict.status=='same':
                by_owner.setdefault(verdict.owner,[]).append(verdict)
        recorded=False
        for claims in by_owner.values():
            # Identical proposals or same-class neighbors require another view.
            if len(claims)==1:
                recorded=self._remember_offtrack(node,fid,claims[0]) or recorded
        return ('observed_old_seek_pursued_clue',None) if recorded else (reason,result)

    def _rank_probe(self,fid,roi,pose,point,scale,excluded,budget=MAX_ACTIONS,uncertain=False):
        node=self._spatial_probe
        if node is not None and uncertain:
            blocked=self._off_track_views.get((node.iid,tuple(node.observation_ids)),())
            extra=[]
            for frame in sorted(blocked):
                matrix=self.client.get_frame_pose(int(frame))
                if matrix is not None:
                    extra.append(tuple(navigator.pose_to_yaw_2d(matrix,self.align_R))[:2])
            if extra:
                ranked=super()._rank_probe(fid,roi,pose,point,scale,[*excluded,*extra],budget,uncertain)
                if ranked is not None:
                    return ranked
        return super()._rank_probe(fid,roi,pose,point,scale,excluded,budget,uncertain)


class RecoverySpatialAgent(SpatialAgent,_RecoveryCapture):
    """Install the same inspection after DelayedRelookAgent for immediate and deferred observations."""
