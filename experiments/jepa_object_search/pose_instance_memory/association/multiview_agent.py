"""Optional local detector proposals and geometric separation of reviewed instances."""
import os
from pathlib import Path
import sys
import cv2
import numpy as np

from geometry_links import match_views, overlap
from map_views import metric_view, point_in_box, rgb_aligned
from reid_agent import ReIDFrontier
from report_agent import PROJECT
from regions import crop_evidence
from identity import relation


class MultiviewFrontier(ReIDFrontier):
    def _init_report_memory(self):
        super()._init_report_memory()
        self._local_boxes={}
        self._submitted=set()
        self._map_step=None
        self._snapshot={}
        self._detector_failed=False
        self._pose_failed=False

    def _category(self,observation,query=""):
        # An image-instance goal may provide no text. Never guess its class.
        return (query or (self._target_phrase(observation) if observation is not None else "")).strip()

    def _box_detector(self):
        if not hasattr(self,"_proposal_worker"):
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"detector"))
            from proposal_worker import ProposalWorker
            self._proposal_worker=ProposalWorker(PROJECT/"b_verification/env/bin/python",PROJECT,
                                                 os.environ.get("NAV_BOX_DEVICE","cpu"))
        return self._proposal_worker

    def _boxes(self,key,rgb,category):
        if self._detector_failed or not category:
            return []
        tag=(key,category)
        if tag not in self._local_boxes:
            try:
                rows=self._box_detector().detect(np.asarray(rgb)[:,:,:3],category)
            except (OSError,ValueError,RuntimeError,EOFError) as exc:
                self._detector_failed=True
                worker=getattr(self,"_proposal_worker",None)
                if worker is not None:
                    worker.close()
                    del self._proposal_worker
                self._note("local_detector_unavailable",None,reason=type(exc).__name__)
                return []
            if len(self._local_boxes)>=48:
                self._local_boxes.pop(next(iter(self._local_boxes)))
            self._local_boxes[tag]=rows
        return self._local_boxes[tag]

    def _build_decider_input(self,observation,local_map=False):
        state,img=super()._build_decider_input(observation,local_map)
        if os.environ.get("NAV_LOCAL_BOXES")!="1":
            return state,img
        category=self._category(observation)
        boxes=self._boxes(("live",observation.step_count),observation.rgb,category)
        if boxes:
            state=dict(state,local_rgb_proposals=dict(
                step=observation.step_count,category=category,boxes=[b["bbox"] for b in boxes[:12]],
                width=observation.rgb.shape[1],height=observation.rgb.shape[0],
                guidance="Unreviewed pixel xyxy rectangles on this observation. Review a matching keyframe before registering an instance."))
        return state,img

    def _map_snapshot(self):
        step=int(self.now)
        if step!=self._map_step:
            self._map_step=step
            self._snapshot={r["frame_id"]:r for r in self.client.get_frame_points(stride=8)}
        return self._snapshot

    def _add_local_candidates(self,fid,category):
        key=(fid,category)
        if key in self._submitted or not category:
            return None
        meta,jpeg=self.client.get_frame_image(fid)
        if not jpeg or not meta.get("found"):
            return None
        image=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            return None
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        rows=self._boxes(("map",fid),rgb,category)
        snap=self._map_snapshot().get(fid) if rows else None
        if snap is None or not rgb_aligned(snap,rgb,8):
            return None
        norm=np.array((1000./rgb.shape[1],1000./rgb.shape[0]))
        chosen=[]
        for r in rows[:12]:
            box=np.array(r["bbox"],float)
            center=((box[:2]+box[2:])*.5*norm).tolist()
            if not self._spot_rejected(fid,center):
                chosen.append((box*np.tile(norm,2),center))
        if not chosen:
            return None
        answer=self.client.prepare_pixels(fid,[cent for _,cent in chosen],normalized=True)
        candidates=answer.get("candidates",[])
        if not candidates:
            return None
        self._submitted.add(key)
        picked=[]
        for c in candidates:
            # Native refined masks are in point-grid pixels; the returned click
            # and normalized click identify the exact grid dimensions.
            pixel=np.array(c.get("pixel",[]),float)
            n=np.array(c.get("pixel_norm",[]),float)
            if not c.get("refined") or c.get("bbox") is None or pixel.shape!=(2,) or n.shape!=(2,) or np.any(pixel<=0) or np.any(n<=0):
                continue
            normalized=np.asarray(c["bbox"])*np.tile(n/pixel,2)
            if max(overlap(normalized,b) for b,_ in chosen)<.30:
                continue
            if any(overlap(normalized,e)>.65 for _,e in picked):
                continue
            picked.append((c,normalized))
        if not picked:
            return None
        # This native path only adds pending transactions and evidence images;
        # the decider must still perform commit_candidates and depth validation.
        return self._register_picked_candidates({"candidates":[c for c,_ in picked]},category,None)

    def _tool_propose_candidates(self,frame_id,query=""):
        native=super()._tool_propose_candidates(frame_id,query)
        if os.environ.get("NAV_LOCAL_BOXES")!="1" or native.get("stopped"):
            return native
        try:
            extra=self._add_local_candidates(int(frame_id),self._category(self._last_observation,query))
        except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
            self._note("local_proposals_unavailable",None,reason=type(exc).__name__)
            return native
        if not extra:
            return native
        out=dict(native)
        if "error" in out:
            out["segmentation_error"]=out.pop("error")
        out["local_pending_proposals"]=extra["proposals"]
        out["next"]=str(out.get("next",""))+" Inspect the local proposal panels and use commit_candidates for independently verified boxes."
        out["_tool_images"]=list(out.get("_tool_images",[]))+list(extra.get("_tool_images",[]))
        return out

    def _distinct_at(self,left_records,right_records):
        if os.environ.get("NAV_COVISIBLE_GUARD")!="1":
            return False
        by_frame={}
        for r in self.instance_store.observations.values():
            if r.frame_id is not None and r.bbox is not None:
                by_frame.setdefault(r.frame_id,[]).append(r)
        lf={r.frame_id for r in left_records}
        rf={r.frame_id for r in right_records}
        pairs=[(i,j) for i in lf for j in rf if i!=j and len(by_frame.get(i,[]))>=2 and len(by_frame.get(j,[]))>=2]
        if not pairs:
            return False
        scale=self._metric_scale_value()
        if scale is None or not np.isfinite(scale) or scale<=0:
            return False
        try:
            snaps=self._map_snapshot()
            frames={}
            for fid in set(x for pair in pairs for x in pair):
                view=metric_view(snaps[fid],scale,8) if fid in snaps else None
                if view:
                    boxes=[r.bbox for r in by_frame[fid]]
                    frames[fid]=dict(view,boxes=boxes,points=[point_in_box(view,b) for b in boxes])
            left_ids={r.oid for r in left_records}
            right_ids={r.oid for r in right_records}
            for i,j in pairs:
                if i in frames and j in frames:
                    proof=match_views(frames[i],frames[j])
                    if any(by_frame[i][a].oid in left_ids and by_frame[j][b].oid in right_ids for a,b in proof["distinct"]):
                        return True
        except (OSError,RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
            self._note("multiview_geometry_unavailable",None,reason=type(exc).__name__)
        return False

    def _tool_resolve_duplicate(self,observation_id,decision,duplicate_of=None,text=""):
        if str(decision).strip().upper()=="DUPLICATE":
            a=self.instance_store.get_observation(observation_id)
            b=self.instance_store.get(duplicate_of)
            if a is not None and b is not None and self._distinct_at([a],self._records(b)):
                return {"error":"Both calibrated nearby views separately support two physical candidates. Preserve both for review."}
        return super()._tool_resolve_duplicate(observation_id,decision,duplicate_of,text)

    def _tool_merge_instances(self,instance_id,other_instance_id,text=""):
        a,b=self.instance_store.get(instance_id),self.instance_store.get(other_instance_id)
        if a is not None and b is not None and self._distinct_at(self._records(a),self._records(b)):
            return {"error":"Independent candidates are supported across two calibrated views. Keep both."}
        return super()._tool_merge_instances(instance_id,other_instance_id,text)

    def _metric_observation(self,r,scale):
        snap=self._map_snapshot().get(r.frame_id)
        if snap is None or r.bbox is None:
            return None
        v=metric_view(snap,scale,8)
        if v is None:
            return None
        point=point_in_box(v,r.bbox)
        if point is None:
            return None
        meta,raw=self.client.get_frame_image(r.frame_id)
        if not meta.get("found") or not raw:
            return None
        bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if bgr is None or not rgb_aligned(snap,bgr[:,:,::-1],8):
            return None
        wh=np.array(v["grid"].shape[1::-1])*8
        box=np.asarray(r.bbox)*np.tile(np.array(bgr.shape[1::-1])/wh,2)
        crop=crop_evidence(raw,box)
        return (crop,v["c2w"],point) if crop is not None else None

    def _pose_hint(self,row,weights):
        obs=self.instance_store.get_observation(row["observation_id"])
        scale=self._metric_scale_value()
        if (obs is None or obs.frame_id is None or scale is None or not np.isfinite(scale) or scale<=0
                or self.instance_store.instance_for_observation(obs.oid)):
            return None
        neighbors=self.instance_store.nearby(obs.point,scale,self.instance_dup_radius_m,
                                            top_k=max(1,len(self.instance_store.nodes)))
        if len(neighbors)!=1:
            return None
        dist,node=neighbors[0]
        if not np.isfinite(dist) or node.iid not in {r["instance_id"] for r in row["neighbors"]}:
            return None
        records=self._records(node)
        if not records or relation([obs],records,obs.evidence.get("goal_index"),self._instance_goal_index.get(node.iid))!="unknown":
            return None
        if self._distinct_at([obs],records):
            return None
        ref=min(records,key=lambda r:(r.step,r.oid))
        if ref.frame_id is None or ref.frame_id==obs.frame_id or ref.step>obs.step:
            return None
        left,right=self._metric_observation(ref,scale),self._metric_observation(obs,scale)
        if left is None or right is None:
            return None
        pa,pb=left[1],right[1]
        angle=np.degrees(np.arccos(np.clip(pa[:3,2] @ pb[:3,2],-1.,1.)))
        if np.linalg.norm(pa[:3,3]-pb[:3,3])>1.25 or angle>45:
            return None
        if not hasattr(self,"_pose_worker"):
            from pose_worker import PoseWorker
            exe=PROJECT.parent/"jepa_instance_bench_20261016/policy_env/bin/python"
            self._pose_worker=PoseWorker(exe,PROJECT,weights,os.environ.get("NAV_REID_DEVICE","cpu"))
        out=self._pose_worker.compare(left,right)
        if not out["match"]:
            return None
        return dict(instance_id=int(node.iid),score=float(out["score"]),threshold=float(out["threshold"]),
                    guidance="View-conditioned support for review; confirm independently before linking.")

    def _dup_review_payload(self):
        rows,images,omit=super()._dup_review_payload()
        weights=os.environ.get("NAV_REID_POSE_WEIGHTS")
        if weights and not self._pose_failed:
            for r in rows:
                try:
                    hint=self._pose_hint(r,weights)
                except (OSError,ValueError,KeyError,RuntimeError,EOFError,np.linalg.LinAlgError) as exc:
                    self._note("pose_hint_unavailable",None,reason=type(exc).__name__)
                    if isinstance(exc,(OSError,RuntimeError,EOFError)):
                        self._pose_failed=True
                        if hasattr(self,"_pose_worker"):
                            self._pose_worker.pipe.close()
                            del self._pose_worker
                        break
                    continue
                if hint:
                    r["pose_jepa_review"]=hint
        return rows,images,omit
