"""Preserve original selected regions and their native semantic review receipts."""
import hashlib
import json
import os
from pathlib import Path


class CandidateEvidenceMixin:
    max_map_retries = 3
    max_review_reoffers = 2
    review_batch_size = 3

    def _init_report_memory(self):
        super()._init_report_memory()
        self._ev_candidates = {}
        self._ev_selections = {}
        self._ev_regions = {}
        self._ev_delayed = {}
        self._ev_retries = {}
        self._ev_offer_step = None
        self._ev_deciding = False
        self._ev_id = 0
        self._ev_confirmed_panels = set()
        self._ev_client_cursor = None

    def _ev_step(self):
        return int(getattr(self._last_observation, "step_count", 0) or 0)

    def _ev_log(self, event, **data):
        self._ev_id += 1
        row = dict(seq=self._ev_id, step=self._ev_step(), event=event, **data)
        with (Path(self.output_dir) / "candidate_evidence.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def _ev_save(self, images):
        refs = {}
        for name, payload in images or []:
            if not isinstance(payload, (bytes, bytearray)) or not payload:
                continue
            digest = hashlib.sha256(payload).hexdigest()
            directory = Path(self.output_dir) / "candidate_evidence_assets"
            directory.mkdir(mode=0o700, exist_ok=True)
            path = directory / digest
            if not path.exists():
                with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as out:
                    out.write(payload)
            refs[str(name)] = digest
        return refs

    def _tool_propose_candidates(self, frame_id, query=""):
        result = super()._tool_propose_candidates(frame_id, query)
        masks = [{k:r[k] for k in ("mask_id","area_frac") if k in r} for r in result.get("masks", [])]
        refs = self._ev_save(result.get("_tool_images"))
        if masks:
            fid = int(result.get("frame_id", frame_id))
            panels = {k:v for k,v in refs.items() if not k.startswith("proposal_")}
            signature = hashlib.sha256(json.dumps([masks,panels], sort_keys=True).encode()).hexdigest()
            previous = self._ev_regions.get(fid)
            if previous and previous != signature:
                old = self._ev_delayed.pop(fid, {}) or {"entry": self._queued_picks.get(fid)}
                if old.get("entry"):
                    self._queued_picks.pop(fid, None)
                    self._queued_pick_results.append(dict(frame_id=fid,failed=True,error="Original region table changed; view the replacement before selecting."))
                    self._ev_log("region_version_changed", frame_id=fid, mask_ids=old["entry"]["mask_ids"])
            self._ev_regions[fid] = signature
        self._ev_log("regions", frame_id=frame_id, masks=masks,
                     mapped=not bool(result.get("pending_mapping")) if "pending_mapping" in result else None,
                     error=result.get("error") or result.get("segmentation_error"),
                     panels=refs)
        return result

    def _ev_selection_key(self, fid, entry):
        return (fid, self._ev_regions.get(fid), tuple(entry["mask_ids"]), entry.get("goal_index"),
                str(entry.get("query") or "")[:300])

    def _tool_pick_segment(self, frame_id, mask_ids, query="", goal_index=None):
        try:
            fid = int(frame_id)
        except (TypeError, ValueError):
            fid = None
        self._ev_log("pick_requested", frame_id=frame_id, mask_ids=mask_ids, goal_index=goal_index)
        old = self._queued_picks.get(fid) or self._ev_delayed.get(fid, {}).get("entry")
        key = None
        if fid is not None and isinstance(mask_ids,(list,tuple)) and mask_ids and all(type(m) is int for m in mask_ids):
            key = self._ev_selection_key(fid, dict(mask_ids=mask_ids, query=query, goal_index=goal_index))
            if old and self._ev_selection_key(fid, old) == key:
                self._ev_log("existing_selection_waiting", frame_id=fid, mask_ids=mask_ids)
                return dict(queued=True, frame_id=fid, next="Your original mask selection is queued for mapping.")
            if key in self._ev_selections:
                rows = self._ev_selections[key]
                states = [dict(**r, status=self._proposal_queue.get(r["candidate_id"],{}).get("status","evicted")) for r in rows]
                self._ev_log("existing_selection_receipt", frame_id=fid, candidates=[r["candidate_id"] for r in rows])
                images = []
                for r in states:
                    cid = r["candidate_id"]
                    ref = self._ev_candidates[cid]["panel"]
                    if ref and r["status"] == "pending":
                        images.append(("proposal_"+cid,self._ev_panel(ref)))
                return dict(frame_id=fid, proposals=states, _tool_images=images, next="These are the original selection receipts. Review only pending candidates.")
        out = super()._tool_pick_segment(frame_id, mask_ids, query, goal_index)
        if key and out.get("proposals"):
            self._ev_selections[key] = list(out["proposals"])
        if fid in self._ev_delayed and (out.get("queued") or out.get("proposals")):
            self._ev_delayed.pop(fid)
        if old and out.get("queued") and list(old["mask_ids"]) != self._queued_picks[fid]["mask_ids"]:
            self._ev_log("queued_selection_superseded", frame_id=fid, mask_ids=old["mask_ids"])
        self._ev_log("pick_receipt", frame_id=fid, queued=bool(out.get("queued")),
                     candidates=[r["candidate_id"] for r in out.get("proposals", [])], error=out.get("error"))
        return out

    def _register_picked_candidates(self, resp, query, goal_index):
        before = set(self._proposal_queue)
        out = super()._register_picked_candidates(resp, query, goal_index)
        refs = self._ev_save(out.get("_tool_images"))
        for r in out.get("proposals", []):
            cid = r["candidate_id"]
            p = self._proposal_queue.get(cid, {})
            ref = refs.get("proposal_" + cid)
            self._ev_candidates[cid] = dict(frame_id=out.get("frame_id"), mask_id=r.get("mask_id"),
                                            panel=ref, panel_retries=0, reoffers=0, last_offer=None)
            self._ev_log("proposal", candidate_id=cid, frame_id=out.get("frame_id"), mask_id=r.get("mask_id"),
                         pixel=p.get("pixel_norm"), bbox=p.get("bbox"), goal_index=goal_index, panel=ref)
        for cid in sorted(before - set(self._proposal_queue)):
            self._ev_log("proposal_evicted", candidate_id=cid)
        if len(resp.get("candidates", [])) != len(out.get("proposals", [])):
            self._ev_log("mapper_rows_filtered", input_count=len(resp.get("candidates", [])),
                         queued_count=len(out.get("proposals", [])))
        return out

    def _process_queued_picks(self):
        step = self._ev_step()
        for fid, pending in list(self._ev_delayed.items()):
            if fid in self._queued_picks:
                del self._ev_delayed[fid]
            elif step >= pending["next_step"]:
                self._queued_picks[fid] = pending["entry"]
                del self._ev_delayed[fid]
        before = self._queued_picks.copy()
        count = len(self._queued_pick_results)
        result = super()._process_queued_picks()
        receipts = self._queued_pick_results[count:]
        retained = []
        for r in receipts:
            fid = r["frame_id"]
            err = str(r.get("error") or "")
            entry = before.get(fid)
            # The native async path retains the message but drops the RPC error code.
            # Only this explicit server message indicates a safe same-mask retry.
            transient = "candidate registration is not available yet" in err.lower()
            self._ev_log("mapping_receipt", frame_id=fid, mask_ids=(entry or {}).get("mask_ids"),
                         failed=bool(r.get("failed")), error=err or None,
                         candidates=[c["candidate_id"] for c in r.get("out", {}).get("proposals", [])])
            n = self._ev_retries.get(fid, 0)
            if entry and r.get("out",{}).get("proposals"):
                self._ev_selections[self._ev_selection_key(fid,entry)] = list(r["out"]["proposals"])
            if entry and transient and fid in self._ev_regions and n < self.max_map_retries:
                self._ev_retries[fid] = n+1
                wait = 4 * (2**n)
                self._ev_delayed[fid] = dict(next_step=step+wait, entry=entry)
                self._ev_log("same_selection_retry_scheduled", frame_id=fid, mask_ids=entry["mask_ids"],
                             retry=n+1, next_step=step+wait)
            else:
                retained.append(r)
        self._queued_pick_results[count:] = retained
        return result

    def _decider_next(self, *args, **kwargs):
        client_log = Path(self.output_dir) / "vlm_calls.jsonl"
        if self._ev_client_cursor is None:
            self._ev_client_cursor = client_log.stat().st_size if client_log.exists() else 0
        self._ev_deciding = True
        try:
            return super()._decider_next(*args, **kwargs)
        finally:
            self._ev_deciding = False
            self._ev_read_client_receipts(client_log)

    def _ev_read_client_receipts(self, path):
        if not path.exists():
            return
        with path.open("rb") as stream:
            stream.seek(self._ev_client_cursor)
            for raw in stream:
                if not raw.endswith(b"\n"):
                    break
                self._ev_client_cursor += len(raw)
                record = json.loads(raw)
                if record.get("kind") != "decision" or not record.get("ok"):
                    continue
                for img in record.get("images", []):
                    label = str(img.get("label", ""))
                    cid = label[len("proposal_"):]
                    if label.startswith("proposal_") and cid in self._ev_candidates:
                        self._ev_confirmed_panels.add(cid)
                        self._ev_log("successful_client_panel", candidate_id=cid, call_id=record.get("call_id"),
                                     request_step=(record.get("context") or {}).get("step"))

    def _build_decider_input(self, observation, local_map=False):
        if not self._ev_deciding and self._queued_pick_results:
            # A logging or planner snapshot builds the same state. Native code
            # drains this one-shot queue, so defer that delivery until a decision.
            saved = self._queued_pick_results
            self._queued_pick_results = []
            try:
                return super()._build_decider_input(observation, local_map)
            finally:
                self._queued_pick_results = saved + self._queued_pick_results
        state, image = super()._build_decider_input(observation, local_map)
        if self._ev_deciding:
            self._ev_offer_pending(state, int(observation.step_count))
        return state, image

    def _ev_offer_pending(self, state, step):
        if self._ev_offer_step == step:
            return
        self._ev_offer_step = step
        queued = state.get("queued_pick_results", [])
        fresh = {r["candidate_id"] for batch in queued for r in batch.get("proposals", [])}
        for cid in sorted(fresh):
            if cid in self._ev_candidates:
                self._ev_candidates[cid]["last_offer"] = step
            self._ev_log("queued_review_in_input", candidate_id=cid)
        # The native state delivers newly mapped rows and panels once. Repeat only
        # explicitly pending rows, capped per candidate and per decision.
        if fresh:
            for batch in queued:
                if batch.get("proposals"):
                    batch["next"] = "Inspect the attached original candidate panels now. Call commit_candidates for your reviewed rows with ACCEPT, REJECT or UNCERTAIN. Accept only visible goal matches."
            return
        missing = [(cid,e) for cid,e in self._ev_candidates.items() if not e["panel"]
                   and e["panel_retries"] < 2 and self._proposal_queue.get(cid,{}).get("status")=="pending"]
        for cid,e in missing[:1]:
            e["panel_retries"] += 1
            try:
                meta, payload = self.client.get_candidate_evidence(cid)
            except (OSError,RuntimeError,ValueError,TimeoutError) as exc:
                self._ev_log("panel_fetch_unavailable", candidate_id=cid, cause=type(exc).__name__)
                continue
            if meta.get("found") and payload:
                e["panel"] = self._ev_save([("proposal_"+cid,payload)])["proposal_"+cid]
                self._ev_log("original_panel_recovered", candidate_id=cid, panel=e["panel"])
            else:
                self._ev_log("panel_fetch_unavailable", candidate_id=cid, cause="mapper_returned_no_panel")
        pending = [(cid,e) for cid,e in self._ev_candidates.items()
                   if self._proposal_queue.get(cid, {}).get("status") == "pending"
                   and e["panel"] and e["reoffers"] < self.max_review_reoffers]
        pending.sort(key=lambda pair:(pair[0] in self._ev_confirmed_panels, pair[1]["reoffers"],
                                      pair[1]["last_offer"] or -1))
        attachments = {label for label, _ in self._pending_decision_images}
        per_frame = {}
        # Native image packing keeps tail panels. Place the highest priority one last.
        for cid, e in reversed(pending[:self.review_batch_size]):
            name = "proposal_" + cid
            data = self._ev_panel(e["panel"])
            if name not in attachments:
                self._pending_decision_images.append((name, data))
            per_frame.setdefault(e["frame_id"], []).append(dict(candidate_id=cid, mask_id=e["mask_id"]))
            e.update(reoffers=e["reoffers"]+1, last_offer=step)
            self._ev_log("pending_review_in_input", candidate_id=cid, frame_id=e["frame_id"],
                         offer=e["reoffers"], panel=e["panel"])
        for fid, rows in per_frame.items():
            state.setdefault("queued_pick_results", []).append(dict(frame_id=fid, proposals=rows,
                next="These selected regions still await your review. Inspect the original attached panels and use commit_candidates with ACCEPT, REJECT or UNCERTAIN. Accept only visible goal matches."))

    def _ev_panel(self, digest):
        data = (Path(self.output_dir) / "candidate_evidence_assets" / digest).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Candidate panel digest mismatch")
        return data

    def _tool_commit_candidates(self, reviews, label=""):
        selected = [r for r in reviews[:16] if isinstance(r, dict)] if isinstance(reviews,(tuple,list)) else []
        for r in selected:
            self._ev_log("review_requested", candidate_id=r.get("candidate_id"), verdict=r.get("verdict"))
        out = super()._tool_commit_candidates(reviews, label)
        for r in selected:
            cid = str(r.get("candidate_id") or "")
            self._ev_log("review_receipt", candidate_id=cid, status=self._proposal_queue.get(cid, {}).get("status"),
                         instances=[row for row in out.get("instances", []) if row.get("candidate_id")==cid],
                         linked=[row for row in out.get("already_registered", []) if row.get("candidate_id")==cid],
                         geometry=[row for row in out.get("geometry_rejections", []) if row.get("candidate_id")==cid],
                         error=out.get("error"))
        for row in out.get("duplicate_review", []):
            self._ev_log("identity_review", observation_id=row.get("observation_id"),
                         candidate_id=self._ev_observation_candidate(row.get("observation_id")))
        return out

    def _ev_observation_candidate(self, oid):
        obs = self.instance_store.get_observation(oid)
        return getattr(obs, "candidate_id", None)

    def _tool_resolve_duplicate(self, observation_id, decision, duplicate_of=None, text=""):
        out = super()._tool_resolve_duplicate(observation_id, decision, duplicate_of, text)
        self._ev_log("identity_receipt", observation_id=observation_id,
                     candidate_id=self._ev_observation_candidate(observation_id), verdict=decision,
                     status=out.get("status"), instance_id=out.get("instance_id"), error=out.get("error"))
        return out
