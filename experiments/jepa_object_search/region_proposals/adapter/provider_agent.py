"""Switch the source of visual proposals while keeping the native semantic and geometric review."""
import os
from pathlib import Path
from run_agent import LearnedVerificationFrontier


class NativeLiveBoxes:
    def __init__(self,project,device):
        from box_service import LocalBoxService
        self.sam=LocalBoxService(project/"mapping_runtime/env/bin/python",project,"sam",device)
        try:
            self.reviewer=LocalBoxService(project/"b_verification/env/bin/python",project,"review",device)
        except BaseException:
            self.sam.close()
            raise

    def detect(self,rgb,category):
        proposals=self.sam.query(rgb,category)
        ranked=self.reviewer.query(rgb,category,proposals) if proposals else []
        # The report gate expects class-conditioned proposals. Arbitrary masks
        # for walls and furniture cannot stand in for a class detector.
        return [dict(row,source="native_sam_local_review") for row in ranked[:3] if row["review_margin"]>0]

    def close(self):
        self.reviewer.close()
        self.sam.close()


class RefinedBoxes:
    def __init__(self,project,device):
        from proposal_worker import ProposalWorker
        from box_refiner import BoxRefiner
        ck=os.environ.get("NAV_BOX_REFINER") or project/"jepa_box_v26/refine/weights/box_refiner.npz"
        self.refiner=BoxRefiner(ck)
        self.original=ProposalWorker(project/"b_verification/env/bin/python",project,device)

    def detect(self,rgb,category):
        return self.refiner.refine(rgb,self.original.detect(rgb,category),category)

    def close(self):
        self.original.close()


class BoxComparisonAgent(LearnedVerificationFrontier):
    def _init_report_memory(self):
        mode=os.environ.get("NAV_BOX_PROVIDER","dino").lower()
        if mode not in ("dino","native","refined"):
            raise ValueError("Unknown NAV_BOX_PROVIDER")
        if getattr(self,"_proposal_mode",mode)!=mode and hasattr(self,"_proposal_worker"):
            self._proposal_worker.close()
            del self._proposal_worker
        super()._init_report_memory()
        self._proposal_mode=mode

    def _box_detector(self):
        if self._proposal_mode=="dino":
            return super()._box_detector()
        if not hasattr(self,"_proposal_worker"):
            project=Path(os.environ["NAV_ROOT"])
            device=os.environ.get("NAV_BOX_DEVICE","cpu")
            self._proposal_worker=(NativeLiveBoxes if self._proposal_mode=="native" else RefinedBoxes)(project,device)
        return self._proposal_worker

    def _add_local_candidates(self,fid,category):
        if self._proposal_mode=="native":
            # Native propose_candidates already emits the authoritative mask
            # IDs for this frame. Do not turn those masks back into point prompts.
            return None
        return super()._add_local_candidates(fid,category)

    def _scout_observe(self,fid,noun):
        if self._proposal_mode=="native":
            self._scout_checked.add(fid)
            return
        return super()._scout_observe(fid,noun)
