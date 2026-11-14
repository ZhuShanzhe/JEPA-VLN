"""Single-category, 19-step controlled fixture. Only decision-time tensors enter score()."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ReportMemoryRanker(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(31)
        basis, _ = torch.linalg.qr(torch.randn(768,24,generator=generator))
        self.register_buffer("basis", basis)
        self.predict = nn.Sequential(nn.Linear(24+2+19*3,48), nn.GELU(), nn.Linear(48,24))
        self.visual_identity = nn.Linear(24,24,bias=False)

    def visual(self, features):
        return F.normalize(features @ self.basis, dim=-1)

    def forward(self, context, candidate_xy, plans, reported, report_xy):
        ctx = self.visual(context).expand(len(plans),-1)
        commands = F.one_hot(plans,num_classes=3).float().flatten(1)
        delta = self.predict(torch.cat([ctx,candidate_xy / 5.,commands],dim=1))
        predicted = F.normalize(ctx + delta,dim=-1)
        if not len(reported):
            # This controlled room has a matching chair in both candidates.
            # With no report its anti-duplication cue is neutral across candidates.
            return torch.zeros(len(plans),device=ctx.device), predicted
        reports = F.normalize(self.visual_identity(self.visual(reported)),dim=1)
        # Report coordinates are camera poses recorded when the observed report was made.
        # They are not the unknown goal location and are available for both variants.
        same_region = torch.exp(-torch.cdist(candidate_xy,report_xy).square() / 2.)
        redundancy = (predicted @ reports.T + same_region).amax(dim=1)
        return -5. * redundancy, predicted

    @torch.inference_mode()
    def score(self, context_embedding, candidate_xy, planned_actions, report_embeddings, report_xy):
        context, xy, reports = [torch.as_tensor(v,dtype=torch.float32,device=self.basis.device)
                                for v in (context_embedding,candidate_xy,report_embeddings)]
        plans = torch.as_tensor(planned_actions,device=context.device)
        if context.shape != (768,) or xy.ndim != 2 or xy.shape[1:] != (2,) or len(xy) == 0:
            raise ValueError("Expected current 768-vector and N nonempty relative (x,z) points")
        if reports.ndim != 2 or reports.shape[1] != 768:
            raise ValueError("Expected K observed visual report vectors of dimension 768")
        report_xy = torch.as_tensor(report_xy,dtype=torch.float32,device=context.device)
        if report_xy.shape != (len(reports),2):
            raise ValueError("Expected K relative (x,z) observed report camera poses")
        if not all(torch.isfinite(v).all() for v in (context,xy,reports,report_xy)):
            raise ValueError("Nonfinite model input")
        if plans.dtype == torch.bool or plans.is_floating_point() or plans.shape != (len(xy),19):
            raise ValueError("Expected N x 19 integer nominal action codes")
        if plans.min() < 0 or plans.max() > 2:
            raise ValueError("Unknown nominal action")
        logits, _ = self(context,xy,plans.long(),reports,report_xy)
        if not torch.isfinite(logits).all():
            raise ArithmeticError("Nonfinite scores")
        return logits.sigmoid().cpu().tolist()


def load_memory_ranker(checkpoint):
    payload = torch.load(Path(checkpoint),map_location="cpu",weights_only=True)
    if payload["format"] != "one-room-report-visual-fixture/v1":
        raise ValueError("Unknown controlled memory checkpoint")
    model = ReportMemoryRanker()
    model.load_state_dict(payload["state"],strict=True)
    return model.eval()
