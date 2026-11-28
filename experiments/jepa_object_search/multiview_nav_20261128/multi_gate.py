"""Keep frozen upper-VLM transport/inputs. Permit visual evidence in any observed camera direction."""
from unittest.mock import patch
import gpt_gate as previous

SYSTEM = """You decide where an embodied camera should search next for MULTIPLE DISTINCT instances of the named target, using the same limited movement and turn budget throughout. Only current observations and online map candidates are known. Select one supplied reachable region ID. Consider plausible scene areas, new ground that could be covered, and prior searches and blocked routes. Prefer a distinct promising unexplored part after checking one location. If a target is clear, choose a region that approaches that target or the next promising area if this location was already checked. A possible support surface alone is usually too weak to justify close inspection.

Look at all three full RGB views, arranged left to right on the page as camera RIGHT (index 0), CENTER (index 1), LEFT (index 2). Positive bearing is to the robot's left. The nine auxiliary patch descriptors in the metadata describe only the center image and are imperfect. Each full camera view is eligible even if its target is absent from the center image. Choose inspect only when a specific visible cue justifies a closer or lateral view to resolve ambiguity: a partial target, a specifically indicated occlusion, or ambiguity whether an observed object is a new instance. Choose explore for general searching, clear targets that need no further inspection, and ordinary walls/tables/furniture without specific evidence. A separate verifier will localize a candidate box, and the lower planner may decline an unreachable inspection.

Return only JSON with region_id (one offered region ID), mode (explore or inspect), patch_id (for inspect, the suggested CAMERA VIEW INDEX 0, 1, or 2; for explore, null), target_evidence (none, suspected, or clear, your visual judgment only), and cue (none, partial_target, occlusion, similar_instance, or target_visible). Do not return a patch-grid cell or a box."""


class MultiGate(previous.GPTGate):
    def choose(self, **kwargs):
        validate = previous.decision
        def view_decision(obj, ids, patch_ids):
            return validate(obj, ids, set(patch_ids) & {0, 1, 2})
        with patch.object(previous, "SYSTEM", SYSTEM), patch.object(previous, "decision", view_decision):
            return super().choose(**kwargs)
