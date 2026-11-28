"""Keep the original region choice and delegate inspection intent in the native MOST host."""
from gpt_agent import GPTRegionAgent
from trigger_policy import InspectOnlyPolicy


class InspectOnlyAgent(GPTRegionAgent):
    policy_class = InspectOnlyPolicy

    def _bind_choice(self, obs, scan, native, decision, region):
        if decision.get("gpt_called"):
            if decision["executed_region_id"] != decision["old_selected_region_id"] or (
                region is None or region["key"] != decision["old_selected_region_key"]
            ):
                raise RuntimeError("native exploration selected a different region from the frozen ranking")
        return super()._bind_choice(obs, scan, native, decision, region)
