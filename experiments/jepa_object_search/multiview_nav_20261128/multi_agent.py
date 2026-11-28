"""Native MOST optional entry; aligned camera depth is unavailable on this interface."""
import os
from pathlib import Path

from commit_agent import CommittedInspectAgent
from cue_service import CueSelector as NativeCue
from multi_gate import MultiGate
from multi_policy import MultiViewPolicy
from multi_worker import start_worker


class MultiViewAgent(CommittedInspectAgent):
    policy_class = MultiViewPolicy

    def init_episode(self, obs):
        if getattr(self,"_native_cue",None) is not None:
            self._native_cue.close()
        super().init_episode(obs)
        self._native_cue = None

    def _connect(self):
        project = Path(os.environ["NAV_ROOT"])
        self.channel,self.worker = start_worker(project,Path(self.output_dir)/"multi_worker.log",cpu=True)

    def _rank_after_scan(self, obs, scan):
        if self._gate is None:
            self._gate = MultiGate(self.episode,Path(self.output_dir)/"multiview_gate")
        if self._native_cue is None:
            self._native_cue = NativeCue(Path(self.output_dir)/"multiview_cue"/self._gate.path.stem)
        if self._gated is None:
            self._gated = self.policy_class(self._gate,instruction=self._instruction,cue=self._native_cue)
        super()._rank_after_scan(obs,scan)
