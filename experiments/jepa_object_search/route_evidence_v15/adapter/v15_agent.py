"""Goal-conditioned evidence worker with the accepted MOST execution adapter."""
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import subprocess
import sys


PROJECT = Path(os.environ.get("NAV_ROOT", Path(__file__).resolve().parents[2]))
ADAPTER = PROJECT / "jepa_evidence_v15/adapter"
sys.path[:0] = [str(PROJECT / "jepa_exec_v14"), str(PROJECT / "jepa_env_v13/most_integration")]
from benchmark_api import Action
from jepa_exec_agent import JepaExecutionFrontier


class EvidenceChannel:
    def __init__(self, stream, report_steps):
        self.stream = stream
        self.report_steps = report_steps

    def send(self, msg):
        self.stream.send(dict(msg, report_steps=list(self.report_steps)))
        self.report_steps.clear()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class EvidenceFrontier(JepaExecutionFrontier):
    def init_episode(self, observation):
        self.observed_reports = []
        super().init_episode(observation)

    def _feed_frame(self, observation):
        previous = self.observed_step
        result = super()._feed_frame(observation)
        if (previous < self.observed_step and self.now > 0 and
                observation.previous_action == int(Action.TARGET_FOUND)):
            self.observed_reports.append(self.now)
        return result

    def _connect(self):
        left, right = socket.socketpair()
        try:
            env = dict(os.environ)
            for name in ("PYTHONPATH", "PYTHONHOME"):
                env.pop(name, None)
            env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            python = PROJECT.parent / "jepa_instance_bench_20261016/policy_env/bin/python"
            with (Path(self.output_dir) / "v15_worker.log").open("ab") as log:
                self.worker = subprocess.Popen([str(python), "-B", str(ADAPTER / "evidence_worker.py"),
                                                str(right.fileno()), self.word], env=env,
                                               pass_fds=(right.fileno(),), stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT)
            self.channel = EvidenceChannel(Connection(left.detach()), self.observed_reports)
        finally:
            left.close()
            right.close()
