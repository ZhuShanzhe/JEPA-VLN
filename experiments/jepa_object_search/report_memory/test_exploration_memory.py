"""Exercise the production V18 view channel and the V19 report-memory wrapper together."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

from exploration_memory import MAX_PENALTY, MemoryChannel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"jepa_confirm_v18/agent"))
from view_explore import ViewChannel


class Worker:
    def __init__(self, reply):
        self.reply = reply

    def send(self, message):
        self.message = message

    def recv(self):
        return self.reply.copy()


class Host:
    _frontier_pose = (0.,0.,0.)
    _frontier_slam_pose = None
    _frontier_scale = 1.

    def __init__(self, rows, anchors, pending):
        self.rows,self.anchors,self.pending = rows,anchors,pending

    def _reported_views_for_planning(self):
        return self.anchors

    def _unreported_points_for_planning(self):
        return self.pending

    def _view_profiles(self, plans):
        return self.rows


def ranked(plans=None, anchors=((.50,0.,0.),), pending=(), gains=(0.,0.),
           allowed=(True,True), lengths=None, scale=1., values=(.64,.56)):
    plans = plans or [[1,1],[2,2,2,1,1,1,1]]
    lengths = lengths or [len(p) for p in plans]
    rows = [dict(allowed=a,view_steps=t,new_view_fraction=g) for a,t,g in zip(allowed,lengths,gains)]
    host = Host(rows,anchors,pending)
    host._frontier_scale=scale
    memory = np.asarray(values)
    supply = np.linspace(.12,.32,len(memory))
    bonus = .2*supply
    reply = dict(index=int((memory+bonus).argmax()),final_scores=(memory+bonus).tolist(),memory_scores=values,
                 clue_deficit=1.,exploration_bonus=bonus.tolist(),unvisited_fraction=supply.tolist(),adaptive_weight=.2,
                 worker_marker="preserved")
    worker = Worker(reply)
    view = ViewChannel(worker,host)
    channel = MemoryChannel(view,host)
    channel.send(dict(plans=plans,step=17))
    out = channel.recv()
    assert channel.last_choice==view.last_choice==(17,out["index"],rows)
    assert out["worker_marker"]=="preserved"
    return out


class ReportedViewTests(unittest.TestCase):
    def test_repeated_report_view_yields_to_other_reachable_branch(self):
        r = ranked()
        self.assertEqual((r["pre_report_memory_index"],r["index"]),(0,1))
        self.assertEqual(r["report_view_penalties"],[MAX_PENALTY,0.])
        self.assertEqual((r["no_evidence_index"],r["no_exploration_index"]),(1,1))

    def test_dominant_evidence_still_wins_and_sole_choice_is_allowed(self):
        self.assertEqual(ranked(values=(.95,.40))["index"],0)
        r = ranked(allowed=(True,False))
        self.assertEqual(r["report_view_penalties"],[0.,0.])
        self.assertEqual(r["index"],0)

    def test_new_area_new_heading_or_new_visible_rays_have_no_penalty(self):
        for anchors, gains in [(((2.,0.,0.),),(0.,0.)), (((.5,0,math.pi),),(0.,0.)),
                               (((.5,0,0),),(.5,0.))]:
            with self.subTest(anchors=anchors,gains=gains):
                r=ranked(anchors=anchors,gains=gains)
                self.assertEqual(r["report_view_penalties"],[0.,0.])
        self.assertEqual(ranked(gains=(.12,0.))["report_view_penalties"][0],MAX_PENALTY*(1-.12/.25))

    def test_unreported_neighbor_and_unknown_pending_positions_preserve_rank(self):
        for pending in [((1.65,0.),),None,((math.nan,0.),)]:
            self.assertEqual(ranked(pending=pending)["report_view_penalties"],[0.,0.])
        mid = ranked(plans=[[1]*20,[2,2,2,1,1,1,1]],anchors=((5.,0,0.),),pending=((2.5,0.),))
        self.assertEqual(mid["report_view_reasons"][0],"pending_instance_near_path")

    def test_all_options_old_or_invalid_anchor_keep_decision(self):
        r = ranked(anchors=((.5,0.,0.),(0.,1.,math.pi/2)))
        self.assertEqual(r["report_view_penalties"],[0.,0.])
        self.assertEqual(r["report_view_reasons"],["no_unpenalized_alternative"]*2)
        self.assertEqual(ranked(anchors=None)["index"],0)
        self.assertEqual(ranked(anchors=((0.,math.nan,0.),))["index"],0)

    def test_prefix_endpoint_metric_scale_and_invalid_candidates(self):
        r = ranked(plans=[[1]*12,[2,2,2,1,1],[1,1,1]],lengths=(2,5,3),scale=.5,
                   anchors=((1.,0.,0.),),allowed=(True,True,False),values=(.64,.56,1.),gains=(0.,0.,0.))
        self.assertEqual(r["report_view_penalties"],[MAX_PENALTY,0.,0.])
        self.assertEqual(r["index"],1)
        self.assertEqual((r["no_evidence_index"],r["no_exploration_index"]),(1,1))


if __name__=="__main__":
    unittest.main()
