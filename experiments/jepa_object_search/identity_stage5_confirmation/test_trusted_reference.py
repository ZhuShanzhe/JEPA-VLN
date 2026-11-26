"""Regression of live-view selection, receipt binding, and immutable appearance storage."""
import sys
import json
import unittest
from types import SimpleNamespace as S
from unittest.mock import patch

import numpy as np
from trusted_reference import ReferenceEvidence, TrustedReferenceMixin, assess_reference


class Base:
    def _init_report_memory(self):
        self._views={}
        self._pending_report=None
        self._reported_count=0

    def _collect_views(self): pass
    def _source(self,fid,roi): return self.backend.get(fid)
    def _references(self,n): return [self._views[i] for i in n.observation_ids]
    def _support(self,fid,ref,obs,boxes): return (boxes[0],"same") if fid==22 else (None,"different")
    def _note(self,*args,**kwargs): pass
    def _category(self,obs): return "clock"
    def _boxes(self,key,rgb,cat): return []
    def _rival_claims(self,n,box,obs,boxes):
        self._support(99,object(),obs,boxes)
        return False
    def _stage_report(self,n,obs,reason):
        self._pending_report=S(instance=n.iid,step=obs.step_count,observations=list(n.observation_ids))
        return 44
    def _settle_report(self,obs):
        if self._pending_report is None or obs.step_count<=self._pending_report.step: return
        self._pending_report=None
        if obs.receipt:
            self.node.reported=True
            self._reported_count+=1
    def _report_gate(self,node,obs):
        return self._references(node)
    def _recover_report(self,node,obs):
        return self._references(node),self._report_gate(node,obs)


class Agent(TrustedReferenceMixin, Base): pass


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.a=a=Agent()
        a._init_report_memory()
        a.node=n=S(iid=1,observation_ids=[1,2,3],reported=False)
        self.obs=S(step_count=8)
        a._last_observation=self.obs
        a.instance_store=S(get_observation=lambda i:S(evidence={"pixel_norm":[500,500]}),
                           instance_for_observation=lambda i:n)
        self.im=np.ones((48,48,3),dtype=np.uint8)
        a.backend={i:(self.im,[12,12,36,36]) for i in [11,22,33]}
        a._views={i:(i*11,S(roi=(.25,.25,.75,.75))) for i in [1,2,3]}
        quality=[ReferenceEvidence("supported",.9,.99),ReferenceEvidence("supported",.6,.8),
                 ReferenceEvidence("supported",.05,.98)]
        with patch.dict(sys.modules,{"source_quality":S(coarse_category=lambda x:x)}),patch(
                "trusted_reference.assess_reference",side_effect=quality):
            a._collect_views()

    def test_rule_and_determinism(self):
        im=np.full((100,100,3),30,dtype=np.uint8)
        im[30:70,30:70]=[234,94,12]
        det=[dict(bbox=(27,27,73,73),score=.90)]
        args=(im,(27,27,73,73),(50,50),det)
        a=assess_reference(*args)
        self.assertTrue(a.eligible())
        json.dumps(dict(eligible=a.eligible(),occupancy=a.occupancy),allow_nan=False)
        self.assertEqual(a,assess_reference(*args))
        self.assertFalse(assess_reference(im,(10,10,90,90),(50,50),det).eligible())
        self.assertFalse(assess_reference(im,(10,10,90,90),(14,14),det).eligible())

    def test_route_keeps_weak_but_gate_never_sees_it(self):
        a=self.a
        a.node.observation_ids=[3]
        route,allowed=a._recover_report(a.node,self.obs)
        self.assertEqual(route,[a._views[3]])
        self.assertEqual(allowed,[])
        self.assertEqual(a._references(a.node),[])

    def _dispatch(self):
        a,n,obs=self.a,self.a.node,self.obs
        self.assertEqual([fid for fid,_ in a._references(n)],[22,11])
        boxes=[(0,0,8,8)]
        for fid,ref in reversed(a._references(n)):
            box,_=a._support(fid,ref,obs,boxes)
            if box is not None and not a._rival_claims(n,box,obs,boxes):
                return a._stage_report(n,obs,"same")
        self.fail("trusted match not dispatched")

    def test_confirmed_uses_winning_reference_after_eviction(self):
        a=self.a
        self.assertEqual(self._dispatch(),44)
        a._settle_report(S(step_count=9,receipt=True))
        self.im[:]=0
        a.backend.clear()
        a.node.observation_ids.clear()
        a._views.clear()
        [(fid,ref)]=a._references(a.node)
        self.assertEqual(fid,22)
        rgb,box=a._source(fid,ref.roi)
        self.assertFalse(rgb.flags.writeable)
        self.assertTrue((rgb==1).all())
        self.assertIsNone(a._source(fid,(0,0,1,1)))
        # A different owner has no access to the retained registered view.
        self.assertEqual(a._references(S(iid=2,observation_ids=[],reported=False)),[])

    def test_unexecuted_action_does_not_pin(self):
        self._dispatch()
        self.a._settle_report(S(step_count=9,receipt=False))
        self.assertFalse(self.a._confirmed_reference)


if __name__=="__main__":
    unittest.main()
