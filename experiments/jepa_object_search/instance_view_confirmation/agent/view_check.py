"""Offline tests of map-only view construction and deployment selection rules."""
import unittest
import numpy as np
from view_explore import ViewChannel, adjust, cone, view_profiles


class Grid:
    res=.1
    def __init__(self):
        self.geometry_observed=np.ones((81,81),bool)
        self.semantic_inspected=self.geometry_observed.copy()
        self.obstacle=np.zeros_like(self.geometry_observed)
        self.semantic_coverage_enabled=True
    def world_to_cell(self,xy):
        return tuple(np.floor(np.asarray(xy)/self.res+40).astype(int))
    def in_bounds(self,c):
        return all(0<=x<81 for x in c)
    def traversable(self,c):
        return bool(self.in_bounds(c) and self.geometry_observed[c[1],c[0]] and not self.obstacle[c[1],c[0]])


class ViewTests(unittest.TestCase):
    def test_walls_occlude_and_truncate(self):
        grid=Grid()
        grid.semantic_inspected[:,48:]=False
        grid.obstacle[:,47]=True
        self.assertTrue(all(c is None for c in cone(grid,(0,0,0),1.)))
        rows=view_profiles(grid,(0,0,0),1.,[[1]*12,[2,2,2]+[1]*9])
        self.assertEqual(rows[0]["free_steps"],2)
        self.assertEqual(rows[0]["reach_check"],"movement_leaves_traversable_map")
        self.assertTrue(rows[1]["allowed"])
        self.assertEqual(rows[1]["free_steps"],12)

    def test_new_view_and_duplicate(self):
        grid=Grid()
        grid.semantic_inspected[63:,:]=False
        rows=view_profiles(grid,(0,0,0),1.,[[1]*12,[2,2,2]+[1]*9,[1]*7])
        self.assertGreater(rows[1]["new_view_fraction"],rows[0]["new_view_fraction"])
        self.assertFalse(rows[2]["allowed"])
        self.assertEqual(view_profiles(None,None,None,[[1],[2,1]]),[])

    def test_mask_is_enforced_on_live_channel(self):
        class Fake:
            def send(self,m):self.sent=m
            def recv(self):
                return dict(final_scores=[.3,.26,10],index=2,clue_deficit=1.,exploration_bonus=[.25,.25,.25],
                            unvisited_fraction=[1.,1.,1.],adaptive_weight=.2,memory_scores=[0.,0.,0.])
        class Host:
            def _view_profiles(self,p):
                return [dict(allowed=a,new_view_fraction=g,view_steps=t) for a,g,t in
                        [(True,0.,2),(True,1.,3),(False,1.,0)]]
        fake=Fake()
        channel=ViewChannel(fake,Host())
        channel.send({"plans":[[1]*8,[2]*3+[1]*3,[3]*6],"step":7})
        self.assertEqual([len(p) for p in fake.sent["plans"]],[2,3,6])
        self.assertEqual(channel.recv()["index"],1)
        self.assertEqual(adjust(fake.recv(),[])["index"],0)


if __name__=="__main__":
    unittest.main()
