import math
import unittest
from local_search import LocalTargetTracker


def site(x, anchor_x=4., score=.5):
    return dict(site_xy=(x,0.),site_yaw=0.,anchor_xy=(anchor_x,0.),
                surface="wall",support=score,score=score)


def open_path(t):
    return dict(status="active",actions=[1]*6,reason="approach_site")


def blocked(t):
    return dict(status="blocked",actions=[],reason="site_untraversable")


class TrackingChecks(unittest.TestCase):
    row=dict(position=(2.,0.),heading=0.,actions=[1]*8)

    def test_region_before_local_then_fixed_under_rescoring(self):
        t=LocalTargetTracker()
        self.assertEqual(t.decide((0.,0.,0.),self.row,[site(1.)],open_path)["action_source"],"region")
        first=t.decide((1.5,0.,0.),self.row,[site(3.)],open_path)
        follow=t.decide((2.,0.,0.),self.row,[site(2.4,5.,.54)],open_path)
        self.assertEqual((first["goal_id"],first["site_xy"]),(follow["goal_id"],follow["site_xy"]))
        self.assertEqual(follow["age"],2)
        self.assertEqual(follow["actions"],[1]*4)
        self.assertIsNone(t.arrival((3.,0.,math.pi/2),open_path))
        self.assertIsNone(t.arrival((3.,0.,0.),blocked))
        self.assertIsNotNone(t.arrival((3.,0.,0.),lambda _:dict(status="arrived")))
        later=t.decide((2.,0.,0.),self.row,[site(3.),site(3.2,6.)],open_path,region_entered=True)
        self.assertEqual(later["anchor_xy"],(6.,0.))

    def test_blocked_goal_and_distinct_new_observation(self):
        t=LocalTargetTracker()
        t.decide((0.,0.,0.),self.row,[site(2.)],open_path,region_entered=True)
        t.decide((.5,0.,0.),self.row,[site(3.,8.,.8)],open_path)
        b=t.decide((1.,0.,0.),self.row,[site(3.,8.,.8)],open_path)
        self.assertEqual(b["released"]["event"],"stronger_new_support")
        self.assertEqual(b["goal_id"],2)
        b=t.decide((1.2,0.,0.),self.row,[],blocked)
        self.assertEqual(b["released"]["geometry"],"site_untraversable")
        self.assertEqual(b["action_source"],"region")


if __name__=="__main__":
    unittest.main()
