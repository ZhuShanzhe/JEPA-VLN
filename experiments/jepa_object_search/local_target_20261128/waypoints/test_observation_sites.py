import math
import unittest

import numpy as np

from observation_sites import project_patches, propose_sites, plan_to_site


class Grid:
    def __init__(self, scale=1., origin=(-4.5, -4.5)):
        self.res = .125 / scale
        self.origin = np.asarray(origin) / scale
        self.geometry_observed = np.ones((88, 88), dtype=bool)
        self.obstacle = np.zeros_like(self.geometry_observed)
        self.blocked = np.zeros_like(self.obstacle)

    def world_to_cell(self, xy):
        return tuple(np.floor((xy-self.origin)/self.res).astype(int))

    def in_bounds(self, c):
        return 0 <= c[0] < 88 and 0 <= c[1] < 88

    def traversable(self, cell):
        x,y = cell
        return self.in_bounds(cell) and self.geometry_observed[y,x] and not (self.blocked[y,x] or self.obstacle[y,x])


def move(pose, actions, grid, scale=1.):
    x,y,yaw = pose
    for a in actions:
        if a == 1:
            for frac in (.25,.5,.75,1.):
                assert grid.traversable(grid.world_to_cell((x+frac*.25*math.cos(yaw)/scale,
                                                           y+frac*.25*math.sin(yaw)/scale)))
            x += .25*math.cos(yaw)/scale
            y += .25*math.sin(yaw)/scale
        else:
            assert a == 2 or a == 3
            yaw += math.pi/6 if a==2 else -math.pi/6
    return x,y,yaw


class ObservationSiteTests(unittest.TestCase):
    def setUp(self):
        self.grid=Grid()
        self.patch=dict(patch_id=13, uv=[.5,.5], surface='desk', support=.63)
        self.depth=np.full((192,192),3.,dtype=np.float32)
        self.pose=(0.,0.,0.)

    def anchor(self, grid=None):
        return project_patches(grid or self.grid,self.pose,1.,[self.patch],self.depth)

    def test_axis_units_right_rays_and_rotated_camera(self):
        g=Grid(scale=2.)
        patches=[dict(self.patch,uv=(.75,.5))]
        row=project_patches(g,(.5,-.5,math.pi/2),2.,patches,self.depth)[0]
        np.testing.assert_allclose(row['anchor_xy'],(2.5,2.),atol=.02)
        self.assertEqual(row['patch_id'],13)
        self.assertEqual(row['anchor_source'],'rgbd_measured')

    def test_empty_invalid_or_floor_depth_has_no_anchor(self):
        depth=np.zeros_like(self.depth)
        self.assertEqual(project_patches(self.grid,self.pose,1.,[self.patch],depth),[])
        floor=np.full_like(depth, 1.4)
        self.assertEqual(project_patches(self.grid,self.pose,1.,[dict(self.patch,uv=(.5,.99))],floor),[])

    def test_no_depth_proxy_stops_at_unknown_and_requires_observed_wall(self):
        g=self.grid
        self.assertEqual(project_patches(g,self.pose,1.,[self.patch]),[])
        x,y=g.world_to_cell((2.,0.))
        g.obstacle[y,x]=True
        found=project_patches(g,self.pose,1.,[self.patch])
        self.assertEqual(found[0]['anchor_source'],'observed_map_column_proxy')
        self.assertIsNone(found[0]['anchor_height_m'])
        g.geometry_observed[y,x-2]=False
        self.assertEqual(project_patches(g,self.pose,1.,[self.patch]),[])
        self.assertEqual(project_patches(g,self.pose,1.,[dict(self.patch,uv=(.5,.03))]),[])

    def test_replan_multiple_legs_site_and_heading(self):
        rows=propose_sites(self.grid,self.pose,1.,self.anchor())
        self.assertGreaterEqual(len(rows),2)
        for row in rows:
            self.assertEqual(row['patch_id'],13)
            p=move(self.pose,row['actions'],self.grid)
            self.assertLessEqual(math.dist(p[:2],row['site_xy']),.18)
            self.assertLessEqual(abs(math.remainder(p[2]-row['site_yaw'],math.tau)),math.radians(15.01))
        site=rows[0]
        pose=self.pose
        for _ in range(24):
            leg=plan_to_site(self.grid,pose,1.,site,max_leg_steps=2)
            if leg['status']=='arrived':
                break
            self.assertEqual(leg['status'],'active')
            self.assertLessEqual(len(leg['actions']),2)
            pose=move(pose,leg['actions'],self.grid)
        self.assertEqual(leg['status'],'arrived')
        rotated=(pose[0],pose[1],pose[2]+math.pi)
        self.assertEqual(plan_to_site(self.grid,rotated,1.,site)['status'],'active')
        self.assertEqual(plan_to_site(self.grid,rotated,1.,site,budget=0)['status'],'budget')

    def test_metric_sites_survive_origin_and_scale_changes(self):
        site=propose_sites(self.grid,self.pose,1.,self.anchor())[0]
        g=Grid(scale=2.,origin=(-3.8,-4.7))
        pose=(.125,0.,0.)
        for _ in range(6):
            leg=plan_to_site(g,pose,2.,site,max_leg_steps=4)
            if leg['status']=='arrived':
                break
            self.assertTrue(leg['actions'])
            pose=move(pose,leg['actions'],g,2.)
        self.assertEqual(leg['status'],'arrived')
        self.assertLessEqual(math.dist([pose[0]*2,pose[1]*2],site['site_xy']),.18)

    def test_short_detour_and_loss_of_sight(self):
        g=self.grid
        site=propose_sites(g,self.pose,1.,self.anchor())[0]
        x,y=g.world_to_cell((.70,0.))
        g.blocked[y-1:y+2,x]=True
        pose=self.pose
        for _ in range(10):
            leg=plan_to_site(g,pose,1.,site,max_leg_steps=4)
            if leg['status']=='arrived':
                break
            self.assertEqual(leg['status'],'active',leg)
            pose=move(pose,leg['actions'],g)
        self.assertEqual(leg['status'],'arrived')
        x,y=g.world_to_cell((2.5,0.))
        g.obstacle[y-1:y+2,x]=True
        result=plan_to_site(g,pose,1.,site)
        self.assertEqual(result['reason'],'sightline_lost')
        self.assertFalse(result['actions'])

    def test_no_stand_off_in_unseen_area(self):
        self.grid.geometry_observed[:]=False
        self.assertFalse(propose_sites(self.grid,self.pose,1.,self.anchor()))


if __name__=='__main__':
    unittest.main()
