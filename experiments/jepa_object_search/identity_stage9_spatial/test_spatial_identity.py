"""Metric place association exercises, independent of Habitat and the action controller."""
from dataclasses import replace
import unittest

import numpy as np

from spatial_identity import Anchor, Surface, associate, make_anchor


def wall_camera(camera_x=0.,object_x=0.,color=(31,31,31)):
    # Two displaced parallel cameras looking at the same wall at z=2 metres.
    size,stride,f=128,2,110.
    pose=np.eye(4)
    pose[0,3]=camera_x
    K=np.array([[f,0,size/2],[0,f,size/2],[0,0,1.]])
    yy,xx=np.mgrid[:size:stride,:size:stride]+.5
    grid=np.dstack([(xx-size/2)*2/f+camera_x,(yy-size/2)*2/f,np.full_like(xx,2.)])
    center_x=size/2+(object_x-camera_x)*f/2
    box=np.array([center_x-9,55,center_x+9,73])
    rgb=np.full((size,size,3),180,np.uint8)
    x,y,u,v=box.astype(int)
    rgb[y:v,x:u]=color
    view=dict(K=K,grid=grid,c2w=pose,stride=stride,frame_id=round(100*camera_x)+100)
    return rgb,view,box


def anchor_at(camera,object_x,**kwargs):
    rgb,view,box=wall_camera(camera,object_x,**kwargs)
    return make_anchor(view,rgb,box)


class SpatialTest(unittest.TestCase):
    def setUp(self):
        self.p=anchor_at(0.,0.)
        self.q=anchor_at(.3,0.)

    def test_registered_wall_position_is_view_invariant(self):
        np.testing.assert_allclose(self.p.center,self.q.center,atol=1e-8)
        self.assertAlmostEqual(self.p.center[2],2.)
        self.assertGreater(np.linalg.norm(np.subtract(self.p.camera,self.q.camera)),.25)
        self.assertIsNotNone(self.q.surface)
        self.assertLess(self.q.surface.normal[2],-.99)
        self.assertEqual(associate(self.q,{9:[self.p]}).owner,9)

    def test_strong_visual_cannot_override_other_wall_position(self):
        b=anchor_at(.3,.34)
        self.assertEqual(associate(b,{1:[self.p]},appearances={1:'strong'}).status,'distinct')

    def test_adjacent_plausible_owners_require_new_view(self):
        near=anchor_at(-.25,.12)
        r=associate(self.q,{2:[self.p],3:[near]},appearances={2:'strong'})
        self.assertEqual(r.reason,'multiple_plausible_owners')
        self.assertIsNone(r.owner)

    def test_grounded_far_neighbor_does_not_veto_a_unique_same_instance(self):
        far=anchor_at(-.25,.62)
        r=associate(self.q,{2:[self.p],3:[far]})
        self.assertEqual((r.status,r.owner),('same',2))

    def test_proven_coview_exclusion_cannot_be_revived_by_appearance(self):
        r=associate(self.q,{2:[self.p]},appearances={2:'strong'},excluded={2})
        self.assertEqual(r.status,'distinct')

    def test_discriminative_foreground_color_can_veto_weak_identity(self):
        a=anchor_at(0.,0.,color=(215,31,12))
        b=anchor_at(.3,0.,color=(24,81,220))
        self.assertIsNotNone(a.hue)
        self.assertIsNotNone(b.hue)
        self.assertEqual(associate(b,{1:[a]}).status,'uncertain')
        self.assertIsNone(self.q.hue)  # a dark generic clock supplies no color vote

    def test_measured_shape_is_only_a_veto_from_complete_frontal_views(self):
        rgb,view,box=wall_camera(0.,0.)
        rgb[:]=180
        box=np.array([53,52,75,76])
        rgb[52:76,53:75]=31
        wide=make_anchor(view,rgb,box)
        self.assertIsNotNone(wide.dimensions)
        rgb,view,box=wall_camera(.3,0.)
        rgb[:]=180
        box=np.array([35.5,37,59.5,91])
        rgb[37:91,35:60]=31
        tall=make_anchor(view,rgb,box)
        self.assertIsNotNone(tall.dimensions)
        self.assertEqual(associate(tall,{2:[wide]},appearances={2:'strong'}).status,'uncertain')
        self.assertEqual(associate(tall,{2:[replace(wide,dimensions=None)]}).status,'same')
        self.assertEqual(associate(tall,{2:[wide,replace(wide,dimensions=None)]}).status,'uncertain')

    def test_crop_and_oblique_view_do_not_generate_shape_evidence(self):
        rgb,view,box=wall_camera(0.,0.)
        rgb[:]=180
        rgb[44:82,0:22]=31
        edge=make_anchor(view,rgb,[0,44,22,82])
        self.assertTrue(edge.clipped)
        self.assertIsNone(edge.dimensions)
        rgb[:]=180
        rgb[45:85,75:82]=31  # detector sees only a sliver in the proposed box
        partial=make_anchor(view,rgb,[60,45,82,85])
        self.assertIsNone(partial.dimensions)
        rgb,view,box=wall_camera(0.,0.)
        yy,xx=np.mgrid[:128:2,:128:2]+.5
        ray_x,ray_y=(xx-64)/110.,(yy-64)/110.
        depth=2/(1+.95*ray_x)
        view['grid']=np.dstack([depth*ray_x,depth*ray_y,depth])
        oblique=make_anchor(view,rgb,box)
        self.assertIsNotNone(oblique)
        self.assertIsNone(oblique.dimensions)

    def test_surrounding_wall_color_is_not_an_intrinsic_attribute(self):
        rgb,view,box=wall_camera(0,0)
        rgb[:,:,:]=(180,10,14)
        self.assertIsNone(make_anchor(view,rgb,box).hue)

    def test_missing_surface_cannot_be_replaced_with_similarity(self):
        self.assertEqual(associate(replace(self.q,surface=None),{2:[self.p]},
                                   appearances={2:'strong'}).status,'uncertain')

    def test_other_side_of_thin_wall(self):
        s=Surface(self.q.surface.point,(0.,0.,1.))
        self.assertEqual(associate(replace(self.q,surface=s),{2:[self.p]}).status,'uncertain')

    def test_wall_and_table_observation_is_not_the_same_support(self):
        horizontal=Surface(self.q.center,(0.,-1.,0.))
        self.assertEqual(associate(replace(self.q,surface=horizontal),{2:[self.p]}).status,'uncertain')
        old_table=replace(self.p,surface=horizontal)
        new_table=replace(self.q,surface=horizontal)
        self.assertEqual(associate(new_table,{1:[old_table]}).status,'same')
        neighbor=replace(new_table,center=(.42,0.,2.))
        self.assertEqual(associate(neighbor,{1:[old_table]}).status,'distinct')

    def test_same_frame_rotating_camera_or_uncertain_depth_can_never_self_confirm(self):
        for b in [replace(self.q,frame_id=self.p.frame_id),replace(self.q,camera=self.p.camera),
                  replace(self.q,sigma=.25)]:
            self.assertEqual(associate(b,{1:[self.p]}).status,'uncertain')

    def test_input_from_scaled_rgb_uses_native_map_grid(self):
        rgb,view,box=wall_camera(0.,0.)
        big=rgb.repeat(2,0).repeat(2,1)
        np.testing.assert_allclose(make_anchor(view,big,box*2).center,self.p.center)

    def test_missing_depth_and_invalid_detector_box(self):
        rgb,view,box=wall_camera(0.,0.)
        for invalid in ([0,0,132,120], [0,0,2,2], [0,np.nan,80,92]):
            self.assertIsNone(make_anchor(view,rgb,invalid))
        view['grid'][:]=np.nan
        self.assertIsNone(make_anchor(view,rgb,box))
        self.assertEqual(associate(None,{3:[self.p]}).status,'uncertain')

    def test_inconsistent_or_unknown_rival_memory_prevents_a_merge(self):
        far=anchor_at(-.25,.62)
        for memories in [{1:[self.p,far]},{1:[self.p],2:[]}]:
            self.assertEqual(associate(self.q,memories).status,'uncertain')


if __name__=='__main__':
    unittest.main()
