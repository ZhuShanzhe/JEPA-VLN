import unittest

import cv2
import numpy as np

from appearance import build_reference, match_reference, match_registered_view, same_image_region


def scene():
    image = np.full((240, 320, 3), 97, np.uint8)
    box = (102, 65, 162, 119)
    item = np.full((54, 60, 3), (120, 129, 140), np.uint8)
    cv2.rectangle(item, (1, 1), (58, 52), (25, 32, 55), 3)
    cv2.line(item, (3, 4), (18, 5), (236, 165, 68), 4)
    cv2.circle(item, (53,46), 4, (85, 168, 74), -1)
    image[65:119,102:162] = item
    return image, box, item


def place(frame, patch, x, y):
    h, w = patch.shape[:2]
    frame[y:y+h,x:x+w] = patch
    return frame


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.box, self.item = scene()

    def test_smooth_center_segment_frame_and_jpeg_match(self):
        self.assertLess(self.frame[79:108,118:147].std(axis=(0,1)).max(), 1)
        raw = cv2.imencode('.jpg', cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))[1].tobytes()
        ref = build_reference(raw, self.box, point_1000=(412,380))
        self.assertIsNotNone(ref)
        live = place(np.full_like(self.frame,93),self.item,170,9)
        result = match_reference(ref,live)
        self.assertTrue(result.matched, result)
        self.assertEqual(result.box,(170,9,230,63))

    def test_scaled_target_disambiguated_from_wrong_color(self):
        ref = build_reference(self.frame, self.box)
        live = np.full_like(self.frame,97)
        larger = cv2.resize(self.item, None, fx=1.6, fy=1.6)
        place(live, larger, 100, 105)
        wrong = np.empty_like(self.item)
        wrong[:] = (170, 165, 45)
        place(live, wrong, 20, 23)
        self.assertTrue(match_reference(ref,live).matched)
        self.assertFalse(match_reference(ref,np.full_like(live,140)).matched)

    def test_two_identical_instances_at_different_scales_are_ambiguous(self):
        ref = build_reference(self.frame,self.box)
        live = np.full_like(self.frame,97)
        place(live,self.item,3,3)
        place(live,cv2.resize(self.item,None,fx=1.25,fy=1.25),200,90)
        result = match_reference(ref,live)
        self.assertFalse(result.matched, result)
        self.assertEqual(result.reason,"multiple_plausible_locations")

    def test_exact_capture_locates_two_same_appearance_objects_separately(self):
        frame = self.frame.copy()
        place(frame,self.item,194,10)
        raw = cv2.imencode('.jpg',cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))[1].tobytes()
        a = build_reference(raw,self.box)
        b = build_reference(raw,(194,10,254,64))
        self.assertFalse(match_reference(a,frame).matched)
        left,right = match_registered_view(a,frame),match_registered_view(b,frame)
        self.assertTrue(left.matched and right.matched)
        self.assertFalse(same_image_region(left.box,right.box))
        self.assertTrue(same_image_region(left.box,self.box))

    def test_wrong_capture_rejects_matching_box_and_changed_content_rejected_in_place(self):
        ref = build_reference(self.frame,self.box)
        changed = place(np.full_like(self.frame,210),self.item,102,65)
        self.assertEqual(match_registered_view(ref,changed).reason,'different_capture_context')
        changed = self.frame.copy()
        changed[65:119,102:162]=120
        place(changed,self.item,208,11)
        self.assertFalse(match_registered_view(ref,changed).matched)
        self.assertFalse(same_image_region(None,self.box))
        self.assertFalse(same_image_region((1,2,3,2),self.box))

    def test_clock_and_wall_with_repeated_mark(self):
        # A repeated wall mark at the old click position is not a clock.
        canvas = np.full((240,320,3),150,np.uint8)
        shelf_mark = np.tile(np.array([[75,115,150,185]],np.uint8), (9,3))
        canvas[71:80,122:134] = shelf_mark[...,None]
        cv2.circle(canvas,(145,99),23,(32,40,60),3)
        cv2.line(canvas,(145,99),(145,83),(41,56,65),3)
        cv2.line(canvas,(145,99),(156,105),(41,56,65),3)
        ref = build_reference(canvas,(120,70,172,125),point_1000=(400,312))
        other_wall = np.full_like(canvas,150)
        other_wall[71:80,122:134] = shelf_mark[...,None]
        self.assertFalse(match_reference(ref,other_wall).matched)
        self.assertTrue(match_reference(ref,canvas).matched)

    def test_no_background_only_confirmation(self):
        # The context surrounding the candidate can have excellent texture.
        noisy = np.random.default_rng(141).integers(0,255,self.frame.shape,dtype=np.uint8)
        noisy[65:119,102:162] = 146
        self.assertIsNone(build_reference(noisy,self.box))
        self.assertIsNone(build_reference(noisy,None,point_1000=(416,370)))
        self.assertFalse(match_reference(None,noisy).matched)

    def test_native_pixel_grid_and_normalized_point_not_interchangeable(self):
        enlarged = cv2.resize(self.frame,(640,480),interpolation=cv2.INTER_NEAREST)
        larger_box = (204,130,324,238)
        a = build_reference(enlarged,larger_box,point_1000=(408,375))
        b = build_reference(self.frame,larger_box,grid_size=(640,480),point_1000=(408,375))
        self.assertTrue(np.array_equal(a.patch,b.patch))
        self.assertIsNone(build_reference(self.frame,larger_box,point_1000=(408,375)))
        self.assertIsNone(build_reference(self.frame,self.box,point_1000=(885,375)))
        self.assertIsNone(build_reference(self.frame,(10,12,5,13)))


if __name__ == "__main__":
    unittest.main()
