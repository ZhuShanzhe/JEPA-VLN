import unittest
from types import SimpleNamespace

from identity import GuardedReplayMixin, relation


class IdentityTests(unittest.TestCase):
    def test_live_and_archived_native_record(self):
        live = SimpleNamespace(oid=9, frame_id=12, candidate_id="c2", bbox=None)
        self.assertEqual(relation([live], [{"observation_id": 9}]), "same")
        self.assertEqual(relation([live], [{"candidate_id": "c2"}]), "same")

    def test_separate_instances_survive_same_image_and_same_category(self):
        a = {"frame_id": 5, "bbox": [12, 20, 60, 70], "point": [1, 1, 0], "text": "painting"}
        b = dict(a, bbox=[65, 20, 113, 70])
        self.assertEqual(relation([a], [b]), "distinct")
        self.assertEqual(relation([a], [dict(b, frame_id=6)]), "unknown")
        self.assertEqual(relation([a], [dict(a, frame_id=6)]), "unknown")

    def test_explicit_separation_overrides_replay_handle_and_iou(self):
        a = {"frame_id": 1, "bbox": [0, 0, 20, 20], "candidate_id": "c1"}
        other = dict(a, bbox=[21, 0, 40, 20])
        self.assertEqual(relation([a], [a, other]), "distinct")
        self.assertEqual(relation([a], [a], 0, "1"), "distinct")
        self.assertEqual(relation([], [], 0, 0), "unknown")

    def test_only_near_identical_boxes_at_same_camera_are_sufficient(self):
        a = {"frame_id": 0, "bbox": [0, 0, 100, 100]}
        self.assertEqual(relation([a], [dict(a, bbox=[1, 1, 100, 100])]), "same")
        self.assertEqual(relation([a], [dict(a, bbox=[0, 0, 70, 100])]), "unknown")
        self.assertEqual(relation([a], [dict(a, frame_id=None)]), "unknown")

    def test_missing_invalid_and_placeholder_records_do_not_match(self):
        self.assertEqual(relation(None, [None]), "unknown")
        for raw in ([], {}, [2, 0, 2, 20], [-1, 0, 20, 20], [0, 0, 2], [0, 0, float("nan"), 2]):
            bad = {"oid": -1, "candidate_id": " ", "frame_id": 7, "bbox": raw}
            self.assertEqual(relation([bad], [bad]), "unknown")
        self.assertEqual(relation([{"oid": True}], [{"oid": 1}]), "unknown")


class FixtureStore(GuardedReplayMixin):
    exact_bbox_iou = .90

    def __init__(self):
        self.observations = {}
        self.owners = {}
        self._candidate_to_observation = {}

    def observe(self, oid, frame, bbox, handle, owner):
        row = SimpleNamespace(oid=oid, frame_id=frame, bbox=bbox, candidate_id=handle,
                              pixel=(7, 4))
        self.observations[oid] = row
        self._candidate_to_observation[handle] = oid
        self.owners[oid] = owner
        return row

    def get_observation(self, oid):
        return self.observations.get(oid)

    def instance_for_observation(self, oid):
        iid = self.owners[oid]
        return None if iid is None else SimpleNamespace(iid=iid)


class GuardedReplayTests(unittest.TestCase):
    def test_nearby_centers_do_not_discard_new_target_and_search_all_rows(self):
        store = FixtureStore()
        store.observe(1, 7, [1, 1, 6, 7], "c1", 14)
        obj_b = dict(candidate_id="c2", frame_id=7, pixel=(7, 4), bbox=[7, 1, 12, 7])
        self.assertIsNone(store._find_replay_observation(**obj_b))
        second = store.observe(2, 7, obj_b["bbox"], "c2", 15)
        self.assertIs(store._find_replay_observation(**dict(obj_b, candidate_id="c3")), second)
        self.assertIsNone(store._find_replay_observation(**dict(obj_b, candidate_id="c3", bbox=None)))

    def test_frame_change_and_conflicting_owners(self):
        store = FixtureStore()
        a = store.observe(1, 7, [1, 1, 12, 12], "c1", 14)
        self.assertIs(store._find_replay_observation(candidate_id="c1"), a)
        self.assertIsNone(store._find_replay_observation(candidate_id="c1", frame_id=7, bbox=[14, 2, 21, 11]))
        self.assertIsNone(store._find_replay_observation(candidate_id="c2", frame_id=8, bbox=a.bbox))
        store.observe(2, 7, a.bbox, "c2", 17)
        self.assertIsNone(store._find_replay_observation(candidate_id="c3", frame_id=7, bbox=a.bbox))
        self.assertIs(store._find_replay_observation(candidate_id="c1", frame_id=7, bbox=a.bbox), a)


if __name__ == "__main__":
    unittest.main()
