"""Exercise the production MOST report adapter using simulated RGB, poses and actions."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from verify_rank import deny_ip_sockets

deny_ip_sockets()

import cv2
import numpy as np

from agents.instance_store import InstanceMemory
from agents.nav_agent import NavAgent
from benchmark_api import Action
from confirm_agent import ConfirmFrontier
from exploration_memory import MemoryChannel
from identity import guarded_instance_memory
from report_agent import ReportFrontier
from view_explore import ViewChannel


REPORT = int(Action.TARGET_FOUND)
TURN = int(Action.TURN_LEFT)


class ReportChainTests(unittest.TestCase):
    def setUp(self):
        # All scene content is generated locally. There is no live mapper or VLM.
        self.rgb = np.full((320, 320, 3), 36, dtype=np.uint8)
        rng = np.random.default_rng(93)
        pigment = rng.integers(20, 235, (24, 24, 3), dtype=np.uint8)
        self.rgb[80:240, 80:240] = cv2.resize(pigment, (160, 160), interpolation=cv2.INTER_CUBIC)
        _, image = cv2.imencode('.jpg', cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))
        self.jpeg = image.tobytes()
        self.agent = self.make_agent(ReportFrontier)
        self.first = self.register(self.agent, 3, (0., .5, .5))
        self.agent.target_instance_id = self.first.iid
        self.agent._report_gate = Mock(return_value=(True, 'synthetic_close_current_view'))

    def make_agent(self, agent_type):
        agent = agent_type.__new__(agent_type)
        agent._views, agent._retry_after, agent._attempts, agent._endorsed = {}, {}, {}, set()
        agent._confirm_pending = agent._confirm_frame = None
        agent._confirm_orient = 0
        agent._unreachable_instance_ids = set()
        agent._arrival_transition_active = False
        agent.instance_store = guarded_instance_memory() if isinstance(agent, ReportFrontier) else InstanceMemory()
        agent._reported_count = 0
        agent._no_hit_queries = 2
        agent._instance_goal_index, agent._goal_found = {}, set()
        agent._scanning = False
        agent.scan = agent.route = agent.follower = None
        agent.mode = 'nav'
        agent.target_instance_id = None
        agent.report_near_dist_m = 1.
        agent.now = 20
        agent._last_observation = self.observation(20, None)
        agent._log_event, agent._note, agent._stamp_pending_outcome = Mock(), Mock(), Mock()
        agent._return = Mock(return_value=None)
        agent._clear_current_target = Mock()
        agent._geometry = lambda *args: (.1, .5, .5, 0.)
        agent._instance_distance_m = lambda node: .5
        agent.align_R = np.eye(3)
        agent._last_feed_info = {'frame_id': 3}
        agent._estimated_current_pose = lambda: (0., 0., 0.)
        agent._metric_scale_value = lambda: 1.
        agent._frontier_scale = 1.
        agent._proposal_queue, agent._dup_ledger = {}, {}
        agent._bind_goal_index = Mock()
        agent._record_registered_spot = Mock()
        agent.client = SimpleNamespace(get_frame_pose=lambda f: np.eye(4),
                                       get_frame_image=lambda f: ({'found': True}, self.jpeg))
        if isinstance(agent, ReportFrontier):
            agent._init_report_memory()
        return agent

    @staticmethod
    def register(agent, fid, point):
        rec = agent.instance_store.new_observation(point, 'framed wall painting',
            {'pixel_norm': [500, 500], 'bbox': [80, 80, 240, 240]}, frame_id=fid, step=7,
            pixel=(500, 500), bbox=(80, 80, 240, 240))
        return agent.instance_store.create_instance(rec)

    def observation(self, step, action):
        return SimpleNamespace(step_count=step, previous_action=action, rgb=self.rgb.copy(),
            episode_id='synthetic-room', target_mode='many', target_count=2,
            goal_text='Report two distinct paintings.')

    def feed(self, step, action):
        obs = self.observation(step, action)
        # Exercise the real entry hook. The parent frame delivery would normally
        # contact the mapper; only that unrelated delivery is replaced.
        with patch.object(ConfirmFrontier, '_feed_frame', return_value=None):
            self.agent._feed_frame(obs)
        self.agent.now = step
        self.agent._last_observation = obs

    def test_canonical_memory_changes_after_observed_report_only(self):
        a = self.agent
        a._instance_goal_index[self.first.iid] = 1
        self.assertEqual(a._report_found(self.first.iid), REPORT)
        self.assertFalse(self.first.reported)
        self.assertEqual(a._reported_count, 0)
        self.assertNotIn(1, a._goal_found)
        self.feed(21, REPORT)
        self.assertTrue(self.first.reported)
        self.assertEqual((len(a.instance_store.report_claims), a._reported_count), (1, 1))
        self.assertIn(1, a._goal_found)
        self.feed(21, REPORT)
        self.assertEqual(a._reported_count, 1)
        self.assertNotEqual(a._report_found(self.first.iid), REPORT)
        self.assertEqual(len(a.instance_store.report_claims), 1)

    def test_replaced_action_does_not_poison_later_attempt(self):
        a = self.agent
        self.assertEqual(a._report_found(self.first.iid), REPORT)
        self.feed(21, TURN)
        self.assertFalse(self.first.reported)
        self.assertIsNone(a._pending_report)
        self.assertEqual(a._reported_count, 0)
        self.assertEqual(a._report_found(self.first.iid), REPORT)
        self.feed(22, REPORT)
        self.assertEqual(a._reported_count, 1)

    def test_out_of_range_lead_remains_available_then_reports(self):
        a = self.agent
        a._report_gate.side_effect = [(False, 'synthetic_far_view'), (True, 'synthetic_close_view')]
        self.assertNotEqual(a._report_found(self.first.iid), REPORT)
        self.assertIn(self.first, a.instance_store.available())
        self.assertIsNone(a._pending_report)
        self.feed(21, TURN)
        self.assertEqual(a._report_found(self.first.iid), REPORT)
        self.feed(22, REPORT)
        self.assertTrue(self.first.reported)

    def test_neighbor_same_appearance_remains_reportable(self):
        a = self.agent
        # Centroids only 18 cm apart. Identical class and RGB are insufficient
        # reason to invalidate a separate canonical instance.
        second = self.register(a, 4, (.18, .5, .5))
        a.target_instance_id = second.iid
        self.assertEqual(a._report_found(self.first.iid), REPORT)
        self.feed(21, REPORT)
        self.assertEqual(a.target_instance_id, second.iid)
        self.assertNotIn(self.first, a.instance_store.available())
        self.assertIn(second, a.instance_store.available())
        self.assertEqual([r['id'] for r in a._tool_search_instances('painting', reported=False)], [second.iid])
        self.assertNotEqual(a._report_found(self.first.iid), REPORT)
        self.assertEqual(a._report_found(second.iid), REPORT)
        self.feed(22, REPORT)
        self.assertEqual({c.instance_id for c in a.instance_store.report_claims}, {self.first.iid, second.iid})
        self.assertEqual(a._reported_count, 2)

    def test_reported_alias_cannot_spend_another_report(self):
        a = self.agent
        self.assertEqual(a._report_found(self.first.iid), REPORT)
        self.feed(21, REPORT)
        # Duplicate registration, same mapper keyframe and same segmentation box.
        alias = self.register(a, 3, (0.02, .5, .5))
        self.assertNotEqual(self.first.iid, alias.iid)
        self.assertNotEqual(a._report_found(alias.iid), REPORT)
        self.assertEqual(len(a.instance_store.report_claims), 1)
        self.assertEqual(a._reported_count, 1)

    def test_decider_report_dispatch_uses_the_same_acknowledgement(self):
        a = self.agent
        result = SimpleNamespace(action='REPORT_FOUND', target_id=self.first.iid)
        self.assertEqual(a._execute_decider_result(a._last_observation, result, 'arrival'), REPORT)
        self.assertFalse(self.first.reported)
        self.feed(21, REPORT)
        self.assertTrue(self.first.reported)

    def test_public_act_records_claim_only_on_next_observation(self):
        a = self.agent
        del a._report_gate
        a.episode, a.observed_step = 'synthetic-room', 19
        a.vlm, a._paper_steps, a._metric_snapshot = object(), [], {}
        a._adjusting = a._last_motion_failed = a._metric_replan_required = False
        a._finish_check_action = None
        a.pose_free, a.scale_source = True, 'fixture'
        for name in ('_process_queued_picks','_settle_action_outcomes','_capture_goal_images',
                     '_ensure_caption_task','_capture_pool_world_anchor','_record_and_update','_write_probe'):
            setattr(a, name, Mock())
        a._should_finish = Mock(return_value=False)
        a._choose_high_level_target = Mock(return_value=TURN)
        with patch.object(ConfirmFrontier, '_feed_frame', return_value=None):
            for step, previous, expected, count in ((20, TURN, REPORT, 0), (21, REPORT, TURN, 1)):
                obs = self.observation(step, previous)
                obs.max_steps = 300
                self.assertEqual(a.act(obs), expected)
                self.assertEqual(a._paper_steps[-1]['action'], expected)
                self.assertEqual(a._paper_steps[-1]['reported_count'], count)
                self.assertEqual(self.first.reported, count == 1)

    def test_disjoint_boxes_in_same_view_protect_new_instance(self):
        a = self.agent
        self.assertEqual(a._report_found(self.first.iid), REPORT)
        self.feed(21, REPORT)
        rec = a.instance_store.new_observation((.18, .5, .5), 'framed wall painting',
            {'pixel_norm': [900, 500]}, frame_id=3, step=21,
            pixel=(900, 500), bbox=(267, 115, 311, 145))
        refused = a._tool_resolve_duplicate(rec.oid, 'DUPLICATE', duplicate_of=self.first.iid)
        self.assertIn('error', refused)
        self.assertIsNone(a.instance_store.instance_for_observation(rec.oid))
        allowed = a._tool_resolve_duplicate(rec.oid, 'NEW')
        self.assertIn('instance_id', allowed)
        self.assertNotEqual(allowed['instance_id'], self.first.iid)
        self.assertNotIn('merged', a._tool_merge_instances(self.first.iid, allowed['instance_id']))
        self.assertEqual(a._report_found(allowed['instance_id']), REPORT)
        self.feed(22, REPORT)
        self.assertEqual(a._reported_count, 2)

    def test_real_gate_visual_reacquisition_during_navigation(self):
        a = self.agent
        del a._report_gate
        a._last_observation.rgb[:] = 36
        self.assertNotEqual(a._report_found(self.first.iid), REPORT)
        self.assertIn(self.first, a.instance_store.available())
        self.feed(21, TURN)
        # The VLM does not submit a second REPORT_FOUND. On a new frame the
        # ordinary movement entry re-examines the saved object and dispatches it.
        action, _, _ = a._nav_action(a._last_observation)
        self.assertEqual(action, REPORT)
        self.assertFalse(self.first.reported)
        self.feed(22, REPORT)
        self.assertTrue(self.first.reported)
        self.assertEqual(len(a._reported_views_for_planning()), 1)

    def test_goto_reported_instance_keeps_other_active_target(self):
        a = self.agent
        second = self.register(a, 4, (.18, .5, .5))
        a.target_instance_id = second.iid
        self.assertEqual(a._report_found(self.first.iid), REPORT)
        self.feed(21, REPORT)
        stale = SimpleNamespace(action='GOTO_INSTANCE', target_id=self.first.iid)
        self.assertNotEqual(a._execute_decider_result(a._last_observation, stale, 'explore'), REPORT)
        self.assertEqual(a.target_instance_id, second.iid)
        a._clear_current_target.assert_not_called()

    def test_unrelated_observed_report_creates_no_claim(self):
        self.feed(21, REPORT)
        self.assertFalse(self.first.reported)
        self.assertEqual(self.agent._reported_count, 0)

    def test_acknowledged_report_changes_real_route_ranking_and_spares_new_clue(self):
        a = self.agent
        plans = [[1,1],[2,2,2,1,1,1,1]]
        a._frontier_pose = (-.5, 0., 0.)
        a._view_profiles = lambda paths: [dict(allowed=True, view_steps=len(p), new_view_fraction=0.) for p in paths]
        worker = Mock()
        worker.recv.return_value = dict(index=0, final_scores=[.64,.56], memory_scores=[.64,.56],
            clue_deficit=1., exploration_bonus=[0.,0.], unvisited_fraction=[0.,0.], adaptive_weight=.2)
        channel = MemoryChannel(ViewChannel(worker, a), a)
        def select():
            channel.send(dict(plans=plans, step=a.now))
            return channel.recv()
        self.assertEqual(a._report_found(self.first.iid), REPORT)
        before = select()
        self.assertEqual(before['index'], 0)
        self.assertEqual(before['report_view_penalties'], [0.,0.])
        self.assertEqual(a._reported_views_for_planning(), [])
        self.feed(21, REPORT)
        self.assertEqual(len(a._reported_views_for_planning()), 1)
        self.assertEqual(a._unreported_points_for_planning(), [])
        after = select()
        self.assertEqual((after['pre_report_memory_index'],after['index']), (0,1))
        self.assertGreater(after['report_view_penalties'][0], 0.)
        self.assertEqual(after['report_view_penalties'][1], 0.)
        self.register(a, 8, (.20, 0., .5))
        protected = select()
        self.assertEqual(protected['index'], 0)
        self.assertEqual(protected['report_view_penalties'], [0.,0.])
        self.assertEqual(protected['report_view_reasons'][0], 'pending_instance_near_path')

    def test_normal_reset_installs_guarded_registration(self):
        a = self.agent
        with patch.object(ConfirmFrontier, '_nav_reset_state', return_value=None):
            a._nav_reset_state()
        store = a.instance_store
        self.assertEqual(store.available(), [])
        rec = store.new_observation((0., 1., .5), 'painting', frame_id=3,
            pixel=(500,500), bbox=(154,154,158,166), candidate_id='synthetic-A')
        first = store.create_instance(rec)
        self.assertIsNone(a._registered_spot_at(3,(505,500)))
        self.assertIsNone(store.find_replay(frame_id=3,pixel=(505,500),bbox=(159,154,163,166)))
        self.assertIs(store.find_replay(candidate_id='synthetic-A',frame_id=3,bbox=rec.bbox), first)

    def test_skipped_step_cannot_confirm_an_old_intent(self):
        self.assertEqual(self.agent._report_found(self.first.iid), REPORT)
        self.feed(23, REPORT)
        self.assertEqual(self.agent._reported_count, 0)
        self.assertFalse(self.first.reported)

    def test_original_and_adapter_compared_on_interrupted_dispatch(self):
        native = self.make_agent(NavAgent)
        node = self.register(native, 3, (0., .5, .5))
        native.target_instance_id = node.iid
        self.assertEqual(native._report_found(node.iid), REPORT)
        self.assertEqual(self.agent._report_found(self.first.iid), REPORT)
        self.feed(21, TURN)
        self.assertTrue(node.reported)
        self.assertFalse(self.first.reported)
        self.assertEqual((len(native.instance_store.report_claims), len(self.agent.instance_store.report_claims)), (1, 0))

    def test_same_scripted_reports_scored_outside_both_policies(self):
        # Both receive the same four REPORT requests, target IDs, RGBs and visit
        # sequence. The fixture fixes the poses rather than following each turn.
        # Ground-truth IDs for scoring are never placed in the observation.
        def play(kind):
            a = self.make_agent(kind)
            first = self.register(a, 3, (0., .5, .5))
            sibling = self.register(a, 4, (.18, .5, .5))
            sibling_rgb = np.ascontiguousarray(np.flip(self.rgb, axis=1))
            _, enc = cv2.imencode('.jpg', cv2.cvtColor(sibling_rgb, cv2.COLOR_RGB2BGR))
            images = {3: self.jpeg, 4: enc.tobytes()}
            a.client.get_frame_image = lambda f: ({'found': True}, images[f])
            seen = set()
            issued = []
            previous = TURN
            for i, (node, distance, truth) in enumerate(((first, 2.4, None), (first, .45, 'one'),
                                                        (first, .45, 'one'), (sibling, .45, 'two'))):
                step = 20 + i
                obs = self.observation(step, previous)
                obs.rgb = (self.rgb if node is first else sibling_rgb).copy()
                a._last_feed_info = {'frame_id': 30+i, 'is_keyframe': True}
                if kind is ReportFrontier:
                    with patch.object(ConfirmFrontier, '_feed_frame', return_value=None):
                        a._feed_frame(obs)
                a.now, a._last_observation = step, obs
                a.target_instance_id = node.iid
                a._instance_distance_m = lambda _, dist=distance: dist
                a._geometry = lambda *args, dist=distance: (.1, .5, dist, 0.)
                request = SimpleNamespace(action='REPORT_FOUND', target_id=node.iid)
                previous = a._execute_decider_result(obs, request, 'arrival')
                if previous == REPORT:
                    issued.append(i)
                    if truth is not None:
                        seen.add(truth)
            return seen, issued

        self.assertEqual(play(NavAgent), ({'two'}, [0, 3]))
        self.assertEqual(play(ReportFrontier), ({'one','two'}, [1, 3]))

    def test_identical_nearby_instances_require_unambiguous_live_regions(self):
        a = self.agent
        del a._report_gate
        a.instance_store = guarded_instance_memory()
        tile = cv2.resize(self.rgb[80:240,80:240], (64,64))
        old = np.full_like(self.rgb, 36)
        old[90:154,25:89] = tile
        both = old.copy()
        both[90:154,198:262] = tile
        frames = {11:old,12:old,21:both}
        def frame_image(fid):
            _, encoded = cv2.imencode('.jpg', cv2.cvtColor(frames[fid], cv2.COLOR_RGB2BGR))
            return {'found': True}, encoded.tobytes()
        a.client.get_frame_image = frame_image
        def register_box(fid, left, node=None):
            center = [(left+32)*1000/320, 122*1000/320]
            rec = a.instance_store.new_observation((left*.001, .5, .5), 'identical framed painting',
                {'pixel_norm':center},frame_id=fid,pixel=center,bbox=(left,90,left+64,154))
            if node is None:
                return a.instance_store.create_instance(rec)
            a.instance_store.attach_observation(node, rec)
            return node
        first, second = register_box(11,25), register_box(12,25)
        a._last_feed_info = {'frame_id':11,'is_keyframe':True}
        a._last_observation.rgb = old.copy()
        self.assertEqual(a._report_found(first.iid), REPORT)
        self.feed(21, REPORT)
        a._last_feed_info = {'frame_id':20,'is_keyframe':True}
        a._last_observation.rgb = old.copy()
        self.assertEqual(a._report_gate(second,a._last_observation),(False,'other_instance_explains_same_live_region'))
        self.assertNotEqual(a._report_found(second.iid), REPORT)
        self.assertFalse(second.reported)
        self.feed(22, TURN)
        register_box(21,25,first)
        register_box(21,198,second)
        a._last_feed_info = {'frame_id':21,'is_keyframe':True}
        a._last_observation.rgb = both.copy()
        self.assertEqual(a._report_found(second.iid), REPORT)
        self.feed(23, REPORT)
        self.assertEqual([c.instance_id for c in a.instance_store.report_claims], [first.iid,second.iid])

    def test_new_episode_clears_waiting_transaction(self):
        self.assertEqual(self.agent._report_found(self.first.iid), REPORT)
        with patch.object(ConfirmFrontier, 'init_episode', return_value=None):
            self.agent.init_episode(self.observation(0, None))
        self.assertIsNone(self.agent._pending_report)
        self.feed(21, REPORT)
        self.assertEqual(self.agent._reported_count, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
