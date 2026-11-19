import unittest
from unittest import mock

from datasets.scripts.e_bike import plan_grounded_distribution as planner


def fixture():
    backgrounds = [
        {'id': f'frame_{index}', 'image': f'datasets/bg{index}.png',
         'parent_image': f'datasets/parent{index}.png', 'source_video': 'datasets/one.mp4',
         '_frame_reviews': {'composite_background': {'record_id': index}}}
        for index in range(4)
    ]
    vehicles = {
        category: [({'path': f'datasets/{category}.png'}, object())]
        for category in ('upright', 'fallen')
    }
    return backgrounds, vehicles


class GroundedDistributionTest(unittest.TestCase):
    def setUp(self):
        self.backgrounds, self.vehicles = fixture()
        self.patches = [
            mock.patch.object(planner.grounded, 'ZONES', {bg['id']: {} for bg in self.backgrounds}),
            mock.patch.object(planner.grounded, 'background_review_reference', return_value={}),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_parent_cap_is_optional_and_recorded_when_explicit(self):
        with mock.patch.object(planner.grounded, 'render_ground_state'):
            unrestricted = planner.make_plan(self.backgrounds, self.vehicles, 72, 123)
            capped = planner.make_plan(self.backgrounds, self.vehicles, 48, 123, 40)
        self.assertEqual(unrestricted['budget'], 72)
        self.assertEqual(unrestricted['allocation_evidence']['planned_by_parent_group'],
                         {'datasets/one.mp4': 72})
        self.assertIsNone(unrestricted['allocation_evidence']['per_parent_group_cap'])
        self.assertEqual(capped['budget'], 40)
        self.assertEqual(capped['allocation_evidence']['remaining_shortfall'], 8)
        self.assertEqual(capped['allocation_evidence']['per_parent_group_cap'], 40)
        context = unrestricted['design_context']
        self.assertEqual(context['task'], 'e_bike')
        self.assertEqual(context['status'], 'referenced')
        self.assertFalse(context['executable'])
        self.assertEqual(context['coverage'], 'unverified')
        self.assertEqual({item['id'] for item in context['crosses']}, {'pose_context', 'occlusion_view'})

    def test_failed_placement_does_not_abort_other_strata(self):
        def render(state, *_):
            if state['background']['id'] == 'frame_0':
                raise ValueError('no valid placement on this background')

        with mock.patch.object(planner.grounded, 'render_ground_state', side_effect=render):
            plan = planner.make_plan(self.backgrounds, self.vehicles, 24, 42)
        self.assertEqual(plan['budget'], 24)
        self.assertTrue(all(state['background']['id'] != 'frame_0' for state in plan['strata']))
        evidence = plan['allocation_evidence']
        self.assertEqual(evidence['remaining_shortfall'], 24 - plan['budget'])
        self.assertTrue(evidence['placement_shortfalls'])
        self.assertGreater(evidence['reallocated_states'], 0)
        self.assertTrue(all(row['background_id'] == 'frame_0' and row['attempts'] == 500
                            for row in evidence['placement_shortfalls']))

    def test_rejects_invalid_parent_cap(self):
        for value in (0, -1, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive integer'):
                planner.make_plan(self.backgrounds, self.vehicles, 8, 42, value)

    def test_missing_posture_does_not_block_other_posture(self):
        self.vehicles['fallen'] = []
        for budget in (1, 16):
            with self.subTest(budget=budget), mock.patch.object(planner.grounded, 'render_ground_state'):
                plan = planner.make_plan(self.backgrounds, self.vehicles, budget, 43)
                self.assertEqual(plan['budget'], budget)
                self.assertEqual(plan['allocation_evidence']['missing_postures'], ['fallen'])
                self.assertTrue(all(s['target']['vehicles'][0]['class'] == 'upright' for s in plan['strata']))

    def test_empty_only_uses_other_parents_after_cap(self):
        for index, bg in enumerate(self.backgrounds):
            bg['source_video'] = f'datasets/parent{index}.mp4'
            bg['_frame_reviews'] = {'empty_negative': {'record_id': index}}
        plan = planner.make_plan(self.backgrounds, {'upright': [], 'fallen': []}, 8, 43, 2)
        self.assertEqual(plan['budget'], 8)
        self.assertEqual(plan['allocation_evidence']['remaining_shortfall'], 0)
        self.assertEqual(plan['allocation_evidence']['missing_postures'], ['upright', 'fallen'])
        self.assertEqual(list(plan['allocation_evidence']['planned_by_parent_group'].values()), [2] * 4)

    def test_parent_cap_shortfall_moves_to_other_parent(self):
        for index, bg in enumerate(self.backgrounds):
            bg['source_video'] = 'datasets/first.mp4' if index < 3 else 'datasets/second.mp4'
        with mock.patch.object(planner.grounded, 'render_ground_state'):
            plan = planner.make_plan(self.backgrounds, self.vehicles, 8, 43, 4)
        self.assertEqual(plan['budget'], 8)
        self.assertEqual(plan['allocation_evidence']['planned_by_parent_group'],
                         {'datasets/first.mp4': 4, 'datasets/second.mp4': 4})
        self.assertGreater(plan['allocation_evidence']['reallocated_states'], 0)


if __name__ == '__main__':
    unittest.main()
