"""Budget and infeasible-state behavior for sign plan compilation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from datasets.scripts.sign import plan_sign_sources as planner


class SignPlanCapacityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.annotated = self.root / 'foreground'
        self.annotated.mkdir()
        (self.annotated / 'manifest.json').write_text(json.dumps({'records': [
            {'index': 20, 'source': 'front.png'}, {'index': 14, 'source': 'side.png'}]}), encoding='utf8')
        self.frames = []
        for index in range(5):
            self.frames.append({'image': f'background_{index}.png', 'source_group': f'parent_{index}',
                                'support_rectangle_xyxy': [0, 300, 300, 450],
                                'protected_rectangles_xyxy': [], 'frame_id': index,
                                'record_id': index, 'evidence_ids': [], 'placement_algorithm': 'fixture'})

    def compile(self, budget, contact=None, requested_budget=None):
        def contact_for(frame, *_args, **_kwargs):
            return 'unavailable' if contact and frame['source_group'] == contact else 'candidate'

        with patch.object(planner, 'ROOT', self.root), \
             patch.object(planner, 'load_task_config', return_value=(
                 {'train_target_images': 2500}, {'background_pool_manifest': 'fixture.json'})), \
             patch.object(planner.measured, 'pool_plan_inputs', return_value=self.frames), \
             patch.object(planner.measured.base, 'ANNOTATED', self.annotated), \
             patch.object(planner.measured.base, 'cutout', return_value=np.zeros((100, 30, 4))), \
             patch.object(planner.measured, '_sign_contact_width', return_value=12), \
             patch.object(planner.measured, 'recorded_contact_status', side_effect=contact_for), \
             patch.object(planner.measured, '_prepare'):
            return planner.build(budget, 12, shared_pool=True, requested_budget=requested_budget)

    def test_small_budget_with_five_parents_is_valid(self):
        plan = self.compile(16)
        self.assertEqual(plan['budget'], 16)
        self.assertEqual(len(plan['strata']), 16)
        self.assertEqual(sum(group['min_count'] for group in plan['group_allocation']['strata']), 0)
        self.assertEqual(plan['design_context']['task'], 'sign')
        self.assertEqual(plan['design_context']['coverage'], 'unverified')
        self.assertEqual({item['id'] for item in plan['design_context']['crosses']},
                         {'fixture_background', 'view_tone'})

    def test_failed_parent_is_recorded_and_other_groups_fill_budget(self):
        plan = self.compile(20, contact='parent_0', requested_budget=25)
        outcome = plan['placement_outcome']
        self.assertEqual(plan['budget'], 20)
        self.assertTrue(outcome['failed_groups'])
        self.assertGreater(outcome['reallocated_images'], 0)
        self.assertEqual(plan['quota_gap']['remaining_shortfall'], 5)
        self.assertTrue(all(state['background']['image'] != 'background_0.png' for state in plan['strata']))

    def test_requested_budget_above_design_target_records_shortfall(self):
        plan = self.compile(16, requested_budget=3000)
        self.assertEqual(plan['budget'], 16)
        self.assertEqual(plan['quota_gap']['requested_budget'], 3000)
        self.assertEqual(plan['quota_gap']['remaining_shortfall'], 2984)

    def test_compiled_budget_above_design_target_is_allowed(self):
        plan = self.compile(2501)
        self.assertEqual(plan['placement_outcome']['requested_budget'], 2501)
        self.assertEqual(plan['budget'] + plan['placement_outcome']['unfilled_images'], 2501)
        self.assertGreater(plan['budget'], 0)

    def test_invalid_budget_is_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive integer'):
                self.compile(value)
        with self.assertRaisesRegex(ValueError, 'must include'):
            self.compile(16, requested_budget=15)

    def test_when_all_groups_fail_no_partial_plan_is_fabricated(self):
        for frame in self.frames:
            frame['support_rectangle_xyxy'] = [0, 0, 1, 1]
        with self.assertRaisesRegex(ValueError, 'No feasible sign placements'):
            self.compile(1)


if __name__ == '__main__':
    unittest.main()
