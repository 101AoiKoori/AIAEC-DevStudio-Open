"""Exercise train planning when a coverage target or source mix cannot be met."""

import copy
import unittest
from unittest.mock import patch

from datasets.scripts.person import build_full_person_plan as planner


class TrainPlanCapacityTest(unittest.TestCase):
    def build_limited_plan(self, total):
        background = {
            'image': 'background.png', 'state': 'candidate_eligible', 'boxes': [],
            'generation_method': 'blender_render', 'source_group': 'scene',
            'label': {'evidence': {'vehicle_visible': False, 'scene_state': 'other'}},
        }
        states = []
        for index in range(3):
            states.append({
                'id': f'state_{index}', 'min_count': 0, 'max_count': 1,
                'background': {'record_index': 0},
                'axes': {'person_count': 4, 'vehicle_occlusion': 'none'},
                'target': {'actors': [{'kind': 'person', 'category': 'community'}]},
                'parameters': {'layout': index}, 'augmentation': {},
            })
        batch = {'model_key': 'person', 'target_split': 'train', 'constraints': {},
                 'design': {}, 'conditional_balance': {}, 'strata': states}

        def read_json(path):
            if path == 'fake-dispatch':
                return {'records': [background]}
            if path == planner.DIGITAL_WHITE_BACKS:
                return {'records': []}
            if path == 'datasets/annotated/person/v4_white_back_source_review/source.json':
                return {'foreground': 'observed_white_back.png'}
            raise AssertionError(path)

        def allocate(declaration, _root):
            return {'strata': [dict(state, planned_count=int(index < declaration['budget']))
                               for index, state in enumerate(declaration['strata'])]}

        def inspect(job):
            _declaration, state = job
            return state['id'], [], {'unoccluded_classes': ['community', 'noncommunity'],
                                     'vehicle': {}, 'mutual_occlusion': [],
                                     'boxes_by_class': {'community': 1}}, None

        with patch('datasets.scripts.common.package_direct_real.prepare', return_value=([], None)), \
             patch('datasets.scripts.person.planned_sources.configured_background_dispatch',
                   return_value=('fake-dispatch', True)), \
             patch('datasets.scripts.person.planned_sources.complete_background', return_value=True), \
             patch('datasets.scripts.person.planned_sources.verify_current_backgrounds'), \
             patch('lab.training_common.bayesian_allocation.plan_generation', side_effect=allocate), \
             patch.object(planner, 'read_json', side_effect=read_json), \
             patch.object(planner, 'build_sample', return_value=(copy.deepcopy(batch), [])), \
             patch.object(planner, '_full_train_pool', return_value=copy.deepcopy(batch)), \
             patch.object(planner, 'attach_sources'), \
             patch.object(planner, '_inspect_train_candidate', side_effect=inspect):
            return planner._build_supervised_train(total)

    def test_blender_only_pool_remains_usable_and_reports_gaps(self):
        result = self.build_limited_plan(5)
        self.assertEqual(result['budget'], 3)
        self.assertEqual(len(result['strata']), 3)
        self.assertEqual(result['design']['planned_source_mix']['blender_background_composite'], 3)
        self.assertFalse(result['design']['blender_cap_enforced'])
        self.assertEqual(result['design']['requested_images'], 5)
        self.assertEqual(result['design']['planned_images'], 3)
        self.assertEqual(result['design']['unfilled_images'], 2)
        self.assertGreater(result['design']['scaled_joint_quota_capacity_gaps']['community_upright'], 0)

    def test_budget_above_design_target_keeps_feasible_states_and_reports_shortfall(self):
        result = self.build_limited_plan(3001)
        self.assertEqual(result['budget'], 3)
        self.assertEqual(result['design']['requested_images'], 3001)
        self.assertEqual(result['design']['planned_images'], 3)
        self.assertEqual(result['design']['unfilled_images'], 2998)

    def test_nonpositive_or_noninteger_train_budget_is_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive integer'):
                planner.build_supervised_train(value)


if __name__ == '__main__':
    unittest.main()
