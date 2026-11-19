"""Evaluation planning retains complete feasible groups under limited capacity."""
import unittest
from unittest.mock import patch

from datasets.scripts.person import build_evaluation_plan as planner


def _actor(actor_id, category, source):
    return {'id': actor_id, 'kind': 'person', 'category': category, 'source': source,
            'foot_xy': [20, 30], 'size_wh': [20, 40], 'angles_degrees': {'yaw': 0}}


def _state(index):
    actors = [_actor('a', 'community', 'a.png'), _actor('b', 'noncommunity', 'b.png')]
    for actor in actors:
        actor['foot_xy'][0] += index * 10
    return {'id': f'layout_{index}', 'failure_event': 'multi', 'min_count': 0, 'max_count': 1,
            'background': {'record_index': 0}, 'augmentation': {'test': index},
            'target': {'actors': actors},
            'parameters': {'ground_placement': {'actors': [{'actor_id': 'a'}, {'actor_id': 'b'}]}},
            'axes': {'person_count': 2, 'vehicle_occlusion': 'none'}}


class EvaluationCapacityTest(unittest.TestCase):
    def build(self, total, count, *, failed_negative=False, no_layout=False, pairing=None):
        backgrounds = [
            {'split': 'val', 'state': 'complete_negative', 'boxes': [], 'frame_id': 'v',
             'source_group': 'shared', 'parent_group': 'shared', 'source_image_size': [640, 480]},
            {'split': 'train', 'state': 'complete_negative', 'boxes': [], 'frame_id': 't',
             'source_group': 'shared', 'parent_group': 'shared', 'source_image_size': [640, 480]},
        ]
        if pairing:
            backgrounds.append({'split': 'val', 'state': 'complete_negative', 'boxes': [], 'frame_id': 'ok',
                                'source_group': 'allowed', 'parent_group': 'allowed', 'source_image_size': [640, 480]})
        selected_index = 2 if pairing else 0
        batch = {'schema_version': 1, 'model_key': 'person', 'target_split': 'val',
                 'constraints': {}, 'design': {}, 'conditional_balance': {},
                 'strata': [_state(index) for index in range(count)]}
        for state in batch['strata']:
            state['background']['record_index'] = selected_index
        white = {'a.png': (0, 'back_a.png', .5, 'white.json'),
                 'b.png': (1, 'back_b.png', .5, 'white.json')}
        negative = {'id': 'negative', 'failure_event': 'no_standee_negative', 'min_count': 1, 'max_count': 1,
                    'background': {'record_index': selected_index}, 'target': {'actors': []}, 'parameters': {}, 'axes': {}}

        def validate(declaration, limit=None):
            return ([{'state_id': 'negative', 'error': 'bad'}]
                    if failed_negative and any(state['id'] == 'negative' for state in declaration['strata']) else [])

        def allocate(declaration, _root):
            return {'strata': [dict(state, planned_count=int(index < declaration['budget']))
                               for index, state in enumerate(declaration['strata'])]}

        sample = ValueError('no feasible person sample scenes') if no_layout else (batch, [])
        with patch.object(planner, 'load_task_config', return_value=({'target_spec': {
                'minimum_placed_height_fraction': 0}}, {'pairing': pairing or {}})), \
             patch.object(planner, 'configured_background_dispatch', return_value=('dispatch', False)), \
             patch.object(planner, 'prepare', return_value=([], None)), \
             patch.object(planner, 'prepare_direct'), patch.object(planner, 'read_source_history', return_value=[]), \
             patch.object(planner, 'historical_source', side_effect=lambda path, _history: path), \
             patch.object(planner, 'complete_background', side_effect=lambda row, _state, split: row['split'] == split), \
             patch.object(planner.scenes, 'read_json', return_value={'records': backgrounds}), \
             patch.object(planner.scenes, 'build_sample', side_effect=sample if no_layout else None,
                          return_value=sample if not no_layout else None), \
             patch.object(planner.scenes, 'white_back_textures', return_value=white), \
             patch.object(planner.scenes, 'attach_sources'), \
             patch.object(planner.scenes, 'validate', side_effect=validate), \
             patch.object(planner, '_negatives', return_value=iter([negative])), \
             patch.object(planner, 'plan_generation', side_effect=allocate):
            return planner.build('val', total)

    def test_limited_pool_keeps_complete_triplet_and_reports_shortfall(self):
        result = self.build(10, 2)
        self.assertEqual(result['budget'], len(result['strata']))
        self.assertLess(result['design']['planned_images'], 10)
        self.assertEqual(result['design']['unfilled_images'], 10 - result['design']['planned_images'])
        self.assertEqual(result['design']['matched_groups'], 2)
        self.assertEqual(result['design']['matched_group_shortfall'], 10)
        self.assertEqual(result['design']['selected_parent_overlap_with_other_split_candidates'], ['shared'])
        self.assertEqual(sum(state['failure_event'] == 'person_single' for state in result['strata']), 2)
        self.assertEqual(sum(state['failure_event'] == 'white_back_hard_negative' for state in result['strata']), 2)
        self.assertEqual(sum(state['failure_event'] == 'person_with_white_back' for state in result['strata']), 2)

    def test_no_person_layout_still_emits_confirmed_negative(self):
        result = self.build(10, 0, no_layout=True)
        self.assertEqual(result['budget'], 1)
        self.assertEqual(result['design']['unfilled_images'], 9)

    def test_failed_negative_does_not_block_valid_layout(self):
        result = self.build(4, 1, failed_negative=True)
        self.assertEqual(result['design']['invalid_negative_states'], ['negative'])
        self.assertGreater(result['budget'], 0)

    def test_explicit_reservations_and_declared_same_source_allowance(self):
        from datasets.scripts.common.source_ledger.config import build_groups
        pairing = {'task_real': [{'id': 'test_only', 'paths': ['capture/test.mp4'],
                                  'split': {'primary': 'test', 'status': 'reserved'}},
                                 {'id': 'shared_diagnostic', 'paths': ['capture/shared.mp4'],
                                  'split': {'primary': 'test', 'status': 'reserved',
                                            'same_source_allowed_in': ['val']}}]}
        groups = build_groups('person', pairing)
        self.assertEqual(planner._reservation_conflicts({'parent_video': 'capture/test.mp4'}, 'val', groups, []),
                         [{'group': 'test_only', 'reserved_split': 'test'}])
        self.assertEqual(planner._reservation_conflicts({'parent_video': 'capture/shared.mp4'}, 'val', groups, []), [])
        self.assertEqual(planner._reservation_conflicts({'parent_video': 'capture/other.mp4'}, 'val', groups, []), [])

    def test_reserved_candidate_is_excluded_without_blocking_other_background(self):
        pairing = {'backgrounds': [{'id': 'shared', 'paths': [],
                                    'split': {'primary': 'test', 'status': 'reserved'}}]}
        result = self.build(4, 1, pairing=pairing)
        self.assertEqual(result['design']['eligible_backgrounds'], 1)
        self.assertEqual(result['design']['excluded_reserved_sources'][0]['reservations'],
                         [{'group': 'shared', 'reserved_split': 'test'}])
        self.assertTrue(all(state['background']['record_index'] == 2 for state in result['strata']))

    def test_matched_groups_do_not_exceed_budget_across_backgrounds(self):
        first, second = _state(0), _state(1)
        second['background']['record_index'] = 1
        white = {'a.png': (0, 'back_a.png', .5, 'white.json'),
                 'b.png': (1, 'back_b.png', .5, 'white.json')}
        with patch.object(planner.scenes, 'white_back_textures', return_value=white), \
             patch.object(planner.scenes, 'attach_sources'), \
             patch.object(planner.scenes, 'validate', return_value=[]):
            selected, used = planner._contrasts({'strata': [first, second]}, [{}, {}],
                                                per_class=6, max_groups=1)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(used), 1)
        self.assertEqual({state['failure_event'] for state in selected},
                         {'person_single', 'white_back_hard_negative', 'person_with_white_back'})


if __name__ == '__main__':
    unittest.main()
