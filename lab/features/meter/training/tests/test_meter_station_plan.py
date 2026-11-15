"""Contract checks for station allocation mapping; no rendering is performed."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from meter_station_plan import (allocated_station_states, validate_station_declaration,
                                allocated_arena_states, validate_arena_declaration)


class StationPlanTests(unittest.TestCase):
    def state(self, name='a', count=1):
        return {'id': name, 'planned_count': count, 'sources': ['scene.blend'],
                'parameters': {'temperature_fraction': .2, 'humidity_fraction': .7,
                               'ground_angle_deg': 65, 'camera_z': 9,
                               'camera_x_offset': 0, 'camera_yaw_deg': 0,
                               'exposure_ev': -.5, 'saturation': .65,
                               'light_side': 1, 'occlusion': 'none'}}

    def plan(self, *states, budget=1):
        return {'model_key': 'meter', 'budget': budget, 'strata': list(states)}

    def test_exact_allocation_and_single_sample(self):
        a, b = self.state('a', 1), self.state('b', 1)
        b['parameters']['temperature_fraction'] = .9
        result = allocated_station_states(self.plan(a, b, budget=2), count=2,
                                          source='scene.blend', allow_occlusion=False)
        self.assertEqual([item['id'] for item in result], ['a', 'b'])
        self.assertEqual(result[-1]['parameters']['temperature_fraction'], .9)
        self.assertEqual(len(allocated_station_states(self.plan(self.state()), count=1,
                                                       source='scene.blend', allow_occlusion=False)), 1)

    def test_rejects_unsupported_states_and_unallocated_budget(self):
        for mutate in ('unknown', 'nan', 'occlusion', 'source', 'quota', 'duplicate_quota'):
            with self.subTest(mutate=mutate):
                state = self.state()
                if mutate == 'unknown': state['parameters']['unknown'] = 1
                if mutate == 'nan': state['parameters']['saturation'] = float('nan')
                if mutate == 'occlusion': state['parameters']['occlusion'] = 'temperature_tip'
                if mutate == 'source': state['sources'] = ['other.blend']
                if mutate == 'quota': state['planned_count'] = 0
                if mutate == 'duplicate_quota': state['planned_count'] = 2
                with self.assertRaises(ValueError):
                    allocated_station_states(self.plan(state), count=1,
                                             source='scene.blend', allow_occlusion=False)

    def test_declaration_requires_supported_semantics(self):
        state = self.state()
        state.update(background={'kind': 'station_scene'}, target={'kind': 'dual_needle_meter'},
                     labels={'kind': 'candidate_pose'}, augmentation={'kind': 'explicit_station_parameters'},
                     max_count=1)
        state['conditions'] = []
        declaration = dict(self.plan(state), target_split='train')
        declaration['constraints'] = {'resolution': [640, 480], 'camera_fallback': 'reject',
                                      'label_policy': 'candidate_pose'}
        validate_station_declaration(declaration, count=1, source='scene.blend', allow_occlusion=False)
        for field in ('background', 'target', 'labels', 'augmentation'):
            changed = dict(declaration)
            changed['strata'] = [dict(state, **{field: {'kind': 'unsupported'}})]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_station_declaration(changed, count=1, source='scene.blend', allow_occlusion=False)
        changed = dict(declaration, constraints=dict(declaration['constraints'], extra=True))
        with self.assertRaises(ValueError):
            validate_station_declaration(changed, count=1, source='scene.blend', allow_occlusion=False)
        changed = dict(declaration, strata=[dict(state, max_count=2)])
        with self.assertRaises(ValueError):
            validate_station_declaration(changed, count=1, source='scene.blend', allow_occlusion=False)
        changed = dict(declaration, strata=[dict(state, conditions=['unimplemented'])])
        with self.assertRaises(ValueError):
            validate_station_declaration(changed, count=1, source='scene.blend', allow_occlusion=False)
        duplicate = dict(state, id='b', parameters=dict(state['parameters'], light_side=1.0))
        changed = dict(declaration, budget=2, strata=[state, duplicate])
        with self.assertRaises(ValueError):
            validate_station_declaration(changed, count=2, source='scene.blend', allow_occlusion=False)


class ArenaPlanTests(unittest.TestCase):
    def declaration(self):
        state = {'id': 'first', 'min_count': 1, 'max_count': 1, 'conditions': [],
                 'background': {'kind': 'arena_scene'}, 'target': {'kind': 'dual_needle_meter'},
                 'labels': {'kind': 'candidate_pose'}, 'augmentation': {'kind': 'explicit_arena_parameters'},
                 'sources': ['arena.blend', 'station.blend'],
                 'parameters': {'temperature_fraction': .2, 'humidity_fraction': .7, 'ground_angle_deg': 65,
                                'station_x': -.60, 'station_y': .15, 'camera_x': -.6, 'camera_y': -.75,
                                'camera_z': .42, 'exposure_ev': -.5, 'saturation': .65}}
        return {'model_key': 'meter', 'target_split': 'train', 'budget': 1,
                'constraints': {'resolution': [640, 480], 'label_policy': 'candidate_pose', 'foreground': True},
                'strata': [state]}

    def test_single_and_batch_allocations(self):
        declaration = self.declaration()
        validate_arena_declaration(declaration, count=1, sources=['arena.blend', 'station.blend'], foreground=True)
        state = dict(declaration['strata'][0], planned_count=1)
        plan = dict(declaration, strata=[state])
        self.assertEqual(allocated_arena_states(plan, count=1)[0]['parameters']['camera_z'], .42)
        second = dict(state, id='second', parameters=dict(state['parameters'], temperature_fraction=.4))
        plan = dict(plan, budget=2, strata=[state, second])
        self.assertEqual([x['id'] for x in allocated_arena_states(plan, count=2)], ['first', 'second'])

    def test_rejects_unsupported_or_duplicate_states(self):
        for kind in ('field', 'limit', 'source', 'max', 'conditions', 'duplicate'):
            with self.subTest(kind=kind):
                declaration = self.declaration()
                state = declaration['strata'][0]
                if kind == 'field': state['parameters']['other'] = 1
                if kind == 'limit': state['parameters']['camera_z'] = 9
                if kind == 'source': state['sources'] = ['arena.blend']
                if kind == 'max': state['max_count'] = 2
                if kind == 'conditions': state['conditions'] = ['unimplemented']
                if kind == 'duplicate': declaration['strata'].append(dict(state, id='duplicate'))
                with self.assertRaises(ValueError):
                    validate_arena_declaration(declaration, count=1,
                                               sources=['arena.blend', 'station.blend'], foreground=True)


if __name__ == '__main__':
    unittest.main()
