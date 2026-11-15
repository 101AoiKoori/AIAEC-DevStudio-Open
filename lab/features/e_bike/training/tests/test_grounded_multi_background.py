import copy
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import compose_v4_grounded_train as entry
from lab.features.e_bike.training.tests.test_grounded_plan import fixture


class GroundedMultiBackgroundTest(unittest.TestCase):
    def custom_fixture(self):
        plan, state, bg, vehicles = fixture()
        bg['id'] = 'custom'
        bg['ground_zone'] = {'poly': [[0, 300], [639, 300], [639, 479], [0, 479]]}
        state['background']['id'] = bg['id']
        return plan, state, bg, vehicles

    def test_custom_polygon_multiple_vehicles_and_visible_masks(self):
        plan, state, bg, vehicles = self.custom_fixture()
        target = copy.deepcopy(state['target']['vehicles'][0])
        actor = copy.deepcopy(state['parameters']['actors'][0])
        # Declare the front actor first to verify compositing order comes from depth.
        actor.update(foot_x=350, foot_y=405)
        state['target']['vehicles'].insert(0, target)
        state['parameters']['actors'].insert(0, actor)
        with mock.patch.object(entry, 'source_role', return_value=('train', None)), \
             mock.patch.object(entry, 'read_image', return_value=np.full((480, 640, 3), 120, np.uint8)):
            entry.validate_ground_plan(plan, [bg], vehicles, {})
            _, objects, masks, _ = entry.render_ground_state(state, [bg], vehicles)
        self.assertEqual([obj['foot_xy'] for obj in objects], [[320, 400], [350, 405]])
        self.assertEqual(len(masks), 2)
        self.assertFalse(np.any(masks[0] & masks[1]))
        self.assertLess(objects[0]['visible_fraction'], 1.0)
        self.assertEqual(objects[1]['visible_fraction'], 1.0)
        for obj, mask in zip(objects, masks):
            y, x = np.nonzero(mask)
            self.assertEqual(obj['bbox_xyxy'], [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1])

    def test_invalid_polygon_and_outside_contact_are_rejected(self):
        plan, state, bg, vehicles = self.custom_fixture()
        with mock.patch.object(entry, 'source_role', return_value=('train', None)), \
             mock.patch.object(entry, 'read_image', return_value=np.full((480, 640, 3), 120, np.uint8)):
            bg['ground_zone']['poly'] = [[0, 300], [1, 300], [2, 300]]
            with self.assertRaisesRegex(ValueError, 'enclose an area'):
                entry.validate_ground_plan(plan, [bg], vehicles, {})
            bg['ground_zone']['poly'] = [[0, 410], [639, 410], [639, 479], [0, 479]]
            entry.validate_ground_plan(plan, [bg], vehicles, {})
            with self.assertRaisesRegex(ValueError, 'outside the fitted ground zone'):
                entry.render_ground_state(state, [bg], vehicles)
            bg['ground_zone']['poly'][0] = [-1, 410]
            with self.assertRaisesRegex(ValueError, 'pixel coordinates'):
                entry.validate_ground_plan(plan, [bg], vehicles, {})

    def test_each_background_uses_its_own_polygon(self):
        plan, state, first, vehicles = self.custom_fixture()
        second = copy.deepcopy(first)
        second.update(id='other', image='datasets/other.png')
        second['ground_zone']['poly'] = [[0, 410], [639, 410], [639, 479], [0, 479]]
        other_state = copy.deepcopy(state)
        other_state['id'] = 'other-state'
        other_state['background'] = {'id': second['id'], 'image': second['image']}
        other_state['sources'] = sorted((set(state['sources']) - {first['image']}) | {second['image']})
        plan['strata'].append(other_state)
        plan['budget'] = 2
        with mock.patch.object(entry, 'source_role', return_value=('train', None)), \
             mock.patch.object(entry, 'read_image', return_value=np.full((480, 640, 3), 120, np.uint8)):
            entry.validate_ground_plan(plan, [first, second], vehicles, {})
            entry.render_ground_state(state, [first, second], vehicles)
            with self.assertRaisesRegex(ValueError, 'outside the fitted ground zone'):
                entry.render_ground_state(other_state, [first, second], vehicles)

    def test_external_manifest_is_guarded_and_selected(self):
        plan, _, bg, vehicles = self.custom_fixture()
        with tempfile.TemporaryDirectory(dir=entry.ROOT / 'tmp') as temp:
            root = Path(temp)
            manifest = root / 'backgrounds.json'
            plan_path = root / 'plan.json'
            manifest.write_text(json.dumps({'backgrounds': [bg]}), encoding='utf-8')
            plan_path.write_text(json.dumps(plan), encoding='utf-8')
            captured = {}

            @contextmanager
            def guard(*args, **kwargs):
                captured.update(kwargs)
                yield type('Run', (), {'reused': True})()

            args = ['entry', '--output', 'datasets/samples/e_bike',
                    '--plan', str(plan_path.relative_to(entry.ROOT))]
            source_config = {'ground_background_manifest': str(manifest.relative_to(entry.ROOT)),
                             'ground_vehicle_selection': str(entry.bike.SELECTION.relative_to(entry.ROOT))}
            with mock.patch.object(sys, 'argv', args), \
                 mock.patch('datasets.scripts.common.config.load_task_config', return_value=({}, source_config)), \
                 mock.patch.object(entry.bike, 'sources', return_value=({}, vehicles)) as sources, \
                 mock.patch.object(entry, 'build_queue', return_value={}), \
                 mock.patch.object(entry, 'source_role', return_value=('train', None)), \
                 mock.patch('e_bike_queue_inputs.queue_input_report', return_value=([], [], [])), \
                 mock.patch('lab.training_common.production_guard.generation_guard', side_effect=guard):
                entry.main()
            self.assertIn(manifest, captured['procedural_inputs'])
            self.assertEqual(captured['production_plan'], plan)
            self.assertEqual(sources.call_args.kwargs['selection'], entry.bike.SELECTION)
            self.assertIn(entry.ROOT/'datasets/scripts/e_bike/config/source.yaml', captured['procedural_inputs'])


if __name__ == '__main__':
    unittest.main()
