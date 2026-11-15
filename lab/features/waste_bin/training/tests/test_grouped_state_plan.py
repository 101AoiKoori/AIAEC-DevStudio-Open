import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
grouped = importlib.import_module('generate_waste_v4_grouped')


class GroupedStatePlanTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp')
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.source = self.root / 'render'
        self.source.mkdir()
        for name in ('bg.png', 'parent.png', 'render/image.png', 'render/mask.png',
                     'render/state_mask.png', 'render/label.txt'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture')
        self.bg_manifest = self.root / 'backgrounds.json'
        frame = {'id': 'frame', 'image': 'bg.png', 'parent_image': 'parent.png',
                 'split': 'train', 'ground_polygon_xy': [[0, 300], [639, 300], [639, 479], [0, 479]]}
        self.bg_manifest.write_text(json.dumps({'frames': [frame]}), encoding='utf8')
        self.sample = {'image': 'image.png', 'mask_visible': 'mask.png', 'mask_state_visible': 'state_mask.png',
                       'candidate_label': 'label.txt', 'labels': [{'bin': 'blue', 'state': 'open', 'label_eligible': True}],
                       'visible_bins': ['blue'], 'bin_count': 1, 'open_pose': 'upright',
                       'states': {'blue': 'open', 'red': 'closed', 'green': 'closed', 'gray': 'closed'}}
        self.report = {'status': 'complete_candidates', 'split': 'train',
                       'label_contract_id': grouped.STATE_CONTRACT, 'samples': [self.sample]}
        (self.source / 'manifest.json').write_text(json.dumps(self.report), encoding='utf8')
        self.state = {'id': 'a', 'conditions': [], 'planned_count': 1,
                      'sources': ['bg.png', 'render/image.png', 'render/mask.png'],
                      'background': {'manifest': 'backgrounds.json', 'id': 'frame'},
                      'target': {'render_manifest': 'render/manifest.json', 'render_image': 'image.png',
                                 'bins': {'blue': 'open'}},
                      'labels': {'mode': 'candidate', 'open': 0, 'closed': 1},
                      'augmentation': {'value_gain': 1, 'saturation_gain': 1, 'shadow': 'simulated_gaussian'},
                      'parameters': {'placement_xywh': [10, 100, 100, 150]}}
        self.plan = {'model_key': 'waste_bin_state', 'target_split': 'train', 'budget': 1,
                     'constraints': {'resolution': [640, 480], 'label_mode': 'candidate', 'ground_placement': 'polygon'},
                     'strata': [self.state]}
        self.patches = [patch.object(grouped, 'ROOT', self.root), patch.object(grouped, 'BG_MANIFEST', self.bg_manifest),
                        patch.object(grouped.source_gate, 'queue_inputs', return_value=(set(), set(), []))]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_sources_and_state_are_checked_before_composition(self):
        jobs, media, procedural, _ = grouped._planned_inputs(self.plan)
        self.assertEqual(len(jobs), 1)
        self.assertIn(self.root / 'render/mask.png', media)
        self.assertIn(self.source / 'manifest.json', procedural)
        self.state['target']['bins']['blue'] = 'closed'
        with self.assertRaisesRegex(ValueError, 'state evidence'):
            grouped._planned_inputs(self.plan)
        self.state['target']['bins']['blue'] = 'open'
        self.state['planned_count'] = 2
        with self.assertRaisesRegex(ValueError, 'repeated'):
            grouped._planned_inputs(self.plan)
        self.state['planned_count'] = 1
        self.state['augmentation']['blur'] = 1
        with self.assertRaisesRegex(ValueError, 'augmentation'):
            grouped._planned_inputs(self.plan)

    def test_explicit_single_split_quantity_pool_and_mismatch(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = np.zeros((480, 640, 4), dtype=np.uint8)
        mask[100:120, 200:210] = [0, 0, 255, 255]
        def read(path, flags=None):
            return mask if Path(path).name == 'mask.png' else rgb
        with patch.object(grouped.base, 'read_image', side_effect=read):
            pools = grouped.load_quantity_sources([self.source], split_counts={'train': [1]}, source_splits=['train'])
            self.assertEqual(len(pools[('train', 1)]), 1)
            self.assertEqual(pools[('train', 1)][0]['source_code'], '1000')
            with self.assertRaisesRegex(ValueError, 'manifest split'):
                grouped.load_quantity_sources([self.source], split_counts={'val': [1]}, source_splits=['val'])
            with self.assertRaisesRegex(ValueError, 'lacks'):
                grouped.load_quantity_sources([self.source], split_counts={'train': [2]}, source_splits=['train'])
            with self.assertRaises(FileNotFoundError):
                grouped.load_quantity_sources([self.root / 'missing'], split_counts={'train': [1]}, source_splits=['train'])

    def test_composition_uses_plan_parameters_and_records_actual_state(self):
        self.state['parameters']['placement_xywh'] = [10, 200, 100, 200]
        self.state['parameters']['ground_polygon_xy'] = [[5, 305], [634, 305], [634, 474], [5, 474]]
        self.state['target']['bin_ids'] = {'blue': 'scene-a-bin-1'}
        self.state['augmentation']['value_gain'] = .8
        self.state['augmentation']['saturation_gain'] = .75
        jobs, _, _, _ = grouped._planned_inputs(self.plan)
        rgb = np.full((480, 640, 3), 100, dtype=np.uint8)
        mask = np.zeros((480, 640, 4), dtype=np.uint8)
        mask[100:120, 200:210] = [0, 0, 255, 255]
        def read(path, flags=None):
            return mask.copy() if Path(path).name == 'mask.png' else rgb.copy()
        run = SimpleNamespace(plan=self.plan, record_sample=Mock())
        out = self.root / 'output' / 'batch'
        with patch.object(grouped, 'sample_dir', return_value=out), \
             patch.object(grouped, 'build_queue', return_value={}), \
             patch.object(grouped, 'source_role', return_value=('unknown', 'missing history')), \
             patch.object(grouped.base, 'read_image', side_effect=read), \
             patch.object(grouped, 'write_png') as writer, \
             patch.object(grouped, 'apply_tone', side_effect=lambda image, **kwargs: image) as tone:
            grouped._generate(SimpleNamespace(out=str(out)), run, jobs)
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(tone.call_args.kwargs, {'value_gain': .8, 'saturation_gain': .75})
        self.assertEqual(run.record_sample.call_count, 1)
        self.assertEqual(run.record_sample.call_args.kwargs['sources'], self.state['sources'])
        self.assertEqual(run.record_sample.call_args.kwargs['parameters'], self.state['parameters'])
        actual = run.record_sample.call_args.kwargs['actual']
        self.assertEqual(actual['placement_xywh'], [10, 200, 100, 200])
        self.assertEqual(actual['bins'][0]['state'], 'open')
        self.assertEqual(actual['bins'][0]['bin_id'], 'scene-a-bin-1')
        self.assertIn('parent_ground_polygon_xy', actual['background'])
        self.assertEqual(actual['background_exposure_role'], 'unknown')
        self.assertEqual(actual['background_exposure_reason'], 'missing history')

    def test_plan_rejects_invalid_identity_and_region(self):
        self.state['target']['bin_ids'] = {'blue': ''}
        with self.assertRaisesRegex(ValueError, 'bin_ids'):
            grouped._planned_inputs(self.plan)
        self.state['target']['bin_ids'] = {'blue': 'bin-1'}
        self.state['parameters']['ground_polygon_xy'] = [[0, 290], [100, 290], [100, 350]]
        with self.assertRaisesRegex(ValueError, 'within the background region'):
            grouped._planned_inputs(self.plan)

    def test_independent_placements_use_final_visible_mask_and_guard_state(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        image = np.full_like(frame, 120)
        blue = np.zeros((480, 640), dtype=bool)
        red = np.zeros_like(blue)
        blue[100:140, 100:120] = True
        red[100:140, 200:220] = True
        state = np.zeros((480, 640, 4), dtype=np.uint8)
        state[102:106, 102:106] = [0, 0, 255, 255]
        state[102:106, 202:206] = [0, 255, 0, 255]
        job = {'target': {'bins': {'blue': 'closed', 'red': 'closed'}},
               'parameters': {'placements_xywh': {'blue': [100, 330, 20, 40], 'red': [110, 330, 20, 40]},
                              'back_to_front': ['blue', 'red']}}
        floor = np.ones((480, 640), dtype=np.uint8)
        _, boxes, supports = grouped._independent_composite(frame, image, {'blue': blue, 'red': red}, state, floor, job)
        self.assertEqual(boxes[0]['bbox_xyxy'], [100, 330, 110, 370])
        self.assertEqual(boxes[1]['bbox_xyxy'], [110, 330, 130, 370])
        self.assertEqual(set(supports), {'blue', 'red'})
        state[102:106, 102:106] = 0
        state[102:106, 115:119] = [0, 0, 255, 255]
        with self.assertRaisesRegex(ValueError, 'obscures state evidence'):
            grouped._independent_composite(frame, image, {'blue': blue, 'red': red}, state, floor, job)

    def test_independent_plan_requires_full_source_visibility(self):
        self.state['parameters'] = {'placements_xywh': {'blue': [10, 270, 100, 200]},
                                    'back_to_front': ['blue']}
        with self.assertRaisesRegex(ValueError, 'fully visible source bins'):
            grouped._planned_inputs(self.plan)
        self.sample['labels'][0].update(visible_fraction=1.0, in_frame=True)
        (self.source / 'manifest.json').write_text(json.dumps(self.report), encoding='utf8')
        jobs, _, _, _ = grouped._planned_inputs(self.plan)
        self.assertEqual(len(jobs), 1)
        self.sample['labels'][0]['label_eligible'] = False
        (self.source / 'manifest.json').write_text(json.dumps(self.report), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'state evidence'):
            grouped._planned_inputs(self.plan)
        self.sample['labels'][0]['label_eligible'] = True
        (self.source / 'manifest.json').write_text(json.dumps(self.report), encoding='utf8')
        self.state['parameters']['back_to_front'] = ['blue', 'blue']
        with self.assertRaisesRegex(ValueError, 'depth order'):
            grouped._planned_inputs(self.plan)

    def test_independent_plan_records_ids_and_visible_boxes(self):
        self.sample['labels'] = [{'bin': color, 'state': 'closed', 'label_eligible': True,
                                  'visible_fraction': 1.0, 'in_frame': True} for color in ('blue', 'red')]
        self.sample['visible_bins'] = ['blue', 'red']
        self.state['target']['bins'] = {'blue': 'closed', 'red': 'closed'}
        self.state['target']['bin_ids'] = {'blue': 'bin-a', 'red': 'bin-b'}
        self.state['parameters'] = {'placements_xywh': {'blue': [100, 330, 20, 40], 'red': [110, 330, 20, 40]},
                                    'back_to_front': ['blue', 'red']}
        (self.source / 'manifest.json').write_text(json.dumps(self.report), encoding='utf8')
        jobs, _, _, _ = grouped._planned_inputs(self.plan)
        image = np.full((480, 640, 3), 100, dtype=np.uint8)
        mask = np.zeros((480, 640, 4), dtype=np.uint8)
        mask[100:140, 100:120] = [0, 0, 255, 255]
        mask[100:140, 200:220] = [0, 255, 0, 255]
        state = np.zeros_like(mask)
        state[102:106, 102:106] = [0, 0, 255, 255]
        state[102:106, 202:206] = [0, 255, 0, 255]
        def read(path, flags=None):
            return {'mask.png': mask, 'state_mask.png': state}.get(Path(path).name, image).copy()
        run = SimpleNamespace(plan=self.plan, record_sample=Mock())
        out = self.root / 'output' / 'independent'
        with patch.object(grouped, 'sample_dir', return_value=out), \
             patch.object(grouped, 'build_queue', return_value={}), \
             patch.object(grouped, 'source_role', return_value=('unknown', 'fixture')), \
             patch.object(grouped.base, 'read_image', side_effect=read), \
             patch.object(grouped, 'write_png'), \
             patch.object(grouped, 'apply_tone', side_effect=lambda value, **kwargs: value):
            grouped._generate(SimpleNamespace(out=str(out)), run, jobs)
        actual = run.record_sample.call_args.kwargs['actual']
        self.assertEqual([(row['bin_id'], row['bbox_xyxy']) for row in actual['bins']],
                         [('bin-a', [100, 330, 110, 370]), ('bin-b', [110, 330, 130, 370])])
        self.assertEqual(actual['back_to_front'], ['blue', 'red'])
        self.assertEqual(len((out / 'labels/planned_000000.txt').read_text().splitlines()), 2)


if __name__ == '__main__':
    unittest.main()
