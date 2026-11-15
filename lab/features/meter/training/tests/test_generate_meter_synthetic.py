import importlib.util
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'generate_meter_synthetic.py'
spec = importlib.util.spec_from_file_location('meter_synthetic', SCRIPT)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class SyntheticMeterTests(unittest.TestCase):
    def plan(self, count):
        states = []
        for i in range(count):
            states.append({'id': str(i), 'min_count': 1, 'max_count': 1, 'conditions': [],
                           'background': {'kind': 'solid_bgr'}, 'target': {'kind': 'self_drawn_meter'},
                           'labels': {'kind': 'synthetic_pose'},
                           'augmentation': {'kind': 'explicit_projective_no_glare'},
                           'sources': [generator.PROCEDURAL_SOURCE],
                           'parameters': {'temperature_fraction': .1 + i * .01, 'humidity_fraction': .5,
                                          'yaw_deg': 0, 'pitch_deg': 0, 'roll_deg': 0, 'distance': 800,
                                          'focal_px': 700, 'translation_x': 0, 'translation_y': 0,
                                          'brightness_gain': 1, 'background_bgr': [100, 110, 120]}})
        return {'model_key': 'meter', 'target_split': 'train', 'budget': count,
                'constraints': {'resolution': [640, 640], 'glare': 'none', 'label_policy': 'synthetic_pose'},
                'strata': states}

    def setUp(self):
        @contextmanager
        def fixture_guard(*args, **kwargs):
            plan = dict(kwargs['production_plan'])
            plan['strata'] = [dict(s, planned_count=1) for s in plan['strata']]
            yield SimpleNamespace(reused=False, source_refs={}, plan=plan, record_sample=mock.Mock())
        patcher = mock.patch.object(generator, 'generation_guard', fixture_guard)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_projected_landmarks_and_label_files(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'batch'
            generator.generate(output, 8, 120, production_plan=self.plan(8))
            manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(len(manifest['samples']), 8)
            self.assertEqual(len(manifest['keypoint_order']), 8)
            for sample in manifest['samples']:
                name = sample['id']
                image = cv2.imread(str(output / 'images' / 'train' / (name + '.png')))
                self.assertEqual(image.shape, (640, 640, 3))
                tokens = (output / 'labels' / 'train' / (name + '.txt')).read_text().split()
                self.assertEqual(len(tokens), 29)  # class + box + 8*(x,y,visibility)
                self.assertEqual(tokens[0], '0')
                box = np.asarray([float(v) for v in tokens[1:5]])
                self.assertTrue(np.all(box >= 0) and np.all(box <= 1))
                self.assertGreater(box[2], 0)
                self.assertGreater(box[3], 0)
                for i, (point, vis) in enumerate(zip(sample['keypoints'], sample['visibility'])):
                    x, y, v = float(tokens[5 + i * 3]), float(tokens[6 + i * 3]), int(tokens[7 + i * 3])
                    self.assertEqual(v, vis)
                    if v:
                        self.assertTrue(0 <= x <= 1 and 0 <= y <= 1)
                        np.testing.assert_allclose([x * 640, y * 640], point, atol=.001)
                        self.assertLessEqual(abs(x - box[0]), box[2] / 2 + .002)
                        self.assertLessEqual(abs(y - box[1]), box[3] / 2 + .002)
                    else:
                        self.assertEqual((x, y), (0, 0))
                inverse = np.linalg.inv(np.asarray(sample['parameters']['homography']))
                original = generator.map_points(sample['keypoints'], inverse)
                for name, idx in (('temperature', 0), ('humidity', 2)):
                    scale = generator.SCALES[name]
                    np.testing.assert_allclose(original[idx], scale['center'], atol=.01)
                    t = (sample['values'][name] - scale['minimum']) / (scale['maximum'] - scale['minimum'])
                    angle = scale['angles'][0] + t * (scale['angles'][1] - scale['angles'][0])
                    np.testing.assert_allclose(original[idx + 1], generator.polar(scale['center'], scale['radius'] - 25, angle), atol=.01)
                self.assertEqual(len(sample['tick_lines']['temperature']), 81)
                self.assertEqual(len(sample['tick_lines']['humidity']), 51)
                self.assertIn('glare_ellipses', sample)

    def test_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'batch'
            generator.generate(output, 1, 1, production_plan=self.plan(1))
            marker = output / 'marker.txt'
            marker.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                generator.generate(output, 1, 2, production_plan=self.plan(1))
            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')

    def test_generated_geometry_and_highlights_are_reproducible(self):
        first = generator.sample(np.random.default_rng(99))
        second = generator.sample(np.random.default_rng(99))
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])
        self.assertTrue(math.isfinite(first[3]['parameters']['yaw_deg']))

    def test_photo_augmentation_keeps_original_needle_positions(self):
        image, points, _, _, _ = generator.draw_dial(np.random.default_rng(42))
        reference = {'image_size': [640, 640], 'bbox_xyxy': [20, 20, 620, 620],
                     'status': 'unverified', 'source': 'test fixture',
                     'keypoints': [{'name': name, 'x': points[name][0], 'y': points[name][1], 'visibility': 2}
                                   for name in generator.KEYPOINTS]}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'photo'
            source = Path(temp) / 'source.png'
            annotation = Path(temp) / 'reference.json'
            self.assertTrue(cv2.imwrite(str(source), image))
            annotation.write_text(json.dumps(reference), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'photo augmentation has not been adapted'):
                generator.generate(out, 3, 12, source, annotation, production_plan=self.plan(3))
            self.assertFalse(out.exists())
            _, _, _, info = generator.sample_reference(np.random.default_rng(12), image, reference)
            inverse = np.linalg.inv(np.asarray(info['parameters']['homography']))
            restored = generator.map_points(info['keypoints'], inverse)
            np.testing.assert_allclose(restored, [points[k] for k in generator.KEYPOINTS], atol=.01)
            self.assertEqual(info['values'], {'temperature': None, 'humidity': None})

    def test_memory_image_without_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'traceable file path'):
                generator.generate(Path(temp) / 'out', 1, 1, np.zeros((3, 3, 3)), {})

    def test_unplanned_duplicate_or_unsupported_states_are_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'out'
            with self.assertRaises(ValueError):
                generator.generate(out, 1, 1)
            plan = self.plan(2)
            plan['strata'][1]['parameters'] = dict(plan['strata'][0]['parameters'])
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                generator.generate(out, 2, 1, production_plan=plan)
            plan = self.plan(1)
            plan['constraints']['glare'] = 'high'
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                generator.generate(out, 1, 1, production_plan=plan)
            self.assertFalse(out.exists())

    def test_declared_parameters_control_actual_pixels_and_receipt(self):
        plan = self.plan(1)
        records = []
        @contextmanager
        def fixture_guard(*args, **kwargs):
            allocation = dict(kwargs['production_plan'])
            allocation['strata'] = [dict(s, planned_count=1) for s in allocation['strata']]
            def record(state_id, **values):
                self.assertTrue(values['output'].is_file())
                records.append((state_id, values))
            yield SimpleNamespace(reused=False, source_refs={}, plan=allocation, record_sample=record)
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(generator, 'generation_guard', fixture_guard):
            output = Path(temp) / 'batch'
            generator.generate(output, 1, 7, production_plan=plan)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0][1]['parameters'], plan['strata'][0]['parameters'])
            actual = records[0][1]['actual']
            self.assertEqual(actual['geometry']['yaw_deg'], 0)
            self.assertEqual(actual['glare_ellipses'], [])
            self.assertEqual(actual['brightness_gain'], 1)

    def test_reuse_and_database_failure_never_execute_generation(self):
        @contextmanager
        def reused(*args, **kwargs):
            yield SimpleNamespace(reused=True, source_refs={})

        @contextmanager
        def unavailable(*args, **kwargs):
            raise ValueError('SQLite cache missing')
            yield  # pragma: no cover

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'out'
            with mock.patch.object(generator, '_generate') as produce:
                with mock.patch.object(generator, 'generation_guard', reused):
                    generator.generate(output, 1, 1, production_plan=self.plan(1))
                with mock.patch.object(generator, 'generation_guard', unavailable):
                    with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
                        generator.generate(output, 1, 1, production_plan=self.plan(1))
                produce.assert_not_called()
                self.assertFalse(output.exists())

    def test_relative_output_is_guarded_at_the_same_resolved_path(self):
        captured = {}

        @contextmanager
        def reused(operation, inputs, outputs, parameters, **kwargs):
            captured['outputs'] = outputs
            yield SimpleNamespace(reused=True, source_refs={})

        with tempfile.TemporaryDirectory() as temp:
            previous = Path.cwd()
            try:
                os.chdir(temp)
                with mock.patch.object(generator, 'generation_guard', reused):
                    result = generator.generate('relative/out', 1, 1, production_plan=self.plan(1))
                self.assertEqual(captured['outputs'], [Path(temp).resolve() / 'relative/out'])
                self.assertEqual(result, captured['outputs'][0])
            finally:
                os.chdir(previous)

    def test_cli_reuse_is_not_reported_as_new_generation(self):
        @contextmanager
        def reused(*args, **kwargs):
            yield SimpleNamespace(reused=True, source_refs={})

        with tempfile.TemporaryDirectory(dir=generator.ROOT / 'tmp') as temp:
            out = io.StringIO()
            plan_path = Path(temp) / 'plan.json'
            plan_path.write_text(json.dumps(self.plan(16)), encoding='utf8')
            with mock.patch.object(generator, 'generation_guard', reused), \
                 mock.patch.object(generator, '_generate') as produce, \
                 mock.patch('sys.argv', ['entry', '--output', str(Path(temp) / 'batch'),
                                         '--production-plan', str(plan_path)]), \
                 redirect_stdout(out):
                generator.main()
            produce.assert_not_called()
            self.assertIn('REUSED', out.getvalue())
            self.assertNotIn('Generated', out.getvalue())


if __name__ == '__main__':
    unittest.main()
