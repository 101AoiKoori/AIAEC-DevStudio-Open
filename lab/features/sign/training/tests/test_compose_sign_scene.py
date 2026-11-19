import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'compose_sign_scene.py'
from datasets.scripts.sign import compose_sign_scene as module
sys.path.insert(0, str(SCRIPT.parent))
import compose_sign_v4_measured as measured


class TestSignComposition(unittest.TestCase):
    def test_scene_plan_checks_source_duplicates_and_unsupported_negatives(self):
        path = module.ROOT / 'datasets/example/background.png'
        frame = {'frame': '1'}
        origin = {'index': 20, 'source': 'datasets/example/frame.png'}
        state = {'id': 'front', 'background': {'image': path.relative_to(module.ROOT).as_posix()},
                 'target': {'kind': 'positive', 'mask': 'train_20.png'},
                 'labels': {'class_id': 0, 'kind': 'visible_fixture_box'},
                 'augmentation': {'gain': 1.0}, 'conditions': ['positive'], 'max_count': 1,
                 'parameters': {'height': 200, 'left': 100, 'top': 200},
                 'sources': [path.relative_to(module.ROOT).as_posix(),
                     'Archive/Assets/working_sources/legacy/site/场景/综合视频/空场与场景物料_001.mp4',
                     (module.ANNOTATED / 'train_20.png').relative_to(module.ROOT).as_posix(),
                     origin['source'], module.SIGN_VIDEO]}
        plan = {'model_key': 'sign', 'target_split': 'train', 'constraints': {
            'image_size': [960, 480], 'background_crop_xyxy': [480, 600, 1440, 1080],
            'label_kind': 'visible_fixture_box'}, 'strata': [state]}
        with mock.patch.object(module, 'read_image', side_effect=AssertionError('premature image read')):
            module._prepare_plan(plan, 'train', [(path, frame)], {'records': [origin]})
            for states in ([state, {**state, 'id': 'copy', 'augmentation': {'gain': 1}}], [{**state, 'max_count': 2}],
                           [{**state, 'target': {'kind': 'negative', 'mask': 'train_20.png'}}],
                           [{**state, 'augmentation': {'gain': 1.0, 'yaw': 5}}]):
                with self.subTest(states=states), self.assertRaises(ValueError):
                    module._prepare_plan({**plan, 'strata': states}, 'train', [(path, frame)], {'records': [origin]})

    def test_scene_generates_and_records_one_allocated_positive(self):
        state = {'id': 'one', 'planned_count': 1, 'background': {'image': 'bg.png'},
                 'target': {'mask': 'train_20.png'}, 'augmentation': {'gain': 1.0},
                 'parameters': {'height': 200, 'left': 100, 'top': 200}, 'sources': ['bg.png']}
        plan = {'target_split': 'train', 'strata': [state]}
        raw = np.zeros((1080, 1920, 3), dtype=np.uint8)
        sprite = np.full((100, 30, 4), 255, dtype=np.uint8)
        prepared = {'one': ((Path('bg.png'), {'frame': '1'}), {'source': 'frame.png'}, 'video.mp4')}
        run = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(module, 'EXPORTS', Path(tmp)), \
                mock.patch.object(module, 'read_image', return_value=raw), \
                mock.patch.object(module, 'cutout', return_value=sprite):
            output = Path(tmp) / 'batch'
            result = module._generate(output, plan, run, prepared)
            self.assertEqual(result['status'], 'complete_candidates')
            self.assertEqual(len((output / 'labels_candidate/sign_000001.txt').read_text().split()), 5)
            self.assertEqual(run.record_sample.call_count, 1)
            self.assertEqual(run.record_sample.call_args.kwargs['actual']['background_sign_absence'], 'unknown')

    def test_planned_stratum_accepts_exact_supported_placement(self):
        row = {'image': 'road.png', 'road_anchor_center_x': [140, 500]}
        state = {'background': {'image': 'road.png'},
                 'target': {'state': 'mid_full', 'mask': 'train_20.png'},
                 'labels': {'class_id': 0, 'kind': 'visible_fixture_box'},
                 'augmentation': {'tone': 'normal'},
                 'parameters': {'height': 300, 'left': 200, 'top': 120}}
        self.assertEqual(measured.validate_planned_stratum(state, [row]), row)
        far = {**state, 'target': {'state': 'far_full', 'mask': 'train_20.png'},
               'parameters': {'height': 180, 'left': 200, 'top': 240}}
        self.assertEqual(measured.validate_planned_stratum(far, [row]), row)
        with self.assertRaisesRegex(ValueError, 'Infeasible'):
            measured.validate_planned_stratum({**far, 'parameters': {'height': 159, 'left': 200, 'top': 261}}, [row])

    def test_planned_stratum_rejects_uncontrolled_or_infeasible_states(self):
        row = {'image': 'road.png', 'road_anchor_center_x': [140, 500]}
        valid = {'background': {'image': 'road.png'},
                 'target': {'state': 'mid_full', 'mask': 'train_20.png'},
                 'labels': {'class_id': 0, 'kind': 'visible_fixture_box'},
                 'augmentation': {'tone': 'normal'},
                 'parameters': {'height': 300, 'left': 200, 'top': 120}}
        for replacement in ({'parameters': {'height': 300, 'left': 200, 'top': -20}},
                            {'augmentation': {'tone': 'normal', 'yaw': 20}},
                            {'target': {'state': 'unknown', 'mask': 'train_20.png'}},
                            {'background': {'image': 'missing.png'}},
                            {'labels': {'class_id': 1, 'kind': 'visible_fixture_box'}}):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                measured.validate_planned_stratum({**valid, **replacement}, [row])

    def test_measured_generation_requires_plan_before_reading_inputs(self):
        with mock.patch.object(measured, '_prepare', side_effect=AssertionError('unexpected preparation')):
            with self.assertRaisesRegex(ValueError, 'production plan'):
                measured.generate(Path('unused'), 1, False)
            with self.assertRaisesRegex(ValueError, 'seed'):
                measured.generate(Path('unused'), 1, False, {'sampling_seed': 2})
            with self.assertRaisesRegex(ValueError, 'seed'):
                module.generate(Path('unused'), 'train', 1, 1, {'sampling_seed': 2})

    def test_negative_requires_observed_absence_and_writes_candidate_empty_label(self):
        row = {'image': 'road.png', 'parent_image': 'parent.png', 'pts_seconds': 1.0,
               'review_status': 'AI review candidate: no identifiable sign'}
        state = {'id': 'empty', 'max_count': 1, 'planned_count': 1,
                 'conditions': ['negative', 'none', 'normal'], 'background': {'image': 'road.png'},
                 'target': {'state': 'negative', 'mask': 'none'},
                 'labels': {'class_id': 0, 'kind': 'empty_candidate'}, 'augmentation': {'tone': 'normal'},
                 'parameters': {'kind': 'empty_background'},
                 'sources': [(measured.BACKGROUNDS / 'road.png').relative_to(measured.ROOT).as_posix(),
                             'parent.png', 'background.mp4']}
        declaration = {'model_key': 'sign', 'constraints': {'image_size': [640, 480],
                       'label_kind': 'visible_fixture_box'}, 'strata': [state]}
        prepared = measured._prepare(declaration, {'video': 'background.mp4'}, [row], {'records': []})
        with self.assertRaisesRegex(ValueError, 'absence candidate'):
            measured._prepare(declaration, {'video': 'background.mp4'},
                              [{**row, 'review_status': 'unknown'}], {'records': []})
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        run = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(measured, 'OUTPUT_ROOT', Path(tmp) / 'batch'), \
                mock.patch.object(measured.base, 'read_image', return_value=image):
            out = Path(tmp) / 'batch'
            result = measured._generate(out, {'target_split': 'train', 'strata': [state]}, run, prepared)
            self.assertEqual((out / 'labels_candidate/sign_00000.txt').read_text(), '')
            self.assertEqual(result['samples'][0]['review_status'], 'candidate')
            self.assertIsNone(run.record_sample.call_args.kwargs['actual']['bbox_xyxy'])

    def test_preparation_never_composes_and_rejects_duplicate_budget(self):
        row = {'image': 'road.png', 'parent_image': 'parent.png', 'road_anchor_center_x': [140, 500]}
        state = {'id': 'front', 'max_count': 1, 'conditions': ['mid_full', 'front', 'normal'],
                 'background': {'image': 'road.png'}, 'target': {'state': 'mid_full', 'mask': 'train_20.png'},
                 'labels': {'class_id': 0, 'kind': 'visible_fixture_box'}, 'augmentation': {'tone': 'normal'},
                 'parameters': {'height': 300, 'left': 200, 'top': 120},
                 'sources': [(measured.BACKGROUNDS / 'road.png').relative_to(measured.ROOT).as_posix(),
                             'parent.png', 'background.mp4',
                             (measured.base.ANNOTATED / 'train_20.png').relative_to(measured.ROOT).as_posix(),
                             'frame.png', measured.base.SIGN_VIDEO]}
        plan = {'model_key': 'sign', 'constraints': {'image_size': [640, 480],
                 'label_kind': 'visible_fixture_box'}, 'strata': [state]}
        with mock.patch.object(measured, 'compose', side_effect=AssertionError('premature composition')), \
                mock.patch.object(measured.base, 'read_image', side_effect=AssertionError('premature image read')):
            measured._prepare(plan, {'video': 'background.mp4'}, [row], {'records': [{'index': 20, 'source': 'frame.png'}]})
            with self.assertRaisesRegex(ValueError, 'at most one'):
                measured._prepare({**plan, 'strata': [{**state, 'max_count': 2}]},
                                  {'video': 'background.mp4'}, [row], {'records': [{'index': 20, 'source': 'frame.png'}]})
            with self.assertRaisesRegex(ValueError, 'Duplicate sign composition'):
                measured._prepare({**plan, 'strata': [state, {**state, 'id': 'another'}]},
                                  {'video': 'background.mp4'}, [row], {'records': [{'index': 20, 'source': 'frame.png'}]})

    def test_allocated_sample_records_real_label_and_actual_geometry(self):
        state = {'id': 'front', 'planned_count': 1,
                 'sources': ['background.png', 'sign.mp4'],
                 'target': {'mask': 'train_20.png', 'state': 'mid_full'},
                 'augmentation': {'tone': 'normal'},
                 'parameters': {'height': 300, 'left': 200, 'top': 120}}
        plan = {'target_split': 'train', 'strata': [state]}
        row = {'parent_image': 'parent.png', 'pts_seconds': 1.0}
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        row.update(image='road.png', road_anchor_center_x=[140, 500])
        prepared = {'front': (row, {'source': 'frame.png'})}
        run = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(measured, 'OUTPUT_ROOT', Path(tmp) / 'batch'), \
                mock.patch.object(measured.base, 'read_image', return_value=image), \
                mock.patch.object(measured.base, 'cutout', return_value=image), \
                mock.patch.object(measured, 'compose', return_value=(image, [200, 120, 250, 420], {'left': 200})):
            output = Path(tmp) / 'batch'
            result = measured._generate(output, plan, run, prepared, sampling_seed=907)
            self.assertEqual(result['status'], 'complete_candidates')
            label = (output / 'labels_candidate/sign_00000.txt').read_text(encoding='utf8')
            self.assertTrue(label.endswith('\n'))
            self.assertEqual(len(label.split()), 5)
            self.assertEqual(json.loads((output / 'manifest.json').read_text(encoding='utf8'))['samples'][0]['state_id'], 'front')
            self.assertEqual(run.record_sample.call_count, 1)
            self.assertEqual(run.record_sample.call_args.kwargs['actual']['bbox_xyxy'], [200, 120, 250, 420])
            self.assertEqual(run.record_sample.call_args.kwargs['actual']['sampling_seed'], 907)

    def test_yolo_full_frame_and_invalid_boxes(self):
        self.assertEqual(module.yolo([0, 0, 960, 480]), '0 0.500000 0.500000 1.000000 1.000000')
        for box in ([-1, 0, 1, 1], [1, 1, 1, 2], [0, 0, 961, 480]):
            with self.assertRaises(ValueError):
                module.yolo(box)

    def test_negative_has_no_box(self):
        background = np.full((480, 960, 3), 64, dtype=np.uint8)
        result, box = module.make_image(background, None, random.Random(1), 'train')
        self.assertIsNone(box)
        np.testing.assert_array_equal(result, background)

    def test_positive_box_fits_visible_pixels(self):
        background = np.zeros((480, 960, 3), dtype=np.uint8)
        sprite = np.full((100, 30, 4), 255, dtype=np.uint8)
        for split in ('train', 'val', 'test'):
            result, box = module.make_image(background, sprite, random.Random(42), split)
            self.assertTrue(np.any(result))
            module.yolo(box)
            if split == 'val':
                self.assertEqual(box[1], 0)


if __name__ == '__main__':
    unittest.main()
