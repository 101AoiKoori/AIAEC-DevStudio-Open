"""Exercise short needles, missing points, denominators and packaged Pose contracts."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/training_common'))
from engine.meter_report import POINTS, meter_pose_report, write_meter_report
from engine.config import load_config
from engine.runner import _training_options
from engine.training_policy import _ValidationStopper


def row(name='a', *, matched=True):
    xy = [[50, 50], [60, 50], [50, 70], [50, 75], [10, 80], [90, 80], [30, 90], [70, 90]]
    gt = [[x/100, y/100, 1] for x, y in xy]
    pred = [[x, y, .9] for x, y in xy]
    return {'image': name, 'shape_hw': [100, 100],
            'ground_truth': [{'class_id': 0, 'xyxy': [0, 0, 100, 100], 'keypoints': gt}],
            'predictions': [{'class_id': 0, 'keypoints': pred}] if matched else [],
            'matches': [[0, 0, 1.0]] if matched else [],
            'missed_gt_indices': [] if matched else [0], 'false_prediction_indices': []}


class MeterPoseReportTests(unittest.TestCase):
    def test_two_needles_independent_and_normalized_points(self):
        image = row()
        image['predictions'][0]['keypoints'][3][:2] = [55, 70]  # Humidity rotates 90 degrees.
        result, details = meter_pose_report([image])
        self.assertEqual(result['needles']['temperature']['angle_error_degrees']['mean'], 0)
        self.assertAlmostEqual(result['needles']['humidity']['angle_error_degrees']['mean'], 90)
        self.assertAlmostEqual(result['points'][3]['error_pixels']['mean'], math.sqrt(50))
        self.assertAlmostEqual(result['points'][3]['error_dial_short_side_fraction']['mean'], math.sqrt(50)/100)
        self.assertEqual(result['by_needle_length_pixels']['humidity']['under10']['measured'], 1)
        self.assertEqual(len(details), 2)

    def test_misses_low_confidence_unknown_and_negative_do_not_become_zero_errors(self):
        low = row('low')
        low['predictions'][0]['keypoints'][3] = [0, 0, .1]
        unknown = row('unknown')
        unknown['ground_truth'][0]['keypoints'][3][2] = 0
        negative = {'image': 'negative', 'shape_hw': [100, 100], 'ground_truth': [],
                    'predictions': [{'class_id': 0}], 'matches': [],
                    'missed_gt_indices': [], 'false_prediction_indices': [0]}
        result, _ = meter_pose_report([row('good'), row('missed', matched=False), low, unknown, negative])
        point = result['points'][3]
        self.assertEqual((point['eligible'], point['measured'], point['box_missed'], point['point_unavailable']), (3, 1, 1, 1))
        humidity = result['needles']['humidity']
        self.assertEqual((humidity['eligible'], humidity['measured'], humidity['unmeasured']), (3, 1, 2))
        self.assertEqual(humidity['within_angle_degrees']['5']['fraction_of_eligible'], 1/3)
        self.assertEqual(result['negative_image_false_positive_rate'], 1)

    def test_wrapped_angles_and_degenerate_prediction(self):
        image = row()
        image['ground_truth'][0]['keypoints'][1][:2] = [.4, .5001]
        image['predictions'][0]['keypoints'][1][:2] = [40, 49.99]
        image['predictions'][0]['keypoints'][3][:2] = [50, 70]
        result, _ = meter_pose_report([image])
        self.assertLess(result['needles']['temperature']['angle_error_degrees']['mean'], .2)
        self.assertEqual(result['needles']['humidity']['degenerate_prediction'], 1)
        self.assertIsNone(result['needles']['humidity']['angle_error_degrees']['mean'])

    def test_bad_shape_partition_and_nonfinite_rejected(self):
        bad = row(); bad['predictions'][0]['keypoints'][0][0] = float('nan')
        with self.assertRaises(ValueError): meter_pose_report([bad])
        bad = row(); bad['ground_truth'][0]['keypoints'].pop()
        with self.assertRaises(ValueError): meter_pose_report([bad])
        bad = row(); bad['missed_gt_indices'] = [0]
        with self.assertRaises(ValueError): meter_pose_report([bad])
        with self.assertRaises(ValueError): meter_pose_report([row(), row()])

    def test_writes_from_saved_predictions_without_inference(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            evaluation = Path(temp); (evaluation/'predictions').mkdir()
            (evaluation/'summary.json').write_text(json.dumps({'model_key': 'meter', 'task': 'pose', 'split': 'val', 'images': 1}))
            (evaluation/'predictions/predictions.jsonl').write_text(json.dumps(row())+'\n')
            path = write_meter_report(evaluation)
            self.assertEqual(json.loads(path.read_text())['keypoint_order'], list(POINTS))
            self.assertTrue((evaluation/'meter_needle_errors.csv').is_file())
            summary = json.loads((evaluation/'summary.json').read_text()); summary['images'] = 2
            (evaluation/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'coverage'): write_meter_report(evaluation)

    def test_actual_training_configuration_and_early_stop_interval(self):
        config = load_config(ROOT, 'lab/features/meter/training/config/meter_v4.yaml')
        self.assertEqual(tuple(config['keypoint_names']), POINTS)
        self.assertEqual(config['epochs'], 150)
        options = _training_options(config, ROOT/'data.yaml', ROOT/'tmp')
        self.assertEqual(options['pose'], 12)
        self.assertEqual(options['kobj'], 1)
        for key in ('fliplr', 'flipud', 'mosaic', 'mixup', 'copy_paste', 'perspective'):
            self.assertEqual(options[key], 0)
        policy = config['training_policy']
        stopper = _ValidationStopper(policy['min_epochs'], policy['patience'], policy['min_delta'])
        for epoch in range(1, 120): self.assertFalse(stopper(epoch, .5))
        self.assertTrue(stopper(120, .5))
        self.assertFalse(stopper(121, .6))
        launcher = yaml.safe_load((ROOT/'lab/training_common/launcher.yaml').read_text())
        self.assertEqual(launcher['tasks']['meter']['wall_seconds'], 36000)


if __name__ == '__main__':
    unittest.main()
