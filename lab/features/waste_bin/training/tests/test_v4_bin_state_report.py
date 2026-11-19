import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('bin_v4', Path(__file__).resolve().parents[1] / 'scripts/train_waste_bin_state_v4.py')
bin_state = importlib.util.module_from_spec(spec); spec.loader.exec_module(bin_state)
from engine.bin_state_report import bin_state_report, match_whole_bins, write_bin_state_report


class BinStateReportTests(unittest.TestCase):
    def test_opposite_state_by_reviewed_color(self):
        box = [0, 0, 10, 10]
        rows = [{'source_image': 'x', 'ground_truth': [{'class_id': 0, 'xyxy': box}],
                 'predictions': [{'class_id': 1, 'xyxy': box}, {'class_id': 1, 'xyxy': box}],
                 'matches': [], 'missed_gt_indices': [0], 'false_prediction_indices': [0, 1]}]
        review = [{'source_image': 'x', 'gt_index': 0, 'color': 'red', 'review_status': 'confirmed', 'review_evidence': 'a'}]
        result = bin_state.reviewed_color_states(rows, review, .5)
        self.assertEqual(result['colors']['red']['open']['missed'], 1)
        self.assertEqual(result['colors']['red']['open']['opposite_state'], 1)
        self.assertIsNone(result['colors']['gray']['closed']['recall'])

    def test_no_review_fails(self):
        with self.assertRaises(ValueError): bin_state.reviewed_color_states([], [], .5)

    def test_legacy_report_accepts_real_black_bins(self):
        box = [0, 0, 10, 10]
        rows = [{'source_image': 'x', 'ground_truth': [{'class_id': 0, 'xyxy': box}],
                 'predictions': [], 'matches': [], 'missed_gt_indices': [0], 'false_prediction_indices': []}]
        reviews = [{'source_image': 'x', 'gt_index': 0, 'color': 'black',
                    'review_status': 'confirmed', 'review_evidence': 'a'}]
        self.assertEqual(bin_state.reviewed_color_states(rows, reviews, .5)['colors']['black']['open']['missed'], 1)

    def test_highest_iou_ignores_state_and_counts_duplicate_once(self):
        truth = [{'class_id': 0, 'xyxy': [0, 0, 10, 10]}]
        predictions = [{'class_id': 1, 'xyxy': [0, 0, 10, 10]},
                       {'class_id': 0, 'xyxy': [0, 0, 9, 10]}]
        matches, missed, extra, duplicates = match_whole_bins(truth, predictions, .5)
        self.assertEqual(matches, [(0, 0, 1.0)])
        self.assertEqual((missed, extra, duplicates), ([], [1], [1]))

    def test_opposite_state_miss_and_false_prediction_are_exclusive(self):
        truths = [{'class_id': 0, 'xyxy': [0, 0, 10, 10]}, {'class_id': 1, 'xyxy': [30, 30, 40, 40]}]
        row = {'source_image': 'test.png', 'shape_hw': [100, 100], 'ground_truth': truths,
               'predictions': [{'class_id': 1, 'xyxy': [0, 0, 10, 10]},
                               {'class_id': 1, 'xyxy': [0, 0, 10, 10]},
                               {'class_id': 0, 'xyxy': [70, 70, 80, 80]}]}
        record = {'image_size': [100, 100], 'source_kind': 'real_background_composite',
                  'bins': [{'color': 'black', 'state': 'open', 'bbox_xyxy': truths[0]['xyxy']},
                           {'color': 'red', 'state': 'closed', 'bbox_xyxy': truths[1]['xyxy']}],
                  'allocation_axes': {'view': 'oblique'}}
        report = bin_state_report([row], {'test.png': [record]}, .5)
        totals = report['totals']
        self.assertEqual((totals['correct_state'], totals['opposite_state'], totals['localization_miss']), (0, 1, 1))
        self.assertEqual(totals['instances'], 2)
        self.assertEqual(totals['localization_recall'], .5)
        self.assertEqual((report['duplicate_predictions'], report['other_unmatched_predictions']), (1, 1))
        self.assertEqual(report['by_dimension']['color']['black']['opposite_state'], 1)
        self.assertEqual(report['by_dimension']['view']['oblique']['instances'], 2)
        self.assertEqual(report['by_dimension']['tone']['unknown']['instances'], 2)

    def test_manifest_metadata_must_bind_to_gt_coordinates_and_row_order(self):
        truth = {'class_id': 0, 'xyxy': [1, 1, 10, 20]}
        row = {'source_image': 'x', 'shape_hw': [100, 100], 'ground_truth': [truth], 'predictions': []}
        record = {'image_size': [100, 100],
                  'bins': [{'color': 'black', 'state': 'open', 'bbox_xyxy': [30, 30, 40, 50]}]}
        with self.assertRaisesRegex(ValueError, 'bind'):
            bin_state_report([row], {'x': [record]}, .5)
        record['bins'][0]['bbox_xyxy'] = truth['xyxy']
        record['image_size'] = [99, 100]
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            bin_state_report([row], {'x': [record]}, .5)

    def test_packaged_manifest_resolves_images_and_repeat_is_idempotent(self):
        workspace = Path(__file__).resolve().parents[5]
        with tempfile.TemporaryDirectory(dir=workspace / 'tmp') as temporary:
            root = Path(temporary)
            dataset = root / 'dataset'
            dataset.mkdir()
            evaluation = root / 'runs/evaluation/test'
            (evaluation / 'predictions').mkdir(parents=True)
            record = {'image': 'dataset/images/test/a.png', 'split': 'test', 'image_size': [100, 100],
                      'source_kind': 'direct_real',
                      'bins': [{'color': 'black', 'state': 'open', 'bbox_xyxy': [1, 1, 10, 20]}]}
            manifest = dataset / 'workspace_sources.json'
            manifest.write_text(json.dumps({'model_key': 'waste_bin_state', 'records': [record]}), encoding='utf8')
            summary = {'model_key': 'waste_bin_state', 'split': 'test', 'diagnostics': {'match_iou': .5},
                       'evaluation_config': {'imgsz': 640}}
            (evaluation / 'summary.json').write_text(json.dumps(summary), encoding='utf8')
            row = {'source_image': record['image'], 'shape_hw': [100, 100],
                   'ground_truth': [{'class_id': 0, 'xyxy': [1, 1, 10, 20]}],
                   'predictions': [{'class_id': 1, 'xyxy': [1, 1, 10, 20]}]}
            (evaluation / 'predictions/predictions.jsonl').write_text(json.dumps(row) + '\n', encoding='utf8')
            report_path = write_bin_state_report(evaluation, manifest, root=root)
            self.assertEqual(write_bin_state_report(evaluation, manifest, root=root), report_path)
            report = json.loads(report_path.read_text(encoding='utf8'))
            self.assertEqual(report['totals']['opposite_state'], 1)
            self.assertEqual(report['by_dimension']['color']['black']['instances'], 1)


if __name__ == '__main__':
    unittest.main()
