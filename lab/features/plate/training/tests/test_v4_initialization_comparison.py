"""Verify same-development-set comparison without auto-selecting weights."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'train_plate_detection_v4.py'
SPEC = importlib.util.spec_from_file_location('plate_v4', SCRIPT)
plate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plate)


def write_evaluation(root, name, checkpoint):
    run = root / name
    evaluation = run / 'evaluation/val'
    evaluation.mkdir(parents=True)
    (run / 'data/images/val').mkdir(parents=True)
    (run / 'data/labels/val').mkdir(parents=True)
    (run / 'metadata').mkdir()
    (run / 'data/images/val/a.png').write_bytes(b'fixture image')
    (run / 'data/labels/val/a.txt').write_text('0 0.5 0.5 0.1 0.1\n', encoding='utf-8')
    samples = [{'split': 'val', 'image': 'same-source.png', 'run_image': 'data/images/val/a.png',
                'run_label': 'data/labels/val/a.txt'}]
    (run / 'data/samples.json').write_text(json.dumps(samples), encoding='utf-8')
    summary = {'model_key': 'plate_detection', 'split': 'val', 'checkpoint': checkpoint,
               'evaluation_config': {'imgsz': 640, 'match_iou': 0.5}, 'images': 1,
               'annotated_instances': 1, 'metrics': {'map': 0.3},
               'diagnostics': {'nms_iou': 0.7, 'true_positives': 1, 'false_positives': 0,
                               'false_negatives': 0, 'negative_images': 0,
                               'negative_images_with_false_positives': 0}}
    (evaluation / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    environment = {'python': 'same Python', 'packages': {name: 'same version' for name in
        ('torch', 'torchvision', 'ultralytics', 'numpy', 'opencv-python')}}
    (run / 'metadata/environment_evaluate_val.json').write_text(json.dumps(environment), encoding='utf-8')
    return evaluation


class PlateComparisonTests(unittest.TestCase):
    def test_requires_same_staged_bytes_and_does_not_select(self):
        parent = plate.ROOT / 'tmp'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root = Path(temp)
            left = write_evaluation(root, 'left', 'weights/provincial.pt')
            right = write_evaluation(root, 'right', 'weights/returned_v3.pt')
            result = plate.compare_initializations(left, right)
            self.assertEqual(result['compared_images'], 1)
            self.assertIsNone(result['selected_checkpoint'])
            (root / 'right/data/labels/val/a.txt').write_text('different', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                plate.compare_initializations(left, right)

    def test_test_split_rejected(self):
        parent = plate.ROOT / 'tmp'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            evaluation = write_evaluation(Path(temp), 'run', 'weights/x.pt')
            path = evaluation / 'summary.json'
            summary = json.loads(path.read_text(encoding='utf-8'))
            summary['split'] = 'test'
            path.write_text(json.dumps(summary), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'not test'):
                plate.development_records(evaluation)


if __name__ == '__main__':
    unittest.main()
