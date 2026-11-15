import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = ROOT / 'lab/features/e_bike/training/scripts/compare_sam2_candidates.py'
spec = importlib.util.spec_from_file_location('compare_sam2_candidates', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CandidateValidationTest(unittest.TestCase):
    def test_restricts_frame_path_class_and_box(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            review = Path(directory) / 'review'
            (review / 'images').mkdir(parents=True)
            frame = review / 'images' / 'frame.jpg'
            frame.write_bytes(b'frame')
            external = review / 'outside.jpg'
            external.write_bytes(b'outside')
            annotation = {'image': 'images/frame.jpg', 'class': 'upright',
                          'bbox_xyxy_normalized': [0.1, 0.2, 0.8, 0.9]}
            self.assertEqual(module.validate_annotation(review, annotation), frame.resolve())
            for image in ('outside.jpg', 'images/../outside.jpg', str(external.resolve())):
                with self.subTest(image=image), self.assertRaises(ValueError):
                    module.validate_annotation(review, {**annotation, 'image': image})
            with self.assertRaises(ValueError):
                module.validate_annotation(review, {**annotation, 'class': '../outside'})
            with self.assertRaises(ValueError):
                module.validate_annotation(review, {**annotation, 'bbox_xyxy_normalized': [0, 0, 2, 1]})

    def test_model_failure_records_incomplete_state(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            review = root / 'review'
            (review / 'images').mkdir(parents=True)
            (review / 'images/frame.jpg').write_bytes(b'frame')
            annotation = {'image': 'images/frame.jpg', 'class': 'fallen',
                          'bbox_xyxy_normalized': [0.1, 0.2, 0.8, 0.9]}
            (review / 'box_candidates.json').write_text(json.dumps({'annotations': [annotation]}), encoding='utf-8')
            weights = root / 'sam2_t.pt'
            weights.write_bytes(b'dummy')
            output = root / 'output'
            args = argparse.Namespace(review=str(review), model=str(weights), output=str(output), indices=[1], device='0')
            parser = argparse.ArgumentParser()
            with mock.patch.object(module, 'SAM', side_effect=RuntimeError('model failed')):
                with self.assertRaises(SystemExit) as exit_code:
                    module._compare(args, parser)
            self.assertEqual(exit_code.exception.code, 2)
            manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['device'], '0')
            self.assertIn('model failed', manifest['error'])
            self.assertEqual(manifest['records'], [])


if __name__ == '__main__':
    unittest.main()
