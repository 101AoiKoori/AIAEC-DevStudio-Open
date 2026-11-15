import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS = ROOT / 'lab/features/sign/training/scripts'
sys.path[:0] = [str(ROOT), str(SCRIPTS)]
spec = importlib.util.spec_from_file_location('prepare_sign_car_candidate', SCRIPTS / 'prepare_sign_car_candidate.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CarCandidateTests(unittest.TestCase):
    def test_converts_candidate_and_rejects_nonexhaustive_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'datasets/raw/P05.png'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'fixture')
            ann, obs, output = root / 'annotations.json', root / 'observations.jsonl', root / 'output'
            row = {'image': 'datasets/raw/P05.png', 'model_key': 'sign', 'annotation_level': 'candidate',
                   'boxes': [{'class_id': 0, 'xyxy': [10, 20, 30, 60]}], 'image_size': [100, 100]}
            ann.write_text(json.dumps([{}] * 6 + [row]))
            obs.write_text(json.dumps({'path': row['image'], 'task_observations': {'sign': {'exhaustive': True}}}))

            @contextmanager
            def fake_guard(*args, **kwargs):
                yield SimpleNamespace(reused=False)

            with mock.patch.multiple(module, ROOT=root, IMAGE=image, ANNOTATIONS=ann, OBSERVATIONS=obs, OUTPUT=output,
                                     guarded=fake_guard):
                result = module.main()
                self.assertEqual(result['status'], 'candidate_not_verified')
                self.assertEqual((output / 'labels/P05_NOENTRY.txt').read_text().split(),
                                 ['0', '0.20000000', '0.40000000', '0.20000000', '0.40000000'])
                obs.write_text(json.dumps({'path': row['image'], 'task_observations': {'sign': {'exhaustive': False}}}))
                with self.assertRaisesRegex(ValueError, 'not recorded as exhaustive'):
                    module.main()


if __name__ == '__main__':
    unittest.main()
