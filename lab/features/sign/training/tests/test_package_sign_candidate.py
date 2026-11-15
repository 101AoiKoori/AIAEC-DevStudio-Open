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
sys.path.insert(0, str(ROOT))
script = ROOT / 'datasets/scripts/sign/package_sign_candidate.py'
spec = importlib.util.spec_from_file_location('package_sign_candidate', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CandidatePackageTests(unittest.TestCase):
    def test_packages_computed_candidate_labels_into_declared_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'datasets/samples/sign'
            output = root / 'datasets/exports/sign/test'
            (source / 'images').mkdir(parents=True)
            (source / 'labels_candidate').mkdir()
            (source / 'images/a.png').write_bytes(b'fixture')
            (source / 'labels_candidate/a.txt').write_text('', encoding='utf8')
            sample = {'id': 'a', 'state_id': 'empty', 'image': 'images/a.png',
                      'label': 'labels_candidate/a.txt', 'sources': ['source'],
                      'actual': {'bbox_xyxy': None}, 'review_status': 'candidate'}
            (source / 'manifest.json').write_text(json.dumps({'status': 'complete_candidates', 'samples': [sample]}))
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'model_key': 'sign', 'budget': 1, 'target_split': 'test'}))

            @contextmanager
            def guard(*args, **kwargs):
                yield SimpleNamespace(reused=False)

            with mock.patch.object(module, 'ROOT', root), mock.patch.object(module, 'generation_guard', guard), \
                    mock.patch.object(module, 'sample_dir', return_value=source), \
                    mock.patch.object(module, 'split_dir', return_value=output), \
                    mock.patch.object(module, 'load_task_config', return_value=({}, {})):
                result = module.package(source, plan, output)
            self.assertEqual(result['status'], 'complete_candidates')
            self.assertEqual(result['samples'][0]['label_status'], 'candidate')
            self.assertEqual(result['split'], 'test')
            self.assertEqual(result['counts']['candidate_empty_labels'], 1)
            self.assertEqual((output / 'labels/a.txt').read_text(), '')
            self.assertFalse((output / 'data.yaml').exists())


class CandidateAdmissionTests(unittest.TestCase):
    def test_incomplete_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            module.require_usable_labels({'samples': [{'id': 'x', 'scope_complete': False}]})


if __name__ == '__main__':
    unittest.main()
