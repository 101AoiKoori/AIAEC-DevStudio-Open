"""Exercise the V4 validation/freeze/final-test control flow with fixture data."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'lab/training_common'))
from engine.freeze_gate import freeze_selection
from engine.runner import evaluate_run
from engine.runtime import new_run


class V4EvaluationFlowTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / 'tmp'; base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {'schema_version': 3, 'task_id': 'person', 'model_key': 'person',
                       'version': 'v4', 'task': 'detect', 'data': 'dataset/data.yaml',
                       'model': 'weights/model.pt', 'output_root': 'runs/person',
                       'imgsz': 640, 'epochs': 1, 'batch': 1}
        self.weights = self.root / 'weights/model.pt'; self.weights.parent.mkdir()
        self.weights.write_bytes(b'fixture checkpoint')
        data = self.root / 'dataset'
        for split in ('train', 'val', 'test'):
            for kind in ('images', 'labels'):
                (data / kind / split).mkdir(parents=True)
            (data / 'images' / split / 'a.png').write_bytes(('fixture ' + split).encode())
            (data / 'labels' / split / 'a.txt').write_text('', encoding='utf-8')
        (data / 'data.yaml').write_text(yaml.safe_dump({'names': ['person'],
            **{split: 'images/' + split for split in ('train', 'val', 'test')}}), encoding='utf-8')
        self.run = new_run(self.root, self.config, 'flow')
        self.used = []

    def fake_evaluate(self, model, config, run_dir, data_path, split, name):
        self.used.append((Path(model.checkpoint), Path(data_path), split))
        destination = Path(run_dir) / 'evaluation' / name
        destination.mkdir(parents=True, exist_ok=False)
        result = {'model_key': 'person', 'version': 'v4', 'task': 'detect',
                  'split': split, 'images': 1, 'metrics': {'fixture_only': 0.0}}
        (destination / 'summary.json').write_text(json.dumps(result), encoding='utf-8')
        return result

    def evaluate(self, split, weights, name):
        with patch('engine.runner.prepare_ultralytics', return_value=lambda path, task: types.SimpleNamespace(checkpoint=path)), \
                patch('engine.runner._evaluate_loaded', side_effect=self.fake_evaluate):
            return evaluate_run(self.root, self.config, weights, split, self.run, name)

    def test_validation_snapshot_freeze_and_test_are_distinct(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.evaluate('test', self.weights, 'test')
        status = self.run / 'metadata/final_test_status.json'
        self.assertEqual(json.loads(status.read_text())['status'], 'not_executed')
        self.evaluate('val', self.weights, 'selection_val')
        self.assertIn('evaluation_inputs', str(self.used[-1][0]))
        self.assertIn('evaluation_inputs', str(self.used[-1][1]))
        selected = freeze_selection(self.root, self.config, self.run, self.weights, 'selection_val', 'fixture only')
        (self.run / 'data/images/test/test_000000_a.png').write_bytes(b'changed after validation')
        self.evaluate('test', selected, 'test')
        self.assertIn('frozen_selection', str(self.used[-1][0]))
        self.assertIn('frozen_selection', str(self.used[-1][1]))
        self.assertEqual(json.loads(status.read_text())['status'], 'complete')
        with self.assertRaises((ValueError, FileExistsError)):
            self.evaluate('test', selected, 'test_again')

    def test_test_failure_is_not_success(self):
        self.evaluate('val', self.weights, 'selection_val')
        selected = freeze_selection(self.root, self.config, self.run, self.weights, 'selection_val', 'fixture only')
        with patch('engine.runner.prepare_ultralytics', return_value=lambda path, task: object()), \
                patch('engine.runner._evaluate_loaded', side_effect=RuntimeError('fixture error')):
            with self.assertRaisesRegex(RuntimeError, 'fixture error'):
                evaluate_run(self.root, self.config, selected, 'test', self.run, 'test')
        status = json.loads((self.run / 'metadata/final_test_status.json').read_text())
        self.assertEqual(status['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
