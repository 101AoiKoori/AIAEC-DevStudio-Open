"""Exercise the V4 validation/freeze/final-test control flow with fixture data."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'lab/training_common'))
from engine.data import stage_dataset
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
        for split, color in (('train', 'red'), ('val', 'green'), ('test', 'blue')):
            for kind in ('images', 'labels'):
                (data / kind / split).mkdir(parents=True)
            Image.new('RGB', (20, 16), color=color).save(data / 'images' / split / 'a.png')
            label = '0 0.5 0.5 0.5 0.5\n' if split == 'val' else ''
            (data / 'labels' / split / 'a.txt').write_text(label, encoding='utf-8')
        (data / 'data.yaml').write_text(yaml.safe_dump({'names': ['person'],
            **{split: 'images/' + split for split in ('train', 'val', 'test')}}), encoding='utf-8')
        (data / 'workspace_sources.json').write_text(json.dumps({'model_key': 'person', 'records': [
            {'image': f'dataset/images/{split}/a.png', 'label': f'dataset/labels/{split}/a.txt',
             'split': split, 'label_status': 'reviewed',
             'review_state': 'complete_positive' if split == 'val' else 'complete_negative',
             'review_evidence': 'dataset/workspace_sources.json',
             'source_review_evidence': 'dataset/workspace_sources.json'}
            for split in ('train', 'val', 'test')
        ]}), encoding='utf-8')
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
        (self.root / 'dataset/images/test/a.png').write_bytes(b'changed after freezing')
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

    def test_training_snapshot_can_freeze_configured_holdout_in_same_run(self):
        data = self.root / 'dataset'
        input_test = data / 'images/test/a.png'
        original = input_test.read_bytes()
        self.config['training_policy'] = {'mode': 'fixed_validation', 'save_period': 1}
        stage_dataset(self.root, self.config, self.run)
        self.evaluate('val', self.weights, 'selection_val')
        validation = self.run / 'metadata/evaluation_inputs/selection_val/data'
        self.assertFalse((validation / 'images/test').exists())
        selected = freeze_selection(self.root, self.config, self.run, self.weights, 'selection_val')
        frozen = self.run / 'frozen_selection/data'
        input_test.write_bytes(b'changed after freezing')
        self.assertEqual((frozen / 'images/test/test_000000_a.png').read_bytes(), original)
        self.assertFalse((self.run / 'data/images/test').exists())
        self.evaluate('test', selected, 'test')
        self.assertEqual(self.used[-1][1], frozen / 'data.yaml')


if __name__ == '__main__':
    unittest.main()
