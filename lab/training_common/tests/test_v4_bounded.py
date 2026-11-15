"""Check V4 step accounting and forced-stop state with a bounded fixture."""
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.bounded import BudgetExhausted
from engine.runner import bounded_train
from engine.telemetry import UpdateRecorder


class V4BoundedTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[3] / 'tmp'
        base.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {'schema_version': 3, 'task_id': 'person', 'model_key': 'person',
                       'version': 'v4', 'task': 'detect', 'data': 'dataset/data.yaml',
                       'model': 'model.pt', 'output_root': 'runs/person', 'imgsz': 640,
                       'epochs': 10, 'batch': 2}

    def test_actual_optimizer_steps_and_learning_rate_are_recorded(self):
        import torch
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        trainer = types.SimpleNamespace(optimizer=optimizer, epoch=0, train_loader=[1, 2],
                                        batch_size=2, accumulate=2)
        recorder = UpdateRecorder(self.root, epoch_budget=3)
        recorder.attach(trainer)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.param_groups[0]['lr'] = 0.03
        trainer.epoch = 1
        optimizer.step()
        recorder.close('completed_epoch_budget', trainer)
        events = [json.loads(line) for line in (self.root / 'metadata/optimizer_updates.jsonl').read_text().splitlines()]
        self.assertEqual([event['step'] for event in events], [1, 2])
        self.assertEqual([event['epoch'] for event in events], [1, 2])
        self.assertEqual([event['learning_rates'] for event in events], [[0.1], [0.03]])
        self.assertEqual([event['gradient_accumulation'] for event in events], [2, 2])
        summary = json.loads((self.root / 'metadata/optimization_summary.json').read_text())
        self.assertEqual(summary['actual_optimizer_steps'], 2)
        budget = json.loads((self.root / 'metadata/optimization_budget.json').read_text())
        self.assertEqual(budget['microbatches_per_epoch'], 2)
        self.assertEqual(budget['optimizer_step_upper_bound'], 6)

    def test_forced_stop_records_budget_and_unknown_actual_steps(self):
        command = [sys.executable, '-B', '-c', 'import time; time.sleep(10)']
        start = time.monotonic()
        with patch('engine.runner.child_command', return_value=command):
            with self.assertRaises(BudgetExhausted):
                bounded_train(self.root, self.config, 'config.yaml', 0.2, 'fixture')
        self.assertLess(time.monotonic() - start, 10)
        run = self.root / 'runs/person/fixture/metadata'
        self.assertEqual(json.loads((run / 'train_status.json').read_text())['status'], 'budget_exhausted')
        self.assertEqual(json.loads((run / 'final_test_status.json').read_text())['status'], 'not_executed')
        self.assertIsNone(json.loads((run / 'optimization_summary.json').read_text())['actual_optimizer_steps'])
        controller = next((self.root / 'runs/person/_controllers').iterdir())
        self.assertEqual(json.loads((controller / 'status.json').read_text())['status'], 'budget_exhausted')

    def test_invalid_wall_limit_rejected_before_start(self):
        for seconds in (0, -1, float('inf'), float('nan')):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                bounded_train(self.root, self.config, 'config.yaml', seconds)


if __name__ == '__main__':
    unittest.main()
