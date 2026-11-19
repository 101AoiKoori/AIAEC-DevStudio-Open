"""Check V4 step accounting and forced-stop state with a bounded fixture."""
import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.bounded import BudgetExhausted, run_bounded
from engine.cli import main as training_cli
from engine.runner import _restore_checkpoint_outputs, bounded_train, train_run
from engine.runtime import training_lock, write_json
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

    def test_timeout_preserves_reason_with_a_truncated_old_status(self):
        metadata = self.root / 'runs/person/fixture/metadata'

        def simulate_timeout(*_, **__):
            metadata.mkdir(parents=True)
            (metadata / 'train_status.json').write_text('{"status":', encoding='utf-8')
            (metadata / 'optimizer_updates.jsonl').write_text('{"step": 1}\n{"step":', encoding='utf-8')
            raise BudgetExhausted('test deadline')

        with patch('engine.runner.run_bounded', side_effect=simulate_timeout):
            with self.assertRaisesRegex(BudgetExhausted, 'test deadline'):
                bounded_train(self.root, self.config, 'config.yaml', 60, 'fixture')
        status = json.loads((metadata / 'train_status.json').read_text(encoding='utf-8'))
        self.assertEqual((status['status'], status['error']), ('budget_exhausted', 'test deadline'))
        summary = json.loads((metadata / 'optimization_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['recorded_optimizer_steps'], 1)
        self.assertIsNone(summary['actual_optimizer_steps'])

    def test_training_output_is_displayed_while_the_child_is_running(self):
        output = io.StringIO()
        log = self.root / 'child.log'
        command = [sys.executable, '-u', '-c',
                   "import time; print('first', flush=True); time.sleep(1); print('last', flush=True)"]
        with contextlib.redirect_stdout(output):
            worker = threading.Thread(target=run_bounded, args=(command, time.monotonic() + 10, log, self.root),
                                      kwargs={'display_output': True})
            worker.start()
            deadline = time.monotonic() + 5
            while 'first' not in output.getvalue() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertIn('first', output.getvalue())
            self.assertTrue(worker.is_alive())
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(output.getvalue().splitlines(), ['first', 'last'])
        self.assertEqual(log.read_text(encoding='utf-8').splitlines(), ['first', 'last'])

    def test_displayed_training_output_keeps_the_wall_limit(self):
        output = io.StringIO()
        log = self.root / 'timeout.log'
        command = [sys.executable, '-u', '-c', "import time; print('started', flush=True); time.sleep(20)"]
        with contextlib.redirect_stdout(output), self.assertRaises(BudgetExhausted):
            run_bounded(command, time.monotonic() + 1, log, self.root, display_output=True)
        self.assertIn('started', output.getvalue())
        self.assertIn('started', log.read_text(encoding='utf-8'))

    def test_failed_status_publication_keeps_the_previous_json(self):
        status = self.root / 'status.json'
        write_json(status, {'status': 'running'})
        with patch('engine.runtime.os.replace', side_effect=OSError('publication failed')):
            with self.assertRaisesRegex(OSError, 'publication failed'):
                write_json(status, {'status': 'complete'})
        self.assertEqual(json.loads(status.read_text(encoding='utf-8')), {'status': 'running'})
        self.assertFalse(list(self.root.glob('.json-*.tmp')))

    def test_pending_budget_requires_explicit_epochs_before_launch(self):
        config = {**self.config, 'epochs': None,
                  'training_policy': {'mode': 'fixed', 'save_period': 5}}
        (self.root / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        with patch('engine.cli.bounded_train') as launch, patch('sys.stderr'):
            args = ['train', '--wall-seconds', '60']
            code = training_cli(self.root, 'config.yaml', args)
            self.assertEqual(code, 1)
            launch.assert_not_called()
            code = training_cli(self.root, 'config.yaml', [*args, '--epochs', '12'])
            self.assertEqual(code, 0)
            self.assertEqual(launch.call_args.args[1]['epochs'], 12)

    def test_policy_passes_the_frozen_effective_budget_to_child(self):
        config = {**self.config, 'epochs': 12,
                  'training_policy': {'mode': 'fixed', 'save_period': 5}}
        original = self.root / 'config.yaml'
        original.write_text(yaml.safe_dump({**config, 'epochs': None}), encoding='utf-8')
        with patch('engine.runner.run_bounded') as launch:
            bounded_train(self.root, config, 'config.yaml', 60, 'fixture')
        arguments = launch.call_args.args[0]
        snapshot = self.root / arguments[arguments.index('--config') + 1]
        self.assertEqual(yaml.safe_load(snapshot.read_text(encoding='utf-8')), config)
        self.assertIsNone(yaml.safe_load(original.read_text(encoding='utf-8'))['epochs'])
        self.assertEqual(arguments[arguments.index('--controller-wall-seconds') + 1], '60.0')

    def test_started_child_persists_wall_budget_before_staging_failure(self):
        (self.root / 'model.pt').touch()
        with patch('engine.runner.run_context', return_value=contextlib.nullcontext()), \
             patch('engine.runner.stage_dataset', side_effect=ValueError('fixture failure')):
            with self.assertRaisesRegex(ValueError, 'fixture failure'):
                train_run(self.root, self.config, 'fixture', controller_wall_seconds=42.0)
        budget = json.loads((self.root / 'runs/person/fixture/metadata/wall_budget.json').read_text())
        self.assertEqual(budget, {'wall_seconds': 42.0, 'attempt': 'train'})

    def test_invalid_child_budget_does_not_create_a_run(self):
        with self.assertRaisesRegex(ValueError, 'controller_wall_seconds'):
            train_run(self.root, self.config, 'fixture', controller_wall_seconds=float('nan'))
        self.assertFalse((self.root / 'runs/person/fixture').exists())

    def test_resume_updates_budget_only_after_resume_configuration_succeeds(self):
        run = self.root / 'runs/person/fixture'
        weights = run / 'training/weights/last.pt'
        weights.parent.mkdir(parents=True)
        weights.touch()
        (run / 'metadata').mkdir()
        budget = run / 'metadata/wall_budget.json'
        write_json(budget, {'wall_seconds': 42, 'attempt': 'train'})
        with patch('engine.runner.resume_configuration', side_effect=ValueError('invalid resume')):
            with self.assertRaisesRegex(ValueError, 'invalid resume'):
                train_run(self.root, self.config, resume=run, attempt='resume_a', controller_wall_seconds=55)
        self.assertEqual(json.loads(budget.read_text())['wall_seconds'], 42)
        with patch('engine.runner.resume_configuration', return_value=(self.config, {'dataset_signature': 'fixture'})), \
             patch('engine.runner.run_context', return_value=contextlib.nullcontext()), \
             patch('engine.runner._run_signature', side_effect=ValueError('fixture failure')):
            with self.assertRaisesRegex(ValueError, 'fixture failure'):
                train_run(self.root, self.config, resume=run, attempt='resume_a', controller_wall_seconds=55)
        self.assertEqual(json.loads(budget.read_text()), {'wall_seconds': 55, 'attempt': 'resume_a'})

    def test_resume_uses_recorded_config_and_original_run_after_timeout(self):
        run = self.root / 'runs/person/fixture'
        weights = run / 'training/weights'
        weights.mkdir(parents=True)
        (weights / 'last.pt').touch()
        config = {**self.config, 'training_policy': {'mode': 'fixed', 'save_period': 2}}
        write_json(run / 'config/effective.json', config)
        write_json(run / 'metadata/train_status.json', {'status': 'interrupted'})
        with patch('engine.runner.run_bounded', side_effect=BudgetExhausted('resume deadline')) as launch:
            with patch('sys.stderr'):
                result = training_cli(self.root, 'absent-default.yaml', [
                    'train', '--resume', 'runs/person/fixture', '--wall-seconds', '5',
                ])
        self.assertEqual(result, 124)
        args = launch.call_args.args[0]
        self.assertEqual(args[args.index('--resume') + 1], 'runs/person/fixture/training/weights/last.pt')
        snapshot = self.root / args[args.index('--config') + 1]
        self.assertEqual(yaml.safe_load(snapshot.read_text(encoding='utf-8')), config)
        attempt = args[args.index('--attempt') + 1]
        previous = json.loads((run / 'metadata/train_status.json').read_text())
        resumed = json.loads((run / f'metadata/{attempt}_status.json').read_text())
        self.assertEqual(previous['status'], 'interrupted')
        self.assertEqual(resumed['status'], 'budget_exhausted')

    def test_resume_restores_metrics_and_materializes_same_epoch_weight_copies(self):
        run = self.root / 'runs/person/fixture'
        weights = run / 'training/weights'
        weights.mkdir(parents=True)
        last = weights / 'last.pt'
        last.write_bytes(b'complete checkpoint')
        (run / 'training/results.csv').write_text('epoch,loss\n1,3\n2,2\n3,1\n', encoding='utf-8')
        checkpoint = {'train_results': {'epoch': [1, 2], 'loss': [3, 2]},
                      'policy_resume': {'best_checkpoint': True, 'periodic_checkpoint': 'epoch2.pt',
                                        'stopper': {'best_epoch': 2}}}
        _restore_checkpoint_outputs(run, last, checkpoint, 2)
        self.assertEqual((run / 'training/results.csv').read_text(encoding='utf-8'), 'epoch,loss\n1,3\n2,2\n')
        self.assertEqual((weights / 'best.pt').read_bytes(), last.read_bytes())
        self.assertEqual((weights / 'epoch2.pt').read_bytes(), last.read_bytes())

    def test_a_live_attempt_locks_same_run_until_exit(self):
        run = self.root / 'runs/person/fixture'
        (run / 'metadata').mkdir(parents=True)
        with training_lock(run):
            with self.assertRaisesRegex(RuntimeError, 'Cannot acquire training lock'):
                with training_lock(run):
                    self.fail('A second trainer entered the run')
        with training_lock(run):
            pass


if __name__ == '__main__':
    unittest.main()
