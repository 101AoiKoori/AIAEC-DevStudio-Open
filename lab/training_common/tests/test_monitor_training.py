"""Exercise the read-only monitor with copied-format status and log fixtures."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor_training as monitor


class MonitorTrainingTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[3] / 'tmp'
        base.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = Path('runs/person/v4_example')

    def write(self, location, text, modified=100):
        path = self.root / location
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        os.utime(path, (modified, modified))
        return path

    def receipt(self, location, status, modified=100, **extra):
        return self.write(location, json.dumps({'status': status, **extra}), modified)

    def controller(self, status, modified=100, **extra):
        return self.receipt('runs/person/_controllers/controller_a/status.json', status, modified,
                            run=self.run.as_posix(), **extra)

    def test_latest_selects_actual_training_and_ignores_verification_and_controllers(self):
        self.receipt(self.run / 'metadata/status.json', 'complete', phase='train')
        self.receipt('runs/person/older/metadata/status.json', 'failed', 90)
        self.receipt('runs/person/_verification/check/metadata/status.json', 'complete', 150)
        self.receipt('runs/person/_controllers/other/metadata/status.json', 'complete', 200)
        observed = monitor.snapshot(self.root)
        self.assertEqual(observed['run'], self.root / self.run)
        self.assertEqual(observed['status'], 'complete')
        self.assertEqual(monitor.snapshot(self.root, 'runs/person/older')['status'], 'failed')

    def test_new_controller_defers_old_completion_and_does_not_scan_test_phase(self):
        self.receipt(self.run / 'metadata/train_status.json', 'complete', phase='train')
        self.receipt(self.run / 'metadata/status.json', 'complete', 300, phase='evaluate/final_test')
        self.controller('running', 200, attempt='resume_a', wall_seconds=1000)
        observed = monitor.snapshot(self.root, now=201)
        self.assertEqual((observed['task'], observed['status'], observed['phase']),
                         ('complete', 'running', 'resume_a'))
        self.receipt(self.run / 'metadata/resume_a_status.json', 'complete', 220, phase='resume_a')
        self.assertEqual(monitor.snapshot(self.root, now=222)['status'], 'running')
        self.controller('interrupted', 223, attempt='resume_a')
        self.assertEqual(monitor.snapshot(self.root, now=224)['status'], 'interrupted')

    def test_stale_controller_does_not_infer_success_from_old_task(self):
        self.receipt(self.run / 'metadata/train_status.json', 'complete', phase='train')
        self.controller('running', 200, attempt='resume_a', wall_seconds=20)
        observed = monitor.snapshot(self.root, now=240)
        self.assertEqual(observed['status'], 'stale_running')
        self.assertIn('实际进程状态未确认', monitor._render(observed))

    def test_launcher_before_run_is_visible_without_choosing_previous_complete(self):
        self.receipt(self.run / 'metadata/train_status.json', 'complete', phase='train')
        self.receipt('runs/_launcher/current/status.json', 'starting', 120)
        observed = monitor.snapshot(self.root)
        self.assertIsNone(observed['run'])
        self.assertEqual(observed['status'], 'starting')
        self.controller('running', 130, wall_seconds=30)
        observed = monitor.snapshot(self.root, self.run, now=131)
        self.assertEqual(observed['launcher'], 'starting')
        self.assertFalse(observed['launcher_matched'])
        self.assertEqual(observed['status'], 'running')

    def test_confirmed_exited_launcher_stops_follow_for_matched_and_checking_states(self):
        self.receipt('runs/_launcher/current/status.json', 'training', 90, run=self.run.as_posix(),
                     launcher_pid=101, train_pid=102)
        self.controller('running', 100, wall_seconds=10000)
        with patch.object(monitor, '_pid_running', side_effect={101: False, 102: None}.get):
            self.assertEqual(monitor.snapshot(self.root, self.run, now=101)['status'], 'running')
        with patch.object(monitor, '_pid_running', return_value=False):
            self.assertEqual(monitor.snapshot(self.root, self.run, now=101)['status'], 'launcher_exited')
        self.receipt('runs/_launcher/current/status.json', 'checking', 110, launcher_pid=105,
                     resume='invalid\0path')
        with patch.object(monitor, '_pid_running', return_value=False) as pid_status, redirect_stdout(io.StringIO()):
            self.assertEqual(monitor.main(['--root', str(self.root), '--follow']), 1)
            pid_status.assert_called_once_with(105)

    def test_windows_process_query_distinguishes_exited_alive_and_access_denied(self):
        kernel = MagicMock()
        with patch.object(monitor.os, 'name', 'nt'), patch.object(monitor, '_windows_process_api', return_value=kernel), \
                patch.object(monitor.ctypes, 'get_last_error', create=True) as last_error:
            kernel.OpenProcess.return_value = None
            last_error.return_value = 87
            self.assertIs(monitor._pid_running(505), False)
            last_error.return_value = 5
            self.assertIsNone(monitor._pid_running(505))
            kernel.OpenProcess.return_value = 33
            kernel.WaitForSingleObject.return_value = 258
            self.assertIs(monitor._pid_running(505), True)
            self.assertIsNone(monitor._pid_running(None))
            kernel.OpenProcess.assert_called_with(0x00100000, False, 505)
            kernel.WaitForSingleObject.assert_called_once_with(33, 0)
            kernel.CloseHandle.assert_called_once_with(33)

    def test_reads_losses_budget_and_latest_valid_resume_step(self):
        self.receipt(self.run / 'metadata/status.json', 'running', phase='resume_a')
        self.write(self.run / 'config/effective.json', '{"epochs": 8}')
        self.write(self.run / 'training/results.csv',
                   ' epoch,train/box_loss,lr/pg0\n1,0.25,0.001\n2,0.5,0.002\n3,0.9,')
        self.write(self.run / 'metadata/optimizer_updates_resume_a.jsonl',
                   '{"step":23,"epoch":3,"learning_rates":[0.0001]}\n{"step":24')
        observed = monitor.snapshot(self.root, self.run)
        self.assertEqual(observed['progress'], {'epoch': '2', 'max_epochs': 8, 'step_epoch': 3, 'steps': 23,
                                               'steps_confirmed': False, 'loss': {'train/box_loss': 0.5},
                                               'lr': [0.0001]})
        self.assertIn('进程未核实', monitor._render(observed))

    def test_legacy_receipt_lacks_progress_fields(self):
        self.receipt(self.run / 'metadata/status.json', 'interrupted')
        self.write(self.run / 'metadata/optimization_summary.json', '{"actual_optimizer_steps": 0}')
        observed = monitor.snapshot(self.root)
        self.assertEqual(observed['status'], 'interrupted')
        self.assertEqual(observed['progress']['steps'], 0)

    def test_tensorboard_uses_only_matching_local_healthy_log_root(self):
        script = 'runs/_launcher/launch/status.json'
        previous = self.receipt(script, 'training',
                                tensorboard_url='http://127.0.0.1:6006/', run=self.run.as_posix())
        response = MagicMock(status=200)
        response.read.return_value = json.dumps({'data_location': str(self.root / 'runs')}).encode()
        with patch.object(monitor.http.client, 'HTTPConnection') as connection:
            connection.return_value.getresponse.return_value = response
            self.assertEqual(monitor.snapshot(self.root, self.run)['tensorboard'], 'http://127.0.0.1:6006/')
            connection.assert_called_once_with('127.0.0.1', 6006, timeout=0.4)
            connection.return_value.request.assert_called_once_with('GET', '/data/environment')
            response.read.return_value = json.dumps({'data_location': str(self.root / 'other_runs')}).encode()
            self.assertIsNone(monitor.snapshot(self.root, self.run)['tensorboard'])
            previous.write_text(json.dumps({'status': 'training', 'tensorboard_url': 'http://example.com:6006/'}))
            connection.reset_mock()
            self.assertIsNone(monitor.snapshot(self.root, self.run)['tensorboard'])
            connection.assert_not_called()

    def test_follow_waits_for_terminal_and_keeps_the_run(self):
        self.controller('running', wall_seconds=10000)
        output = io.StringIO()
        with redirect_stdout(output), patch.object(monitor.time, 'time', return_value=101), \
                patch.object(monitor.time, 'sleep', side_effect=lambda _: self.controller('budget_exhausted', 102)):
            code = monitor.main(['--root', str(self.root), '--follow', '--interval', '0.01'])
        self.assertEqual(code, 124)
        self.assertIn('running', output.getvalue())
        self.assertIn('budget_exhausted', output.getvalue())

    def test_ctrl_c_stops_only_monitoring(self):
        with patch.object(monitor.time, 'sleep', side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()):
            self.assertEqual(monitor.main(['--root', str(self.root), '--follow']), 130)
        self.assertFalse((self.root / 'runs').exists())


if __name__ == '__main__':
    unittest.main()
