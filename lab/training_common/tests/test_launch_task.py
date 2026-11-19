"""Check the YAML-driven workflow with temporary runs and a stub launcher."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import yaml

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))
from engine.runtime import _block_progress


class LaunchTaskTests(unittest.TestCase):
    def setUp(self):
        scratch = COMMON.parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix='launch task ', dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        scripts = self.root / 'scripts'
        scripts.mkdir()
        for name in ('launch_task.py', 'monitor_training.py'):
            shutil.copy2(COMMON / name, scripts / name)
        shutil.copytree(COMMON / 'engine', scripts / 'common/engine', ignore=shutil.ignore_patterns('__pycache__'))
        (scripts / 'train.py').touch()
        (scripts / 'launch.py').write_text(
            'import json, os\n'
            'def main(argv, *, entry, model_root):\n'
            '    print(json.dumps([argv, str(entry), str(model_root), os.environ["AEC_TRAIN_PROGRESS"]]))\n'
            '    return 0\n', encoding='utf-8')
        self.model_root = self.root / 'runs/person'
        self.config = {'schema_version': 3, 'task_id': 'person', 'model_key': 'person', 'version': 'v4',
                       'task': 'detect', 'data': 'data.yaml', 'model': 'initial.pt', 'output_root': 'runs/person',
                       'epochs': 12, 'imgsz': 320, 'batch': 2, 'training_policy': {'mode': 'fixed', 'save_period': 5}}
        (self.root / 'config').mkdir()
        (self.root / 'config/train.yaml').write_text(yaml.safe_dump(self.config), encoding='utf-8')
        (self.root / 'data.yaml').touch()
        self.settings_file = self.root / 'config/launcher.yaml'
        self.write_settings()

    def write_settings(self, **settings):
        self.settings_file.write_text(yaml.safe_dump(settings), encoding='utf-8')

    def invoke(self, *args):
        return subprocess.run([sys.executable, '-B', '-X', 'utf8', str(self.root / 'scripts/launch_task.py'), *args],
                              cwd=COMMON, stdin=subprocess.DEVNULL, capture_output=True, encoding='utf-8', timeout=20)

    def make_run(self, name, *, complete=False, model='person', timestamp=1):
        run = self.model_root / name
        checkpoint = run / 'training/weights/last.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        os.utime(checkpoint, (timestamp, timestamp))
        (run / 'config').mkdir()
        (run / 'config/effective.json').write_text(json.dumps({**self.config, 'epochs': 20, 'model_key': model}),
                                                   encoding='utf-8')
        if complete:
            (run / 'metadata').mkdir()
            (run / 'metadata/train_status.json').write_text('{"status":"complete"}', encoding='utf-8')
        return run

    def test_train_reads_budget_and_browser_defaults_from_yaml(self):
        self.write_settings(wall_seconds=90)
        result = self.invoke('train')
        self.assertEqual(result.returncode, 0, result.stderr)
        argv, entry, output, style = json.loads(result.stdout[result.stdout.index('['):])
        self.assertEqual(Path(entry), self.root / 'scripts/train.py')
        self.assertEqual(Path(output), self.model_root)
        self.assertEqual(style, 'blocks')
        self.assertIn('--open-browser', argv)
        self.assertEqual(argv[argv.index('--port') + 1], '0')
        self.assertNotIn('--epochs', argv)
        self.assertEqual(argv[argv.index('--wall-seconds') + 1], '90')
        self.assertIn('--run-name', argv)

    def test_resume_selects_latest_unfinished_model_and_reuses_wall_budget(self):
        older = self.make_run('old', timestamp=1)
        selected = self.make_run('retry this run', timestamp=2)
        self.make_run('newer but completed', complete=True, timestamp=3)
        self.make_run('different model', model='sign', timestamp=4)
        receipt = self.root / 'runs/_launcher/previous/status.json'
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps({'run': selected.relative_to(self.root).as_posix(), 'train_pid': 42,
                                       'wall_seconds': 60}), encoding='utf-8')
        result = self.invoke('resume')
        self.assertEqual(result.returncode, 0, result.stderr)
        argv, _, _, _ = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(Path(argv[argv.index('--resume') + 1]), selected)
        self.assertEqual(argv[argv.index('--wall-seconds') + 1], '60.0')
        self.assertNotIn('--epochs', argv)
        self.assertNotIn('--config', argv)
        self.write_settings(wall_seconds=35, tensorboard=False, open_browser=False, keep_tensorboard_open=False)
        result = self.invoke('resume', '--run', older.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv, _, _, _ = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(Path(argv[argv.index('--resume') + 1]), older)
        self.assertIn('--no-tensorboard', argv)
        self.assertNotIn('--open-browser', argv)
        self.assertIn('--no-wait', argv)
        self.assertEqual(argv[argv.index('--wall-seconds') + 1], '35')

    def test_missing_budget_points_to_yaml_without_requiring_stdin(self):
        result = self.invoke('train')
        self.assertEqual(result.returncode, 1)
        self.assertIn('Set wall_seconds in the launcher config', result.stderr)
        self.assertIn('launcher.yaml', result.stderr)

    def test_run_local_budget_survives_copy_and_explicit_setting_overrides(self):
        original = self.make_run('original')
        budget = original / 'metadata/wall_budget.json'
        budget.parent.mkdir()
        budget.write_text(json.dumps({'wall_seconds': 72, 'attempt': 'train'}), encoding='utf-8')
        moved = self.model_root / 'moved'
        shutil.copytree(original, moved)
        shutil.rmtree(original)
        result = self.invoke('resume', '--run', moved.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv, _, _, _ = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(argv[argv.index('--wall-seconds') + 1], '72.0')
        self.write_settings(wall_seconds=35)
        result = self.invoke('resume', '--run', moved.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv, _, _, _ = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(argv[argv.index('--wall-seconds') + 1], '35')

    def test_corrupt_local_budget_falls_back_to_old_launcher_receipt(self):
        run = self.make_run('fixture')
        budget = run / 'metadata/wall_budget.json'
        budget.parent.mkdir()
        budget.write_text('{', encoding='utf-8')
        receipt = self.root / 'runs/_launcher/previous/status.json'
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps({'run': run.relative_to(self.root).as_posix(), 'train_pid': 42,
                                       'wall_seconds': 60}), encoding='utf-8')
        result = self.invoke('resume', '--run', run.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv, _, _, _ = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(argv[argv.index('--wall-seconds') + 1], '60.0')

    def test_latest_valid_receipt_wins_across_old_and_run_local_records(self):
        run = self.make_run('fixture')
        budget = run / 'metadata/wall_budget.json'
        budget.parent.mkdir()
        budget.write_text(json.dumps({'wall_seconds': 72, 'attempt': 'resume_a'}), encoding='utf-8')
        receipt = self.root / 'runs/_launcher/previous/status.json'
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps({'run': run.relative_to(self.root).as_posix(), 'train_pid': 42,
                                       'wall_seconds': 60}), encoding='utf-8')
        os.utime(receipt, (100, 100))
        os.utime(budget, (200, 200))
        result = self.invoke('resume', '--run', run.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv, _, _, _ = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(argv[argv.index('--wall-seconds') + 1], '72.0')
        os.utime(receipt, (300, 300))
        result = self.invoke('resume', '--run', run.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv, _, _, _ = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(argv[argv.index('--wall-seconds') + 1], '60.0')

    @unittest.skipUnless(os.name == 'nt', 'PowerShell package entry is Windows-specific')
    def test_package_powershell_uses_portable_environment(self):
        portable = self.root / 'portable_env'
        portable.mkdir()
        (portable / 'run.ps1').write_text(
            'param($Script,[switch]$Offline,[string[]]$ScriptArguments)\n'
            '& $env:AEC_FIXTURE_PY -B $Script @ScriptArguments\nexit $LASTEXITCODE\n', encoding='utf-8')
        shutil.copy2(COMMON.parent / 'train.ps1', self.root / 'train.ps1')
        shutil.copy2(COMMON / 'run_training_tool.ps1', self.root / 'scripts/run_training_tool.ps1')
        result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                 str(self.root / 'train.ps1'), '-Resume', '-Plan'], cwd=COMMON,
                                capture_output=True, encoding='utf-8', timeout=30,
                                env={**os.environ, 'AEC_FIXTURE_PY': sys.executable})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('resume:', result.stdout)
        self.assertIn('config/train.yaml', result.stdout)
        self.assertFalse((self.root / 'runs').exists())

    def test_blocks_missing_dataset_and_missing_checkpoint(self):
        (self.root / 'data.yaml').unlink()
        for action, expected in [('train', 'Dataset is missing'), ('resume', 'No unfinished V4 run')]:
            result = self.invoke(action)
            self.assertEqual(result.returncode, 1)
            self.assertIn(expected, result.stderr)

    def test_block_progress(self):
        self.assertEqual(_block_progress(SimpleNamespace(total=10, n=4), width=10), '████░░░░░░')
        self.assertEqual(_block_progress(SimpleNamespace(total=0, closed=True), width=3), '███')


if __name__ == '__main__':
    unittest.main()
