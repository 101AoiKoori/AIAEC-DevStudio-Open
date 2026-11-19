import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'check_layout.py'
spec = importlib.util.spec_from_file_location('check_layout', SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class LayoutCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)

    def write(self, name, content='x'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')

    def run_check(self, *args):
        output = io.StringIO()
        with patch.object(checker, 'ROOT', self.root), patch.object(sys, 'argv', ['check_layout.py', *args]), contextlib.redirect_stdout(output):
            code = checker.main()
        return code, output.getvalue()

    def test_audit_and_staged_scope(self):
        self.write('.gitignore', 'datasets/raw/\n')
        self.write('datasets/raw/person/image.jpg')
        self.write('unknown/script.py')
        self.write('ws/src/aec_vision/package.xml')
        subprocess.run(['git', '-C', str(self.root), 'add', '.gitignore', 'ws/src/aec_vision/package.xml'], check=True)
        code, output = self.run_check()
        self.assertEqual(code, 0)
        self.assertIn('unknown/script.py', output)
        self.assertNotIn('image.jpg', output)
        code, output = self.run_check('--staged')
        self.assertEqual(code, 0)
        self.assertNotIn('unknown/script.py', output)
        subprocess.run(['git', '-C', str(self.root), 'add', '-f', 'datasets/raw/person/image.jpg'], check=True)
        code, output = self.run_check('--staged')
        self.assertEqual(code, 1)
        self.assertIn('datasets/raw/person/image.jpg', output)

    def test_dataset_placeholders_allow_empty_directories_and_still_reject_data(self):
        placeholders = ['datasets/samples/person/.gitkeep'] + [
            f'datasets/exports/person/{split}/.gitkeep' for split in ('train', 'val', 'test')
        ]
        for path in placeholders:
            self.write(path, '')
        subprocess.run(['git', '-C', str(self.root), 'add', *placeholders], check=True)
        code, output = self.run_check('--staged')
        self.assertEqual(code, 0, output)
        self.write('datasets/exports/person/train/image.png')
        subprocess.run(['git', '-C', str(self.root), 'add', 'datasets/exports/person/train/image.png'], check=True)
        code, output = self.run_check('--staged')
        self.assertEqual(code, 1)
        self.assertIn('image.png', output)
        for path in ('tmp/.gitkeep', 'datasets/raw/person/.gitkeep',
                     'datasets/samples/person/images/.gitkeep'):
            self.assertEqual(checker.classify(path, set())[0], 'error')

    def test_missing_package_manifest_and_path_suggestion(self):
        self.write('ws/src/aec_new/src/aec_new/node.py')
        subprocess.run(['git', '-C', str(self.root), 'add', 'ws/src/aec_new/src/aec_new/node.py'], check=True)
        code, output = self.run_check('--staged')
        self.assertEqual(code, 1)
        self.assertIn('package.xml', output)
        code, output = self.run_check('--path', 'lab/features/meter/training/scripts/new.py')
        self.assertEqual(code, 0)
        self.assertIn('仍需按用途判断', output)

    def test_deleted_manifest_is_checked_and_outputs_are_advisory(self):
        self.write('ws/src/aec_vision/package.xml')
        self.write('ws/src/aec_vision/src/aec_vision/node.py')
        subprocess.run(['git', '-C', str(self.root), 'add', 'ws'], check=True)
        subprocess.run(['git', '-C', str(self.root), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'baseline'], check=True)
        subprocess.run(['git', '-C', str(self.root), 'rm', '-q', 'ws/src/aec_vision/package.xml'], check=True)
        code, output = self.run_check('--staged')
        self.assertEqual(code, 1)
        self.assertIn('package.xml', output)
        code, output = self.run_check('--path', 'tmp/result.json')
        self.assertEqual(code, 0)
        self.assertIn('禁止提交', output)
        code, output = self.run_check('--path', 'C:\\outside\\file.py')
        self.assertEqual(code, 1)

    def test_suggest_uses_real_task_ids(self):
        self.write('config/perception_tasks.json', json.dumps({'tasks': [{'id': 'meter'}]}))
        code, output = self.run_check('--suggest', 'training-script', '--task', 'meter')
        self.assertEqual(code, 0)
        self.assertIn('lab/features/meter/training/scripts', output)
        with contextlib.redirect_stderr(io.StringIO()):
            code, _ = self.run_check('--suggest', 'training-script', '--task', 'unknown')
        self.assertEqual(code, 2)

    def test_templates_and_package_model_code_are_not_blocked(self):
        manifest = {'ws/src/aec_vision/package.xml'}
        severity, _ = checker.classify('ws/src/aec_vision/src/aec_vision/models/network.py', manifest)
        self.assertEqual(severity, 'ok')
        severity, _ = checker.classify('config/.env.example', manifest)
        self.assertEqual(severity, 'review')
        severity, _ = checker.classify('config/.env.local', manifest)
        self.assertEqual(severity, 'error')


if __name__ == '__main__':
    unittest.main()
