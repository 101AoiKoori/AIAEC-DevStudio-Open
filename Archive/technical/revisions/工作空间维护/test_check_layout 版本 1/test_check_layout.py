import contextlib
import importlib.util
import io
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

    def test_missing_package_manifest_and_path_suggestion(self):
        self.write('ws/src/aec_new/src/aec_new/node.py')
        subprocess.run(['git', '-C', str(self.root), 'add', 'ws/src/aec_new/src/aec_new/node.py'], check=True)
        code, output = self.run_check('--staged')
        self.assertEqual(code, 1)
        self.assertIn('package.xml', output)
        code, output = self.run_check('--path', 'lab/features/meter/training/scripts/new.py')
        self.assertEqual(code, 0)
        self.assertIn('仍需按用途判断', output)


if __name__ == '__main__':
    unittest.main()
