"""Exercise the actual PowerShell bundle selection with isolated fixtures."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == 'win32' and sys.version_info[:2] in ((3, 12), (3, 14))
                     and shutil.which('powershell.exe'), 'Windows and supported Python required')
class BundleSelection(unittest.TestCase):
    def setUp(self):
        parent = ROOT / 'runs' / 'tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copy2(ROOT / 'run.ps1', self.root / 'run.ps1')

    def bundle(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'manifest.json').write_text(json.dumps({
            'python_implementation': sys.implementation.name,
            'python_version': list(sys.version_info[:2]),
            'bits': struct.calcsize('P') * 8,
            'platform': sysconfig.get_platform(),
        }), encoding='utf-8')
        (path / 'site-packages').mkdir(exist_ok=True)
        (path / 'bootstrap.py').write_text("print('SELECTED_BUNDLE=' + __file__)\n", encoding='utf-8')

    def launch(self):
        return subprocess.run([shutil.which('powershell.exe'), '-NoProfile', '-ExecutionPolicy', 'Bypass',
                               '-File', str(self.root / 'run.ps1'), '-Check', '-Python', sys.executable],
                              capture_output=True, text=True, encoding='utf-8', errors='replace')

    def test_invalid_optional_bundle_does_not_block_root(self):
        self.bundle(self.root)
        optional = self.root / 'cp314'
        self.bundle(optional)
        for defect in ('manifest', 'bootstrap', 'site-packages'):
            with self.subTest(defect=defect):
                self.bundle(optional)
                if defect == 'manifest':
                    (optional / 'manifest.json').write_text('{ invalid', encoding='utf-8')
                elif defect == 'bootstrap':
                    (optional / 'bootstrap.py').unlink()
                else:
                    (optional / 'site-packages').rmdir()
                result = self.launch()
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('Skipping unavailable portable bundle', result.stdout)
                self.assertIn(str(self.root / 'bootstrap.py'), result.stdout)

    def test_invalid_root_can_use_valid_optional_bundle(self):
        self.bundle(self.root / 'cp314')
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('manifest.json missing', result.stdout)
        self.assertIn(str(self.root / 'cp314' / 'bootstrap.py'), result.stdout)

    def test_no_valid_matching_bundle_reports_failure(self):
        self.bundle(self.root)
        (self.root / 'bootstrap.py').unlink()
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('bootstrap.py missing', result.stdout + result.stderr)
        self.assertIn('available dependency bundle was not found', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
