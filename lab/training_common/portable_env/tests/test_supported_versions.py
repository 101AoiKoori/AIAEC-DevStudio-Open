"""Exercise version rejection with an actual installed 3.13 interpreter."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def python_313():
    launcher = shutil.which('py')
    if not launcher:
        return None
    result = subprocess.run([launcher, '-0p'], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        match = re.search(r'([A-Za-z]:\\.*python(?:\d+(?:\.\d+)?)?\.exe)\s*$', line)
        if match:
            executable = match.group(1)
            probe = subprocess.run([executable, '-I', '-c', 'import sys; print(sys.version_info[:2])'],
                                   capture_output=True, text=True)
            if probe.returncode == 0 and probe.stdout.strip() == '(3, 13)':
                return executable
    return None


class SupportedVersionTests(unittest.TestCase):
    def test_313_launcher_rejected_even_if_stale_bundle_exists(self):
        executable = python_313()
        powershell = shutil.which('powershell.exe')
        if not executable or not powershell:
            self.skipTest('Windows PowerShell and CPython 3.13 are required')
        parent = ROOT / 'runs' / 'tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            bundle = Path(directory)
            shutil.copy2(ROOT / 'run.ps1', bundle / 'run.ps1')
            manifest = {'python_implementation': 'cpython', 'python_version': [3, 12],
                        'bits': 64, 'platform': 'win-amd64'}
            (bundle / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            stale = bundle / 'cp313'
            stale.mkdir()
            manifest['python_version'] = [3, 13]
            (stale / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            result = subprocess.run([powershell, '-NoProfile', '-ExecutionPolicy', 'Bypass',
                                     '-File', str(bundle / 'run.ps1'), '-Check', '-Python', executable],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Python 3.13 is unsupported', result.stdout + result.stderr)

    def test_313_builder_rejected_before_output(self):
        executable = python_313()
        if not executable:
            self.skipTest('CPython 3.13 is required')
        result = subprocess.run([executable, str(ROOT / 'build.py')], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Python 3.13 is unsupported', result.stderr)


if __name__ == '__main__':
    unittest.main()
