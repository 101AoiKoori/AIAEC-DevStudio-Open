"""The shared launcher can be inspected and updated without rebuilding dependencies."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from unittest.mock import patch

from lab.training_common.portable_env import sync_launcher


class SyncLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'portable_env'
        self.bundle.mkdir()
        self.source = self.root / 'source.ps1'
        self.source.write_bytes(b'new launcher')
        (self.bundle / 'run.ps1').write_bytes(b'old launcher')
        (self.bundle / 'manifest.json').write_bytes(b'manifest')
        (self.bundle / 'bootstrap.py').write_bytes(b'bootstrap')
        (self.bundle / 'site-packages').mkdir()
        (self.bundle / 'site-packages' / 'dependency.bin').write_bytes(b'dependency')

    def test_status_is_read_only_and_update_preserves_environment(self):
        status = sync_launcher.launcher_status(self.bundle, source=self.source)
        self.assertEqual(status['status'], 'outdated')
        self.assertEqual((self.bundle / 'run.ps1').read_bytes(), b'old launcher')
        result = sync_launcher.update_launcher(self.bundle, source=self.source)
        self.assertEqual((result['status'], result['action']), ('current', 'updated'))
        self.assertEqual(result['previous_launcher_sha256'], status['launcher_sha256'])
        self.assertEqual((self.bundle / 'run.ps1').read_bytes(), b'new launcher')
        self.assertEqual((self.bundle / 'manifest.json').read_bytes(), b'manifest')
        self.assertEqual((self.bundle / 'bootstrap.py').read_bytes(), b'bootstrap')
        self.assertEqual((self.bundle / 'site-packages' / 'dependency.bin').read_bytes(), b'dependency')
        self.assertEqual(sync_launcher.update_launcher(self.bundle, source=self.source)['action'], 'unchanged')

    def test_changed_source_aborts_and_keeps_launcher(self):
        real_status = sync_launcher.launcher_status
        calls = 0

        def changed(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.source.write_bytes(b'changed source')
            return real_status(*args, **kwargs)

        with patch.object(sync_launcher, 'launcher_status', side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, 'changed during update'):
                sync_launcher.update_launcher(self.bundle, source=self.source)
        self.assertEqual((self.bundle / 'run.ps1').read_bytes(), b'old launcher')
        self.assertFalse(list(self.bundle.glob('.run-*.tmp')))

    def test_cli_requires_explicit_update(self):
        command = [sys.executable, str(sync_launcher.SOURCE.with_name('sync_launcher.py')),
                   '--bundle', str(self.bundle)]
        report = json.loads(subprocess.check_output(command, text=True, encoding='utf-8'))
        self.assertEqual(report['status'], 'outdated')
        self.assertEqual((self.bundle / 'run.ps1').read_bytes(), b'old launcher')
        report = json.loads(subprocess.check_output([*command, '--update'], text=True, encoding='utf-8'))
        self.assertEqual(report['action'], 'updated')
        self.assertEqual((self.bundle / 'run.ps1').read_bytes(), sync_launcher.SOURCE.read_bytes())
        self.assertEqual((self.bundle / 'manifest.json').read_bytes(), b'manifest')

    @unittest.skipUnless(sys.platform == 'win32' and sys.version_info[:2] in ((3, 12), (3, 14))
                         and shutil.which('powershell.exe'), 'Windows and supported Python required')
    def test_updated_launcher_runs_with_a_broken_optional_bundle(self):
        manifest = {'python_implementation': sys.implementation.name,
                    'python_version': list(sys.version_info[:2]),
                    'bits': struct.calcsize('P') * 8, 'platform': sysconfig.get_platform()}
        (self.bundle / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        (self.bundle / 'bootstrap.py').write_text("print('ROOT_SELECTED')\n", encoding='utf-8')
        optional = self.bundle / 'cp314'
        optional.mkdir()
        (optional / 'manifest.json').write_text('{broken', encoding='utf-8')
        (optional / 'bootstrap.py').write_text("print('OPTIONAL_SELECTED')\n", encoding='utf-8')
        (optional / 'site-packages').mkdir()
        sync_launcher.update_launcher(self.bundle)
        result = subprocess.run([shutil.which('powershell.exe'), '-NoProfile', '-ExecutionPolicy', 'Bypass',
                                 '-File', str(self.bundle / 'run.ps1'), '-Check', '-Python', sys.executable],
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Skipping unavailable portable bundle', result.stdout)
        self.assertIn('ROOT_SELECTED', result.stdout)


if __name__ == '__main__':
    unittest.main()
