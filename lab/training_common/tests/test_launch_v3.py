"""Exercise the copied launcher with an isolated fake environment and no training."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SHELL = shutil.which('pwsh.exe') or shutil.which('powershell.exe')

FAKE_ENV = r'''param([string]$Python,[string]$Script,[switch]$Offline,[string[]]$ScriptArguments)
$env:AEC_PORTABLE_BOOTSTRAP = Join-Path $PSScriptRoot 'bootstrap.py'
& $Python -B $Script @ScriptArguments
exit $LASTEXITCODE
'''
FAKE_BOOTSTRAP = r'''import runpy, socket, sys
args = sys.argv[1:]
if '--module' in args:
    port = int(args[args.index('--port')+1])
    with socket.socket() as server:
        server.bind(('127.0.0.1', port)); server.listen()
        while True:
            connection, _ = server.accept(); connection.close()
else:
    script = args[args.index('--script')+1]
    rest = args[args.index('--')+1:] if '--' in args else []
    sys.argv = [script] + rest
    runpy.run_path(script, run_name='__main__')
'''
FAKE_ENTRY = r'''import json, os, pathlib, sys
root = pathlib.Path(__file__).resolve().parents[1]
phase = sys.argv[1]
(root / (phase + '.json')).write_text(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}))
raise SystemExit(int(os.environ.get('FAKE_' + phase.upper() + '_EXIT', '0')))
'''


@unittest.skipUnless(os.name == 'nt' and SHELL, 'Windows PowerShell is required')
class CopiedLauncherTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / 'tmp'
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='copied launcher ', dir=parent)
        self.addCleanup(self.temp.cleanup)
        home = Path(self.temp.name)
        self.package = home / 'task package with spaces'
        scripts = self.package / 'scripts'
        scripts.mkdir(parents=True)
        shutil.copy2(ROOT/'lab/training_common/launch_v3.py', scripts/'launch.py')
        shutil.copy2(ROOT/'lab/training_common/start_v3.ps1', scripts/'start_training.ps1')
        (scripts/'train.py').write_text(FAKE_ENTRY, encoding='utf-8')
        env = home/'portable_env'
        env.mkdir()
        (env/'run.ps1').write_text(FAKE_ENV, encoding='utf-8')
        (env/'bootstrap.py').write_text(FAKE_BOOTSTRAP, encoding='utf-8')

    def invoke(self, version, wall=None, **environment):
        (self.package/'mirror_manifest.json').write_text(json.dumps({'version': version}), encoding='utf-8')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
        command = [SHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                   str(self.package/'scripts/start_training.ps1'), '-Python', sys.executable,
                   '-Port', str(port), '-NoWait']
        if wall is not None:
            command += ['-WallSeconds', str(wall)]
        return subprocess.run(command, cwd=ROOT, env={**os.environ, **environment},
                              capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)

    def state(self):
        return json.loads(next((self.package/'runs/_launcher').glob('*/status.json')).read_text(encoding='utf-8'))

    def test_v4_requires_explicit_limit_and_records_failure(self):
        result = self.invoke('v4')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--wall-seconds', result.stderr)
        self.assertFalse((self.package/'train.json').exists())
        self.assertEqual(self.state()['status'], 'failed')

    def test_v4_limit_reaches_training_in_copied_package(self):
        result = self.invoke('v4', 17)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        train = json.loads((self.package/'train.json').read_text())
        self.assertEqual(train['argv'], ['train', '--config', 'config/train.yaml', '--wall-seconds', '17.0'])
        self.assertEqual(Path(train['cwd']).resolve(), self.package.resolve())
        self.assertEqual(self.state()['status'], 'complete')

    def test_v3_unbounded_invocation_remains_compatible(self):
        result = self.invoke('v3')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        train = json.loads((self.package/'train.json').read_text())
        self.assertNotIn('--wall-seconds', train['argv'])
        self.assertEqual(self.state()['status'], 'complete')

    def test_invalid_limit_and_timeout_are_recorded(self):
        result = self.invoke('v4', 0)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.state()['status'], 'failed')
        self.assertFalse((self.package/'train.json').exists())

    def test_budget_exhaustion_is_not_reported_as_success(self):
        result = self.invoke('v4', 1, FAKE_TRAIN_EXIT='124')
        self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
        self.assertEqual(self.state()['status'], 'budget_exhausted')

    def test_check_failure_does_not_start_training(self):
        result = self.invoke('v4', 10, FAKE_CHECK_EXIT='13')
        self.assertEqual(result.returncode, 13, result.stdout + result.stderr)
        self.assertEqual(self.state()['status'], 'check_failed')
        self.assertFalse((self.package/'train.json').exists())


if __name__ == '__main__':
    unittest.main()
