"""Exercise the copied launcher with an isolated fake environment and no training."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[3]
SHELL = shutil.which('pwsh.exe') or shutil.which('powershell.exe')

FAKE_ENV = r'''param([string]$Python,[string]$Script,[switch]$Offline,[string[]]$ScriptArguments)
$env:AEC_PORTABLE_BOOTSTRAP = Join-Path $PSScriptRoot 'bootstrap.py'
$driver = Join-Path $PSScriptRoot 'fixture_launcher.py'
& $Python -B $driver $Script @ScriptArguments
exit $LASTEXITCODE
'''
FAKE_DRIVER = r'''import importlib.util, os, pathlib, sys, time
script = pathlib.Path(sys.argv.pop(1))
spec = importlib.util.spec_from_file_location('copied_launcher', script)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

def wait_for_fake_tensorboard(process, port, log_path):
    # Codex's Windows sandbox can reject even local TCP connections.
    ready_file = script.parents[1] / 'tensorboard.ready'
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        if ready_file.is_file():
            return
        time.sleep(.02)
    raise RuntimeError('Fake TensorBoard did not start')

launcher._wait_for_tensorboard = wait_for_fake_tensorboard
if os.environ.get('FAKE_OPEN_BROWSER'):
    def fake_open(url):
        assert (script.parents[1] / 'tensorboard.ready').is_file()
        (script.parents[1] / 'browser_url.txt').write_text(url, encoding='utf-8')
    os.startfile = fake_open
    sys.argv.append('--open-browser')
if os.environ.get('FAKE_JOB_FAILURE'):
    def fail_job():
        raise PermissionError('fake job restriction')
    launcher._join_windows_job = fail_job
raise SystemExit(launcher.main())
'''
FAKE_BOOTSTRAP = r'''import json, os, pathlib, runpy, socket, subprocess, sys
args = sys.argv[1:]
if '--module' in args:
    port = int(args[args.index('--port')+1])
    with socket.socket() as server:
        server.bind(('127.0.0.1', port)); server.listen()
        if os.environ.get('FAKE_SPAWN_TREE'):
            child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(60)'])
            pathlib.Path('tensorboard_processes.json').write_text(json.dumps([os.getpid(), child.pid]))
        pathlib.Path('tensorboard.ready').touch()
        while True:
            connection, _ = server.accept(); connection.close()
else:
    script = args[args.index('--script')+1]
    rest = args[args.index('--')+1:] if '--' in args else []
    sys.argv = [script] + rest
    runpy.run_path(script, run_name='__main__')
'''
FAKE_ENTRY = r'''import json, os, pathlib, subprocess, sys, time
root = pathlib.Path(__file__).resolve().parents[1]
phase = sys.argv[1]
(root / (phase + '.json')).write_text(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}))
if phase == 'train' and os.environ.get('FAKE_SPAWN_TREE'):
    child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(60)'])
    (root / 'training_processes.json').write_text(json.dumps([os.getpid(), child.pid]))
    time.sleep(60)
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
        (env/'fixture_launcher.py').write_text(FAKE_DRIVER, encoding='utf-8')

    def invoke(self, version, wall=None, *, resume=None, epochs=None, no_tensorboard=False, port=None,
               **environment):
        (self.package/'mirror_manifest.json').write_text(json.dumps({'version': version}), encoding='utf-8')
        if port is None:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
        command = [SHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                   str(self.package/'scripts/start_training.ps1'), '-Python', sys.executable,
                   '-Port', str(port), '-NoWait']
        if wall is not None:
            command += ['-WallSeconds', str(wall)]
        if resume:
            command += ['-Resume', resume]
        if epochs is not None:
            command += ['-Epochs', str(epochs)]
        if no_tensorboard:
            command += ['-NoTensorBoard']
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

    def test_automatic_port_opens_browser_after_tensorboard_is_ready(self):
        bootstrap = self.package.parent / 'portable_env/bootstrap.py'
        result = subprocess.run([sys.executable, '-B', str(bootstrap.with_name('fixture_launcher.py')),
                                 str(self.package / 'scripts/launch.py'), '--root', str(self.package), '--port', '0',
                                 '--no-wait', '--wall-seconds', '30'], cwd=self.package, capture_output=True,
                                timeout=30, env={**os.environ, 'AEC_PORTABLE_BOOTSTRAP': str(bootstrap),
                                                 'FAKE_OPEN_BROWSER': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        url = (self.package / 'browser_url.txt').read_text(encoding='utf-8')
        self.assertRegex(url, r'^http://127\.0\.0\.1:[1-9]\d*/$')
        self.assertEqual(url, self.state()['tensorboard_url'])

    def test_v4_epochs_reach_both_commands_without_tensorboard(self):
        (self.package / 'package_manifest.json').write_text('{"version": "v4"}', encoding='utf-8')
        with socket.socket() as reserved:
            reserved.bind(('127.0.0.1', 0))
            result = self.invoke('v3', 17, epochs=8, no_tensorboard=True, port=reserved.getsockname()[1])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        check = json.loads((self.package / 'check.json').read_text())
        train = json.loads((self.package / 'train.json').read_text())
        self.assertEqual(check['argv'], ['check', '--config', 'config/train.yaml', '--epochs', '8'])
        self.assertEqual(train['argv'], ['train', '--config', 'config/train.yaml', '--epochs', '8',
                                         '--wall-seconds', '17.0'])
        self.assertIsNone(self.state()['tensorboard_pid'])
        self.assertEqual(self.state()['epochs'], 8)
        self.assertFalse(list((self.package / 'runs/_launcher').glob('*/tensorboard.log')))

    def test_resume_uses_original_run_without_checking_current_inputs(self):
        run = 'runs/person/old run'
        checkpoint = self.package / run / 'training/weights/last.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        result = self.invoke('v4', 17, resume=run, FAKE_CHECK_EXIT='13')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.package/'check.json').exists())
        train = json.loads((self.package/'train.json').read_text())
        self.assertEqual(train['argv'], [
            'train', '--resume', 'runs/person/old run', '--wall-seconds', '17.0',
        ])
        self.assertEqual(self.state()['resume'], run)
        self.assertEqual(self.state()['run'], run)

    def test_resume_rejects_new_epoch_budget_before_starting_children(self):
        result = self.invoke('v4', 17, resume='runs/person/old run', epochs=8)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('original epoch budget', result.stderr)
        self.assertFalse((self.package / 'train.json').exists())
        self.assertIsNone(self.state()['process_tree'])

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

    def test_windows_job_failure_stops_before_any_child(self):
        result = self.invoke('v4', 10, FAKE_JOB_FAILURE='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Windows Job Object', result.stderr)
        self.assertEqual(self.state()['status'], 'failed')
        self.assertFalse((self.package / 'check.json').exists())
        self.assertFalse((self.package / 'train.json').exists())
        self.assertIsNone(self.state()['tensorboard_pid'])

    def test_killing_launcher_terminates_training_and_tensorboard_descendants(self):
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
        bootstrap = self.package.parent / 'portable_env/bootstrap.py'
        command = [sys.executable, '-B', str(bootstrap.with_name('fixture_launcher.py')),
                   str(self.package / 'scripts/launch.py'), '--root', str(self.package),
                   '--port', str(port), '--no-wait', '--wall-seconds', '30']
        environment = {**os.environ, 'AEC_PORTABLE_BOOTSTRAP': str(bootstrap), 'FAKE_SPAWN_TREE': '1'}
        handles = []
        with subprocess.Popen(command, cwd=self.package, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as launcher:
            try:
                deadline = time.monotonic() + 12
                paths = [self.package / name for name in ('training_processes.json', 'tensorboard_processes.json')]
                while not all(path.is_file() and path.stat().st_size for path in paths):
                    if launcher.poll() is not None:
                        self.fail('Launcher exited before test processes started: ' + launcher.communicate()[0])
                    if time.monotonic() >= deadline:
                        self.fail('Fake descendants did not start')
                    time.sleep(.05)
                pids = [pid for path in paths for pid in json.loads(path.read_text(encoding='utf-8'))]
                for pid in pids:
                    handle = kernel.OpenProcess(0x100001, False, pid)  # SYNCHRONIZE | PROCESS_TERMINATE
                    self.assertTrue(handle, f'Could not open live fake process {pid}')
                    handles.append(handle)
                self.assertEqual(self.state()['process_tree'], 'windows_job_kill_on_close')
                launcher.kill()  # Kill only the launcher, bypassing its finally block.
                launcher.wait(timeout=10)
                for pid, handle in zip(pids, handles):
                    self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0, f'Fake process {pid} survived')
            finally:
                if launcher.poll() is None:
                    launcher.kill()
                    launcher.wait(timeout=10)
                for handle in handles:
                    if kernel.WaitForSingleObject(handle, 0) == 258:
                        kernel.TerminateProcess(handle, 1)
                    kernel.CloseHandle(handle)


if __name__ == '__main__':
    unittest.main()
