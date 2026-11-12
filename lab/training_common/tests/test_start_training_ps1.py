"""Exercise the Windows launcher with fake train/TensorBoard scripts only."""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
LAUNCHER = WORKSPACE / 'lab/training_common/start_training.ps1'
SHELLS = [path for path in (shutil.which('powershell.exe'), shutil.which('pwsh.exe')) if path]

FAKE_TRAIN = '''\
import os, pathlib, socket, sys
root=pathlib.Path(__file__).parent
phase=sys.argv[1]
(root/(phase+'.marker')).write_text('fake entry', encoding='utf8')
(root/(phase+'.cwd')).write_text(str(pathlib.Path.cwd()), encoding='utf8')
if phase=='check':
    sys.exit(int(os.environ.get('FAKE_CHECK_EXIT','0')))
with socket.create_connection(('127.0.0.1',int(os.environ['FAKE_PORT'])),timeout=3):
    (root/'tensorboard_was_ready.marker').write_text('yes',encoding='utf8')
sys.exit(int(os.environ.get('FAKE_TRAIN_EXIT','0')))
'''

FAKE_TENSORBOARD = '''\
import argparse, os, pathlib, socket, sys
parser=argparse.ArgumentParser();parser.add_argument('--host');parser.add_argument('--port',type=int)
args=parser.parse_args()
if os.environ.get('FAKE_TB_FAIL'):
    print('intentional fake startup failure',file=sys.stderr,flush=True)
    sys.exit(4)
with socket.socket() as server:
    server.bind((args.host,args.port));server.listen()
    (pathlib.Path(__file__).parent/'tensorboard.pid').write_text(str(os.getpid()))
    while True:
        connection,_=server.accept();connection.close()
'''


@unittest.skipUnless(os.name == 'nt' and SHELLS, 'Windows PowerShell is required')
class LauncherTests(unittest.TestCase):
    def setUp(self):
        tmp = (WORKSPACE / 'tmp').resolve()
        tmp.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='launcher test ', dir=str(tmp))
        root = Path(self.temp.name).resolve()
        root.relative_to(tmp)
        self.addCleanup(self.temp.cleanup)
        self.package = root / 'copied package with spaces'
        self.package.mkdir()
        shutil.copy2(LAUNCHER, self.package / 'start_training.ps1')
        (self.package / 'train.py').write_text(FAKE_TRAIN, encoding='utf-8')
        (self.package / 'tensorboard_log.py').write_text(FAKE_TENSORBOARD, encoding='utf-8')

    @staticmethod
    def free_port():
        with socket.socket() as server:
            server.bind(('127.0.0.1', 0))
            return server.getsockname()[1]

    def invoke(self, shell, port, **overrides):
        env = os.environ.copy()
        env.update(FAKE_PORT=str(port), **overrides)
        return subprocess.run([shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                               str(self.package / 'start_training.ps1'), '-Python', sys.executable,
                               '-Port', str(port), '-NoWait'], cwd=WORKSPACE, env=env,
                              capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)

    def assert_child_stopped(self, shell):
        pid = int((self.package / 'tensorboard.pid').read_text())
        check = subprocess.run([shell, '-NoProfile', '-Command',
                                f'if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 1 }}'],
                               capture_output=True, timeout=20)
        self.assertEqual(check.returncode, 0, 'fake TensorBoard process still running')

    def test_both_shells_run_fake_train_and_cleanup(self):
        for shell in SHELLS:
            with self.subTest(shell=shell):
                for marker in self.package.glob('*.marker'):
                    marker.unlink()
                result = self.invoke(shell, self.free_port())
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('TensorBoard: http://127.0.0.1:', result.stdout)
                self.assertTrue((self.package / 'check.marker').is_file())
                self.assertTrue((self.package / 'train.marker').is_file())
                self.assertTrue((self.package / 'tensorboard_was_ready.marker').is_file())
                self.assertEqual(Path((self.package / 'train.cwd').read_text(encoding='utf8')).resolve(),
                                 self.package.resolve())
                self.assert_child_stopped(shell)

    def test_failed_train_returns_code_and_cleans_child(self):
        result = self.invoke(SHELLS[0], self.free_port(), FAKE_TRAIN_EXIT='17')
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
        self.assert_child_stopped(SHELLS[0])

    def test_failed_check_does_not_launch_training_or_tensorboard(self):
        result = self.invoke(SHELLS[0], self.free_port(), FAKE_CHECK_EXIT='13')
        self.assertEqual(result.returncode, 13, result.stdout + result.stderr)
        self.assertFalse((self.package / 'train.marker').exists())
        self.assertFalse((self.package / 'tensorboard.pid').exists())

    def test_occupied_port_does_not_run_training(self):
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0));occupied.listen()
            result = self.invoke(SHELLS[0], occupied.getsockname()[1])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unavailable', result.stderr)
        self.assertFalse((self.package / 'train.marker').exists())

    def test_failed_tensorboard_does_not_run_training(self):
        result = self.invoke(SHELLS[0], self.free_port(), FAKE_TB_FAIL='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('TensorBoard exited', result.stderr)
        self.assertFalse((self.package / 'train.marker').exists())


if __name__ == '__main__':
    unittest.main()
