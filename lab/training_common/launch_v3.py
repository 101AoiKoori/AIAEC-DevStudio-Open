"""Launch a portable V3 package with a bounded TensorBoard child process."""
import argparse
import datetime
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

def python_command(script=None, module=None, arguments=()):
    bootstrap = os.environ.get('AEC_PORTABLE_BOOTSTRAP')
    if bootstrap:
        command = [sys.executable, '-I', '-S', '-B', bootstrap, '--offline']
        command += ['--script', str(script)] if script else ['--module', module]
        return command + ['--'] + list(arguments)
    return [sys.executable, '-B', '-u'] + ([str(script)] if script else ['-m', module]) + list(arguments)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--no-wait', action='store_true')
    parser.add_argument('--config', default='config/train.yaml')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    entry = root / 'scripts/train.py'
    if not entry.is_file():
        parser.error('Missing package scripts/train.py')
    tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logroot = root / 'runs/_launcher' / tag
    logroot.mkdir(parents=True, exist_ok=False)
    for name in ('temp', 'cache', 'config'):
        (logroot / name).mkdir()
    os.environ.update(TEMP=str(logroot/'temp'), TMP=str(logroot/'temp'),
                      MPLCONFIGDIR=str(logroot/'cache'), YOLO_CONFIG_DIR=str(logroot/'config'),
                      PYTHONDONTWRITEBYTECODE='1', PIP_NO_INDEX='1')
    check = subprocess.run(python_command(entry, arguments=['check', '--config', args.config]), cwd=root)
    if check.returncode:
        return check.returncode
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    process = None
    with (logroot/'tensorboard.log').open('w', encoding='utf-8') as log:
        try:
            process = subprocess.Popen(python_command(module='tensorboard.main',
                arguments=['--logdir', str(root/'runs'), '--host', '127.0.0.1', '--port', str(args.port)]),
                cwd=root, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            deadline = time.monotonic()+40
            ready = False
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('TensorBoard stopped; see '+str(logroot/'tensorboard.log'))
                try:
                    with socket.create_connection(('127.0.0.1', args.port), timeout=.2):
                        ready = True
                    break
                except OSError:
                    time.sleep(.2)
            if not ready:
                raise RuntimeError('TensorBoard startup timed out; see '+str(logroot/'tensorboard.log'))
            print('TensorBoard: http://127.0.0.1:'+str(args.port)+'/', flush=True)
            result = subprocess.run(python_command(entry, arguments=['train', '--config', args.config]), cwd=root)
            if result.returncode == 0 and not args.no_wait and sys.stdin.isatty():
                input('Training finished. Press Enter to stop TensorBoard.')
            return result.returncode
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print('ERROR:', error, file=sys.stderr)
        raise SystemExit(1)
