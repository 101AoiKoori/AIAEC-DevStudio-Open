"""Launch a portable training package with a bounded TensorBoard child process."""
import argparse
import datetime
import json
import math
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
    parser.add_argument('--wall-seconds', type=float, help='Positive finite training time limit in seconds')
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
    finite_limit = args.wall_seconds is None or math.isfinite(args.wall_seconds)
    state = {'status': 'starting', 'config': args.config,
             'wall_seconds': args.wall_seconds if finite_limit else str(args.wall_seconds)}
    def save_state(**values):
        state.update(values)
        (logroot / 'status.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save_state()
    if args.wall_seconds is not None and (not math.isfinite(args.wall_seconds) or args.wall_seconds <= 0):
        save_state(status='failed', error='wall-seconds must be finite and positive')
        raise ValueError('wall-seconds must be finite and positive')
    manifest = root / 'mirror_manifest.json'
    try:
        package_version = json.loads(manifest.read_text(encoding='utf-8')).get('version') if manifest.is_file() else None
    except (OSError, ValueError, TypeError) as error:
        save_state(status='failed', error='Invalid mirror manifest: ' + str(error))
        raise ValueError('Invalid mirror manifest') from error
    if str(package_version).lower() == 'v4' and args.wall_seconds is None:
        save_state(status='failed', error='V4 training requires --wall-seconds')
        raise ValueError('V4 training requires --wall-seconds; supply -WallSeconds to the PowerShell launcher')
    os.environ.update(TEMP=str(logroot/'temp'), TMP=str(logroot/'temp'),
                      MPLCONFIGDIR=str(logroot/'cache'), YOLO_CONFIG_DIR=str(logroot/'config'),
                      PYTHONDONTWRITEBYTECODE='1', PIP_NO_INDEX='1')
    try:
        check = subprocess.run(python_command(entry, arguments=['check', '--config', args.config]), cwd=root)
    except BaseException as error:
        save_state(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   error=f'{type(error).__name__}: {error}')
        raise
    if check.returncode:
        save_state(status='check_failed', exit_code=check.returncode)
        return check.returncode
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', args.port))
    except OSError as error:
        save_state(status='failed', error='TensorBoard port unavailable: ' + str(error))
        raise
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
            train_args = ['train', '--config', args.config]
            if args.wall_seconds is not None:
                train_args += ['--wall-seconds', str(args.wall_seconds)]
            save_state(status='training')
            result = subprocess.run(python_command(entry, arguments=train_args), cwd=root)
            save_state(status='complete' if result.returncode == 0 else
                       ('budget_exhausted' if result.returncode == 124 else 'failed'), exit_code=result.returncode)
            if result.returncode == 0 and not args.no_wait and sys.stdin.isatty():
                input('Training finished. Press Enter to stop TensorBoard.')
            return result.returncode
        except BaseException as error:
            save_state(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                       error=f'{type(error).__name__}: {error}')
            raise
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
