"""Launch a portable training package and its optional TensorBoard process."""
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


def _join_windows_job():
    """Contain the launcher before it creates any descendant processes."""
    import ctypes
    from ctypes import wintypes

    class BasicJobLimits(ctypes.Structure):
        _fields_ = [('per_process_user_time', ctypes.c_longlong), ('per_job_user_time', ctypes.c_longlong),
                    ('flags', wintypes.DWORD), ('minimum_working_set', ctypes.c_size_t),
                    ('maximum_working_set', ctypes.c_size_t), ('active_process_limit', wintypes.DWORD),
                    ('affinity', ctypes.c_size_t), ('priority_class', wintypes.DWORD),
                    ('scheduling_class', wintypes.DWORD)]

    class ExtendedJobLimits(ctypes.Structure):
        _fields_ = [('basic', BasicJobLimits), ('io_counters', ctypes.c_ulonglong * 6),
                    ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                    ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.argtypes = ()
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL

    handle = kernel.CreateJobObjectW(None, None)  # Null security attributes make the only handle non-inheritable.
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        limits = ExtendedJobLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        info_class = 9  # JobObjectExtendedLimitInformation
        if not kernel.SetInformationJobObject(handle, info_class, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    # The launcher itself is a member. Windows closes the handle at process exit,
    # so its descendants are killed even if Python never executes a finally block.


def python_command(script=None, module=None, arguments=()):
    bootstrap = os.environ.get('AEC_PORTABLE_BOOTSTRAP')
    if bootstrap:
        command = [sys.executable, '-I', '-S', '-B', bootstrap, '--offline']
        command += ['--script', str(script)] if script else ['--module', module]
        return command + ['--'] + list(arguments)
    return [sys.executable, '-B', '-u'] + ([str(script)] if script else ['-m', module]) + list(arguments)


def _wait_for_tensorboard(process, port, log_path):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('TensorBoard stopped; see ' + str(log_path))
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=.2):
                return
        except PermissionError as error:
            raise RuntimeError('TensorBoard loopback connection denied; use --no-tensorboard to disable it') from error
        except OSError:
            time.sleep(.2)
    raise RuntimeError('TensorBoard startup timed out; see ' + str(log_path))


def _existing_resume_run(root, value):
    if not value:
        return None
    try:
        path = (root / value).resolve(strict=True)
    except (OSError, ValueError):
        return None  # The training entry reports invalid resume paths.
    if path.is_file():
        if len(path.parents) < 3 or path.parts[-3:] != ('training', 'weights', 'last.pt'):
            return None
        path = path.parents[2]
    if path.is_relative_to(root / 'runs') and (path / 'training/weights/last.pt').is_file():
        return path.relative_to(root).as_posix()
    return None


def main(argv=None, *, entry=None, model_root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--no-wait', action='store_true')
    parser.add_argument('--no-tensorboard', action='store_true', help='Run without starting TensorBoard')
    parser.add_argument('--open-browser', action='store_true', help='Open the ready TensorBoard in the default browser')
    parser.add_argument('--config')
    parser.add_argument('--resume', help='Resume an existing run directory or its last.pt')
    parser.add_argument('--epochs', type=int, help='Epoch budget for a new V4 policy run')
    parser.add_argument('--run-name', help='Name of a new run under the model output directory')
    parser.add_argument('--wall-seconds', type=float, help='Positive finite training time limit in seconds')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    entry = entry or root / 'scripts/train.py'
    if not entry.is_file():
        parser.error('Missing training entry: ' + str(entry))
    if not 0 <= args.port <= 65535:
        parser.error('--port must be 0 (automatic) or a port from 1 to 65535')
    if args.run_name and (args.resume or Path(args.run_name).name != args.run_name or args.run_name in {'.', '..'}):
        parser.error('--run-name must be a single directory name and is only for new training')
    tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logroot = root / 'runs/_launcher' / tag
    logroot.mkdir(parents=True, exist_ok=False)
    for name in ('temp', 'cache', 'config'):
        (logroot / name).mkdir()
    finite_limit = args.wall_seconds is None or math.isfinite(args.wall_seconds)
    config = args.config or 'config/train.yaml'
    target = model_root / args.run_name if model_root is not None and args.run_name else None
    state = {'status': 'starting', 'config': args.config if args.resume else config, 'resume': args.resume,
             'run': target.relative_to(root).as_posix() if target else _existing_resume_run(root, args.resume),
             'launcher_pid': os.getpid(), 'train_pid': None, 'epochs': args.epochs,
             'process_tree': None, 'tensorboard_pid': None, 'tensorboard_url': None,
             'wall_seconds': args.wall_seconds if finite_limit else str(args.wall_seconds)}
    def save_state(**values):
        state.update(values)
        temporary = logroot / 'status.json.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(logroot / 'status.json')
    save_state()
    if args.wall_seconds is not None and (not math.isfinite(args.wall_seconds) or args.wall_seconds <= 0):
        save_state(status='failed', error='wall-seconds must be finite and positive')
        raise ValueError('wall-seconds must be finite and positive')
    if args.epochs is not None and (args.epochs <= 0 or args.resume):
        message = '--resume uses the original epoch budget' if args.resume else '--epochs must be positive'
        save_state(status='failed', error=message)
        raise ValueError(message)
    manifest = root / 'package_manifest.json'
    if not manifest.is_file():
        manifest = root / 'mirror_manifest.json'
    try:
        metadata = json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else {}
        if not isinstance(metadata, dict):
            raise ValueError('Expected a package manifest object')
        package_version = metadata.get('version')
    except (OSError, ValueError, TypeError) as error:
        save_state(status='failed', error='Invalid package manifest: ' + str(error))
        raise ValueError('Invalid package manifest') from error
    if str(package_version).lower() == 'v4' and args.wall_seconds is None:
        save_state(status='failed', error='V4 training requires --wall-seconds')
        raise ValueError('V4 training requires --wall-seconds; supply -WallSeconds to the PowerShell launcher')
    os.environ.update(TEMP=str(logroot/'temp'), TMP=str(logroot/'temp'),
                      MPLCONFIGDIR=str(logroot/'cache'), YOLO_CONFIG_DIR=str(logroot/'config'),
                      PYTHONDONTWRITEBYTECODE='1', PIP_NO_INDEX='1')
    process, train = None, None
    try:
        if os.name == 'nt':
            try:
                _join_windows_job()
            except OSError as error:
                raise RuntimeError('Could not contain launcher processes in a Windows Job Object: ' + str(error)) from error
            save_state(process_tree='windows_job_kill_on_close')

        epoch_args = ['--epochs', str(args.epochs)] if args.epochs is not None else []
        if not args.resume:
            save_state(status='checking')
            check_args = ['check', '--config', config, *epoch_args]
            check = subprocess.run(python_command(entry, arguments=check_args), cwd=root)
            if check.returncode:
                save_state(status='check_failed', exit_code=check.returncode)
                return check.returncode

        if not args.no_tensorboard:
            try:
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', args.port))
                    port = probe.getsockname()[1]
            except OSError as error:
                raise RuntimeError('TensorBoard port unavailable: ' + str(error)) from error
            log_path = logroot / 'tensorboard.log'
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            board_logdir = model_root or root / 'runs'
            viewer = Path(__file__).with_name('tensorboard_v3.py')
            if not viewer.is_file():
                viewer = Path(__file__).with_name('view_tensorboard.py')
            if model_root is not None and viewer.is_file():
                board_logdir = logroot / 'tensorboard_view'
                command = python_command(viewer, arguments=['--source', str(model_root), '--logdir', str(board_logdir),
                                                              '--port', str(port), '--no-browser'])
            else:
                command = python_command(module='tensorboard.main', arguments=['--logdir', str(board_logdir),
                                         '--host', '127.0.0.1', '--port', str(port)])
            with log_path.open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command,
                                           cwd=root, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            save_state(status='starting_tensorboard', tensorboard_pid=process.pid)
            _wait_for_tensorboard(process, port, log_path)
            url = f'http://127.0.0.1:{port}/'
            save_state(tensorboard_url=url, tensorboard_logdir=str(board_logdir))
            print('TensorBoard: ' + url, flush=True)
            if args.open_browser:
                try:
                    if os.name == 'nt':
                        os.startfile(url)
                    else:
                        import webbrowser
                        webbrowser.open(url)
                except OSError as error:
                    print(f'Open TensorBoard manually at {url} ({error})', file=sys.stderr)

        train_args = ['train']
        if args.resume:
            train_args += ['--resume', args.resume]
        if not args.resume or args.config:
            train_args += ['--config', config]
        if args.run_name:
            train_args += ['--run-name', args.run_name]
        train_args += epoch_args
        if args.wall_seconds is not None:
            train_args += ['--wall-seconds', str(args.wall_seconds)]
        train = subprocess.Popen(python_command(entry, arguments=train_args), cwd=root)
        save_state(status='training', train_pid=train.pid)
        exit_code = train.wait()
        exits = {0: 'complete', 124: 'budget_exhausted', 130: 'interrupted'}
        save_state(status=exits.get(exit_code, 'failed'),
                   exit_code=exit_code)
        if process is not None and exit_code == 0 and not args.no_wait and sys.stdin.isatty():
            try:
                input('Training finished. Press Enter to stop TensorBoard.')
            except EOFError:
                pass
        return exit_code
    except KeyboardInterrupt:
        # The foreground training entry shares the console and receives Ctrl+C too.
        # Give it time to record the interruption before the launcher job closes.
        if train is not None and train.poll() is None:
            try:
                train.wait(timeout=10)
            except subprocess.TimeoutExpired:
                train.terminate()
        if state.get('exit_code') is None:
            save_state(status='interrupted', exit_code=130, error='KeyboardInterrupt')
        raise
    except BaseException as error:
        save_state(status='failed', error=f'{type(error).__name__}: {error}')
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
