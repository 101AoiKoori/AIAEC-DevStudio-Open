"""Run-local output, offline execution, logs and provenance."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n', encoding='utf-8')


def new_run(root, config, name=None):
    from .config import resolve
    base = resolve(root, config['output_root'])
    name = name or (config['version'] + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if Path(name).name != name or name in {'.', '..'}:
        raise ValueError('run-name must be one directory name')
    result = base / name
    result.mkdir(parents=True, exist_ok=False)
    for directory in ('metadata', 'logs', 'config', 'cache', 'temp', 'tensorboard'):
        (result / directory).mkdir()
    return result


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log
    def write(self, value):
        self.console.write(value); self.log.write(value); self.log.flush()
        return len(value)
    def flush(self):
        self.console.flush(); self.log.flush()
    def isatty(self):
        return False
    @property
    def encoding(self):
        return self.console.encoding or 'utf-8'


def environment(root):
    packages = {}
    for package in ('torch', 'torchvision', 'ultralytics', 'numpy', 'opencv-python', 'tensorboard', 'matplotlib', 'PyYAML'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result = {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
              'packages': packages, 'command': sys.argv, 'recorded_at_utc': datetime.now(timezone.utc).isoformat()}
    try:
        proc = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, capture_output=True, timeout=10)
        result['git_commit'] = proc.stdout.strip() if proc.returncode == 0 else None
        dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], text=True, capture_output=True, timeout=10)
        result['git_worktree_modified'] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        result['git_commit'] = None
    return result


def setup_paths(run_dir):
    run_dir = Path(run_dir).resolve()
    mapping = {'YOLO_CONFIG_DIR': 'cache/ultralytics', 'MPLCONFIGDIR': 'cache/matplotlib',
               'TORCH_HOME': 'cache/torch', 'HF_HOME': 'cache/huggingface',
               'XDG_CACHE_HOME': 'cache', 'TEMP': 'temp', 'TMP': 'temp',
               'TMPDIR': 'temp', 'NUMBA_CACHE_DIR': 'cache/numba'}
    for key, relative in mapping.items():
        path = run_dir / relative; path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update({'YOLO_AUTOINSTALL': 'false', 'WANDB_MODE': 'disabled',
                       'COMET_MODE': 'DISABLED', 'MPLBACKEND': 'Agg', 'PYTHONDONTWRITEBYTECODE': '1'})
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(run_dir / 'temp')
    # Matplotlib ships this freely distributable font; avoid Ultralytics font downloads.
    spec = importlib.util.find_spec('matplotlib')
    if spec and spec.origin:
        font = Path(spec.origin).parent / 'mpl-data/fonts/ttf/DejaVuSans.ttf'
        if font.is_file():
            for name in ('Arial.ttf', 'Arial.Unicode.ttf'):
                shutil.copy2(font, run_dir / 'cache/ultralytics' / name)


@contextmanager
def run_context(root, config, run_dir, phase):
    """Capture status and all local caches; refuse remote network connections."""
    run_dir = Path(run_dir).resolve()
    for directory in ('logs', 'metadata', 'config'):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    environment_keys = ('YOLO_CONFIG_DIR', 'MPLCONFIGDIR', 'TORCH_HOME', 'HF_HOME', 'XDG_CACHE_HOME',
                        'TEMP', 'TMP', 'TMPDIR', 'NUMBA_CACHE_DIR', 'YOLO_AUTOINSTALL', 'WANDB_MODE',
                        'COMET_MODE', 'MPLBACKEND', 'PYTHONDONTWRITEBYTECODE')
    previous_environment = {key: os.environ.get(key) for key in environment_keys}
    old_tempdir, old_bytecode = tempfile.tempdir, sys.dont_write_bytecode
    setup_paths(run_dir)
    old_out, old_err, old_cwd = sys.stdout, sys.stderr, Path.cwd()
    old_connect, old_connect_ex = socket.socket.connect, socket.socket.connect_ex
    def offline_connect(sock, address):
        if isinstance(address, tuple) and address[0] not in ('127.0.0.1', 'localhost', '::1'):
            raise OSError('Offline training blocks network access: ' + str(address[0]))
        return old_connect(sock, address)
    def offline_connect_ex(sock, address):
        if isinstance(address, tuple) and address[0] not in ('127.0.0.1', 'localhost', '::1'):
            return 10051
        return old_connect_ex(sock, address)
    socket.socket.connect, socket.socket.connect_ex = offline_connect, offline_connect_ex
    start = time.monotonic()
    metadata = {'phase': phase, 'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat()}
    phase_status = run_dir / 'metadata' / (phase.replace('/', '_') + '_status.json')
    write_json(phase_status, metadata)
    write_json(run_dir / 'metadata/status.json', metadata)
    write_json(run_dir / 'config' / (phase.replace('/', '_') + '.json'), config)
    if not (run_dir / 'config/effective.json').exists():
        write_json(run_dir / 'config/effective.json', config)
    recorded_environment = environment(root)
    write_json(run_dir / 'metadata' / ('environment_' + phase.replace('/', '_') + '.json'), recorded_environment)
    if not (run_dir / 'metadata/environment.json').exists():
        write_json(run_dir / 'metadata/environment.json', recorded_environment)
    with (run_dir / 'logs' / (phase.replace('/', '_') + '.log')).open('a', encoding='utf-8') as log:
        sys.stdout, sys.stderr = Tee(old_out, log), Tee(old_err, log)
        try:
            os.chdir(run_dir)
            yield
        except BaseException as error:
            metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                            error=f'{type(error).__name__}: {error}')
            traceback.print_exc()
            raise
        else:
            metadata['status'] = 'complete'
        finally:
            metadata['elapsed_seconds'] = time.monotonic() - start
            metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(phase_status, metadata)
            write_json(run_dir / 'metadata/status.json', metadata)
            sys.stdout, sys.stderr = old_out, old_err
            socket.socket.connect, socket.socket.connect_ex = old_connect, old_connect_ex
            os.chdir(old_cwd)
            tempfile.tempdir, sys.dont_write_bytecode = old_tempdir, old_bytecode
            for key, value in previous_environment.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value


def prepare_ultralytics(run_dir):
    """Configure installed public settings after run-local environment is ready."""
    from ultralytics import YOLO, settings
    from ultralytics import utils
    settings.file = Path(run_dir) / 'cache/ultralytics/settings.json'
    utils.USER_CONFIG_DIR = Path(run_dir) / 'cache/ultralytics'
    utils.SETTINGS_FILE = settings.file
    for handler in utils.LOGGER.handlers:
        if hasattr(handler, 'stream'):
            handler.stream = sys.stdout
    values = {'sync': False, 'tensorboard': True, 'wandb': False, 'mlflow': False,
              'comet': False, 'clearml': False, 'dvc': False, 'neptune': False,
              'raytune': False, 'hub': False,
              'datasets_dir': str(Path(run_dir) / 'data'),
              'weights_dir': str(Path(run_dir) / 'cache/weights'),
              'runs_dir': str(run_dir)}
    settings.update({key: value for key, value in values.items() if key in settings})
    return YOLO
