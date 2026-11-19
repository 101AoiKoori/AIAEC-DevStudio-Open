"""Run-local output, offline execution, logs and provenance."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import importlib.util
import hashlib
import filecmp
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
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.json-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def training_lock(run_dir):
    """Keep one trainer per run, including while an interrupted run is resumed."""
    with (Path(run_dir) / 'metadata/training.lock').open('a+b') as lock:
        try:
            if os.name == 'nt':
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Cannot acquire training lock for ' + str(run_dir)) from error
        yield


def new_run(root, config, name=None):
    from .config import resolve, validate
    validate(config)
    root = Path(root).resolve()
    base = resolve(root, config['output_root'])
    if 'runs' not in base.relative_to(root).parts:
        raise ValueError('Resolved output_root must stay inside a runs directory')
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


def _portable_environment_identity(root, manifest):
    """Compare manifest identities, not the actual dependency files."""
    selected = os.environ.get('AEC_PORTABLE_ENV')
    if not selected:
        return {'status': 'not_portable'}
    result = {'status': 'unavailable', 'selected': selected,
              'manifest_sha256': None, 'build_manifest_sha256': None}
    try:
        bundle = Path(selected).resolve()
        portable_root = (Path(root) / manifest.get('portable_environment', '../portable_env')).resolve()
        relative = bundle.relative_to(portable_root).as_posix()
        result['bundle'] = relative
        if relative not in ('.', 'cp314'):
            return result
        expected = manifest.get('portable_environment_manifests_sha256')
        if not isinstance(expected, dict):
            expected = {'.': manifest.get('portable_environment_manifest_sha256')}
        result['build_manifest_sha256'] = expected.get(relative)
        digest = hashlib.sha256()
        with (bundle / 'manifest.json').open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        result['manifest_sha256'] = digest.hexdigest()
        if result['build_manifest_sha256']:
            result['status'] = ('matched' if result['manifest_sha256'] == result['build_manifest_sha256']
                                else 'changed')
    except (OSError, ValueError, TypeError):
        pass
    return result


def environment(root):
    packages = {}
    for package in ('torch', 'torchvision', 'ultralytics', 'numpy', 'opencv-python', 'tensorboard', 'matplotlib', 'PyYAML'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result = {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
              'packages': packages, 'command': sys.argv, 'recorded_at_utc': datetime.now(timezone.utc).isoformat()}
    # A portable package is a snapshot, even when physically nested inside a
    # different Git checkout. Never attribute it to an enclosing repository.
    package_manifest = Path(root) / 'package_manifest.json'
    if package_manifest.is_file():
        result.update(git_commit=None, git_worktree_modified=None)
        manifest = {}
        try:
            manifest = json.loads(package_manifest.read_text(encoding='utf-8'))
            if not isinstance(manifest, dict):
                raise ValueError('invalid package manifest')
            fingerprint = manifest['code_fingerprint']
            expected = fingerprint['files']
            if manifest.get('package_kind') != 'aiaec_v4_training' or fingerprint.get('algorithm') != 'sha256':
                raise ValueError('unrecognized training package manifest')
            actual = {}
            for name in expected:
                path = (Path(root) / name).resolve()
                if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
                    raise ValueError('packaged code path missing or outside package')
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                actual[name] = digest.hexdigest()
            payload = json.dumps(actual, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
            actual_digest = hashlib.sha256(payload).hexdigest()
            result['package_code_fingerprint'] = fingerprint['digest']
            result['package_code_current_fingerprint'] = actual_digest
            result['package_code_status'] = 'verified' if actual_digest == fingerprint['digest'] else 'modified'
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            result['package_code_status'] = 'unavailable'
        if not isinstance(manifest, dict):
            manifest = {}
        result['portable_environment_identity'] = _portable_environment_identity(root, manifest)
        return result
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
    os.environ.update({'YOLO_AUTOINSTALL': 'false', 'YOLO_OFFLINE': 'true', 'WANDB_MODE': 'disabled',
                       'COMET_MODE': 'DISABLED', 'MPLBACKEND': 'Agg', 'PYTHONDONTWRITEBYTECODE': '1',
                       'NO_PROXY': '*', 'no_proxy': '*'})
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(run_dir / 'temp')
    # Use installed CJK fonts when present; retain the bundled offline fallback.
    spec = importlib.util.find_spec('matplotlib')
    if spec and spec.origin:
        font = Path(spec.origin).parent / 'mpl-data/fonts/ttf/DejaVuSans.ttf'
        cjk = next((path for path in (
            Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/msyh.ttc',
            Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
            Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
        ) if path.is_file()), None)
        if font.is_file():
            for name in ('Arial.ttf', 'Arial.Unicode.ttf'):
                source = cjk if name == 'Arial.Unicode.ttf' and cjk else font
                destination = run_dir / 'cache/ultralytics' / name
                # FreeType may keep the cached font memory-mapped on Windows.
                if not destination.is_file() or not filecmp.cmp(source, destination, shallow=False):
                    shutil.copy2(source, destination)
        if cjk:
            import matplotlib
            from matplotlib import font_manager
            font_manager.fontManager.addfont(str(cjk))
            family = font_manager.FontProperties(fname=str(cjk)).get_name()
            matplotlib.rcParams['font.family'] = [family, 'DejaVu Sans']


@contextmanager
def run_context(root, config, run_dir, phase):
    """Capture status and all local caches; refuse remote network connections."""
    run_dir = Path(run_dir).resolve()
    for directory in ('logs', 'metadata', 'config'):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    environment_keys = ('YOLO_CONFIG_DIR', 'MPLCONFIGDIR', 'TORCH_HOME', 'HF_HOME', 'XDG_CACHE_HOME',
                        'TEMP', 'TMP', 'TMPDIR', 'NUMBA_CACHE_DIR', 'YOLO_AUTOINSTALL', 'WANDB_MODE',
                        'COMET_MODE', 'MPLBACKEND', 'PYTHONDONTWRITEBYTECODE', 'YOLO_OFFLINE',
                        'NO_PROXY', 'no_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                        'http_proxy', 'https_proxy', 'all_proxy')
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
            from .bounded import BudgetExhausted
            status = 'budget_exhausted' if isinstance(error, BudgetExhausted) else ('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
            metadata.update(status=status,
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


def _block_progress(bar, width=12):
    filled = int(min(1.0, max(0.0, bar.n / bar.total)) * width) if bar.total else (width if bar.closed else 0)
    return '█' * filled + '░' * (width - filled)


def prepare_ultralytics(run_dir):
    """Configure installed public settings after run-local environment is ready."""
    from ultralytics import YOLO, settings
    from ultralytics import utils
    from ultralytics.utils import checks
    if os.environ.get('AEC_TRAIN_PROGRESS') == 'blocks':
        # The pinned Ultralytics formatter calls this method for each progress update.
        utils.TQDM._generate_bar = _block_progress
    settings.file = Path(run_dir) / 'cache/ultralytics/settings.json'
    utils.USER_CONFIG_DIR = Path(run_dir) / 'cache/ultralytics'
    checks.USER_CONFIG_DIR = utils.USER_CONFIG_DIR
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
