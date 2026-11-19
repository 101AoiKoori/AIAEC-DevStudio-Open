"""Read the latest training status without changing the run or starting services."""
import argparse
import csv
import ctypes
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import http.client
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit


ACTIVE = {'starting', 'checking', 'preparing', 'starting_tensorboard', 'training', 'running', 'resuming'}
SUCCESS = {'complete', 'completed'}
FAILURE = {'failed', 'check_failed', 'interrupted', 'budget_exhausted', 'cancelled', 'canceled'}
TIMEOUT_GRACE = 15
LOCAL_HOSTS = {'127.0.0.1', 'localhost', '::1'}


@dataclass(frozen=True)
class Receipt:
    path: Path
    modified: float
    data: dict

    @property
    def status(self):
        return self.data['status']


def _json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, ValueError):
        return {}


def _modified(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _receipt(path):
    data = _json(path)
    status = data.get('status')
    if isinstance(status, str) and status:
        return Receipt(path, _modified(path), data)
    return None


def _latest(receipts):
    return max((item for item in receipts if item), key=lambda item: (item.modified, str(item.path)), default=None)


def _directories(path):
    try:
        return sorted(entry for entry in path.iterdir() if entry.is_dir())
    except OSError:
        return []


def _in_root(root, value):
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    try:
        candidate = (root / value).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if candidate.name == 'last.pt' and candidate.parent.name == 'weights':
        candidate = candidate.parent.parent.parent
    if candidate.is_relative_to(root / 'runs') and candidate != root / 'runs':
        if all(not part.startswith('_') for part in candidate.relative_to(root / 'runs').parts):
            return candidate
    return None


def _task_receipt(run):
    metadata = run / 'metadata'
    primary = _receipt(metadata / 'status.json')
    phase = _phase(primary)
    paths = [metadata / 'train_status.json', *metadata.glob('resume_*_status.json')]
    if primary and (not phase or phase == 'train' or phase.startswith('resume_')):
        paths.append(primary.path)
    return _latest(_receipt(path) for path in paths)


def _locations(root):
    """Only explore run containers, never a run's data, cache or checkpoint trees."""
    runs, controllers = {}, {}
    queue = [(root / 'runs', 0)]
    while queue:
        container, depth = queue.pop()
        for group in _directories(container / '_controllers'):
            receipt = _receipt(group / 'status.json')
            target = _in_root(root, receipt.data.get('run')) if receipt else None
            if target:
                controllers[target] = _latest((controllers.get(target), receipt))
        for directory in _directories(container):
            if directory.name.startswith('_'):
                continue
            task = _task_receipt(directory)
            if task or (directory / 'training/results.csv').is_file() or (directory / 'config/effective.json').is_file():
                runs[directory] = task
            elif depth < 3:
                queue.append((directory, depth + 1))
    return runs, controllers


def _launcher_target(root, launcher):
    return _in_root(root, launcher.data.get('run') or launcher.data.get('resume')) if launcher else None


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and not isinstance(value, bool) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _tail(path, limit=65536):
    try:
        with path.open('rb') as source:
            size = source.seek(0, 2)
            source.seek(max(0, size - limit))
            data = source.read(limit).decode('utf-8', errors='replace')
            return data.split('\n', 1)[-1] if size > limit else data
    except OSError:
        return ''


def _last_step(path):
    for line in reversed(_tail(path).splitlines()):
        try:
            record = json.loads(line)
            if isinstance(record, dict) and type(record.get('step')) is int and type(record.get('epoch')) is int:
                return record
        except ValueError:
            continue
    return {}


def _last_epoch(path):
    try:
        with path.open(newline='', encoding='utf-8-sig') as source:
            result = {}
            for row in csv.DictReader(source):
                clean = {key.strip(): value for key, value in row.items() if key is not None}
                epoch = _number(clean.get('epoch'))
                if len(row) == len(clean) and all(value not in (None, '') for value in clean.values()):
                    if epoch is not None and epoch > 0 and epoch.is_integer():
                        result = clean
            return result
    except (OSError, UnicodeError, csv.Error):
        return {}


def _phase(record):
    phase = record.data.get('phase') if record else None
    return phase if isinstance(phase, str) else ''


def _controller_phase(controller):
    if controller is None:
        return ''
    attempt = controller.data.get('attempt')
    return attempt if isinstance(attempt, str) and attempt else 'train'


def _current_receipt(task, controller, launcher):
    current = _latest((task, controller, launcher))
    # A child may publish completion before its controller has returned and
    # recorded the outcome. An older controller must not mask a later resume.
    if current is task and task and task.status in SUCCESS and controller and controller.status in ACTIVE:
        if _phase(task) == _controller_phase(controller):
            current = controller
    return current


def _controller_stale(controller, now):
    if not controller or controller.status not in ACTIVE:
        return False
    budget = _number(controller.data.get('wall_seconds'))
    if budget is None or budget <= 0:
        return False
    started = controller.modified
    try:
        value = datetime.fromisoformat(controller.data.get('started_at_utc', '').replace('Z', '+00:00'))
        if value.tzinfo is not None:
            started = value.timestamp()
    except (AttributeError, TypeError, ValueError, OSError, OverflowError):
        pass
    return started > 0 and now - started > budget + TIMEOUT_GRACE


@lru_cache(maxsize=1)
def _windows_process_api():
    from ctypes import wintypes

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def _pid_running(pid):
    if os.name != 'nt' or type(pid) is not int or not 0 < pid <= 0xFFFFFFFF:
        return None
    kernel = _windows_process_api()
    handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
    if not handle:
        return False if ctypes.get_last_error() == 87 else None  # ERROR_INVALID_PARAMETER
    try:
        wait = kernel.WaitForSingleObject(handle, 0)
        return {0: False, 258: True}.get(wait)  # WAIT_OBJECT_0, WAIT_TIMEOUT
    finally:
        kernel.CloseHandle(handle)


def _launcher_exited(launcher):
    if not launcher or launcher.status not in ACTIVE:
        return False
    pids = [launcher.data.get('launcher_pid')]
    train_pid = launcher.data.get('train_pid')
    if train_pid is not None or launcher.status in ('training', 'running', 'resuming'):
        pids.append(train_pid)
    return all(_pid_running(pid) is False for pid in pids)


def _tensorboard_health(url, log_root):
    """Confirm the known local TensorBoard endpoint serves this training log root."""
    try:
        address = urlsplit(url)
        port = address.port
        if address.scheme != 'http' or address.hostname not in LOCAL_HOSTS or not port:
            return False
        if address.username or address.password or address.path not in ('', '/') or address.query or address.fragment:
            return False
        connection = http.client.HTTPConnection(address.hostname, port, timeout=0.4)
        try:
            connection.request('GET', '/data/environment')
            response = connection.getresponse()
            if response.status != 200:
                return False
            data = json.loads(response.read(16384))
        finally:
            connection.close()
        location = data.get('data_location') if isinstance(data, dict) else None
        return isinstance(location, str) and Path(location).is_absolute() and Path(location).resolve() == log_root
    except (OSError, UnicodeError, ValueError, http.client.HTTPException):
        return False


def _tensorboard_url(root, launcher):
    if not launcher:
        return None
    url = launcher.data.get('tensorboard_url')
    if not isinstance(url, str):
        match = re.search(r'http://(?:localhost|127\.0\.0\.1|\[::1\]):\d+/', _tail(launcher.path.parent / 'tensorboard.log'))
        url = match.group() if match else None
    log_root = (root / 'runs').resolve()
    recorded_logdir = launcher.data.get('tensorboard_logdir')
    if isinstance(recorded_logdir, str):
        candidate = Path(recorded_logdir).resolve()
        if candidate.is_relative_to(log_root):
            log_root = candidate
    return url if url and _tensorboard_health(url, log_root) else None


def _progress(run, phase):
    metadata = run / 'metadata'
    suffix = '_' + phase if phase.startswith('resume_') else ''
    summary = _json(metadata / f'optimization_summary{suffix}.json')
    update = _last_step(metadata / f'optimizer_updates{suffix}.jsonl')
    budget = _json(metadata / f'optimization_budget{suffix}.json')
    config = _json(run / 'config/effective.json')
    metrics = _last_epoch(run / 'training/results.csv')
    total_steps = summary.get('actual_optimizer_steps')
    steps_from_checkpoint = budget.get('steps_at_checkpoint')
    steps = update.get('step', total_steps if type(total_steps) is int else steps_from_checkpoint)
    if steps is None and type(summary.get('recorded_optimizer_steps')) is int:
        steps = summary['recorded_optimizer_steps']
    lr = update.get('learning_rates')
    if not isinstance(lr, list) or not lr:
        lr = [metrics[key] for key in sorted(metrics) if key.startswith('lr/')]
    return {'epoch': metrics.get('epoch'), 'max_epochs': config.get('epochs', budget.get('epoch_budget')),
            'step_epoch': update.get('epoch'), 'steps': steps, 'steps_confirmed': type(total_steps) is int,
            'loss': {key: _number(value) for key, value in metrics.items() if key.endswith('_loss')},
            'lr': [value for value in (_number(value) for value in lr) if value is not None]}


def snapshot(root, run_dir=None, *, now=None):
    """Return observations for one run and the relevant launcher and controller."""
    root = Path(root).resolve()
    launchers = [_receipt(path / 'status.json') for path in _directories(root / 'runs/_launcher')]
    latest_launcher = _latest(launchers)
    if run_dir is not None:
        run = _in_root(root, run_dir)
        if run is None or (run.exists() and not run.is_dir()):
            raise ValueError('--run-dir must identify a run inside the package runs directory')
        task = _task_receipt(run)
        controller = _latest(item for item in (_receipt(path / 'status.json')
                             for path in _directories(run.parent / '_controllers'))
                             if item and _in_root(root, item.data.get('run')) == run)
        launcher = _latest(item for item in launchers if _launcher_target(root, item) == run)
    else:
        tasks, controllers = _locations(root)
        launcher = latest_launcher
        target = _launcher_target(root, launcher)
        candidates = set(tasks) | set(controllers) | ({target} if target else set())
        def rank(path):
            return (max(_modified(path / 'training/results.csv'),
                        tasks[path].modified if tasks.get(path) else 0,
                        controllers[path].modified if path in controllers else 0,
                        launcher.modified if path == target else 0), str(path))
        run = max(candidates, key=rank, default=None)
        if launcher and target is None and (run is None or launcher.modified > rank(run)[0]):
            # The latest launch may still be checking inputs, or lack a run ID.
            run = None
        task, controller = tasks.get(run), controllers.get(run)
        if target is not None and target != run:
            launcher = _latest(item for item in launchers if _launcher_target(root, item) == run)

    if run:
        # Old fresh-start launchers lack a run ID. Show their package-level
        # receipt, but let the matched task/controller determine the outcome.
        current = _current_receipt(task, controller, launcher if _launcher_target(root, launcher) == run else None)
    else:
        current = launcher
    launcher_display = launcher or latest_launcher
    launcher_matched = bool(run and _launcher_target(root, launcher_display) == run)
    status = current.status if current else 'unknown'
    if (status in ACTIVE and launcher and launcher == latest_launcher and (launcher_matched or current is launcher)
            and _launcher_exited(launcher)):
        status = 'launcher_exited'
    elif current is controller and _controller_stale(controller, time.time() if now is None else now):
        status = 'stale_running'
    phase = _controller_phase(controller) if current is controller else _phase(task)
    progress = _progress(run, phase) if run else {}
    return {'run': run, 'launcher': launcher_display.status if launcher_display else None,
            'launcher_matched': launcher_matched,
            'controller': controller.status if controller else None,
            'task': task.status if task else None, 'phase': phase, 'status': status,
            'error': current.data.get('error') if current else None, 'progress': progress,
            'tensorboard': _tensorboard_url(root, launcher_display)}


def _render(state):
    lines = [f'运行  {state["run"] or "等待运行目录"}']
    for key, label in (('launcher', '启动器'), ('controller', '控制器'), ('task', '训练任务')):
        scope = '（包内最新，未关联此运行）' if key == 'launcher' and state['run'] and state[key] and not state['launcher_matched'] else ''
        lines.append(f'{label}  {state[key] or "暂无状态记录"}{scope}')
    status = state['status']
    if status == 'stale_running':
        lines.append('当前  控制器记录仍为 running，已超过预算，实际进程状态未确认')
    elif status == 'launcher_exited':
        lines.append('当前  launcher_exited（已确认记录中的启动进程退出，训练结果待核对）')
    elif status in ACTIVE:
        lines.append(f'当前  {status}（状态记录，进程未核实）')
    else:
        lines.append(f'当前  {status}')
    if state['error']:
        lines.append(f'原因  {state["error"]}')
    if progress := state['progress']:
        lines.append(f'已记录轮次  {progress["epoch"] or "—"} / {progress["max_epochs"] or "—"}')
        if progress['steps'] is not None:
            qualifier = '实际' if progress['steps_confirmed'] else '已记录'
            step_epoch = f'（第 {progress["step_epoch"]} 轮）' if progress['step_epoch'] else ''
            lines.append(f'优化器更新  {qualifier} {progress["steps"]}{step_epoch}')
        if progress['loss']:
            losses = [f'{name}={value:.6g}' for name, value in progress['loss'].items() if value is not None]
            if losses:
                lines.append('最新损失  ' + '  '.join(losses))
        if progress['lr']:
            lines.append('学习率  ' + '  '.join(f'{item:.6g}' for item in progress['lr']))
    if state['tensorboard']:
        lines.append('TensorBoard  ' + state['tensorboard'])
    return '\n'.join(lines)


def _default_root():
    script = Path(__file__).resolve()
    package = script.parent.parent
    return package if script.parent.name == 'scripts' and (package / 'config/train.yaml').is_file() else Path.cwd()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=_default_root(), help='Training package root')
    parser.add_argument('--run-dir', type=Path, help='Run directory inside the root; defaults to the latest recorded run')
    parser.add_argument('--follow', action='store_true', help='Poll until a terminal status is recorded')
    parser.add_argument('--interval', type=float, default=2.0, help='Seconds between polls')
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be finite and positive')
    try:
        last = None
        run_dir = args.run_dir
        while True:
            state = snapshot(args.root, run_dir)
            rendered = _render(state)
            if rendered != last:
                print(rendered, flush=True)
                last = rendered
            if not args.follow:
                return 0
            if state['status'] in SUCCESS:
                return 0
            if state['status'] in FAILURE:
                return 124 if state['status'] == 'budget_exhausted' else 1
            if state['status'] == 'stale_running':
                return 2
            if state['status'] == 'launcher_exited':
                return 1
            run_dir = state['run'] or run_dir
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
