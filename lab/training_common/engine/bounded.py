"""Hard wall-clock limits for child experiments including staging and evaluation."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time


class BudgetExhausted(RuntimeError):
    """The explicitly bounded child process exhausted its wall-clock budget."""


def command(mode, root, config_path, **options):
    script = Path(__file__).with_name('experiment_child.py')
    bootstrap = os.environ.get('AEC_PORTABLE_BOOTSTRAP')
    if bootstrap:
        prefix = [sys.executable, '-I', '-S', '-B', bootstrap, '--script', str(script), '--']
    else:
        prefix = [sys.executable, '-B', str(script)]
    if os.environ.get('AEC_TRAIN_PROGRESS') == 'blocks':
        prefix[1:1] = ['-X', 'utf8']
    arguments = [mode, '--root', str(root), '--config', str(config_path)]
    for key, value in options.items():
        if value is not None: arguments.extend(['--' + key.replace('_', '-'), str(value)])
    return prefix + arguments


def stop_process_tree(process):
    if process.poll() is not None: return True
    tree_stopped = True
    if os.name == 'nt':
        try:
            result = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            tree_stopped = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            tree_stopped = False
        if not tree_stopped and process.poll() is None:
            # The owned Popen handle can still terminate the training child when taskkill is unavailable.
            process.kill()
    else:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        tree_stopped = False
        process.kill(); process.wait(timeout=10)
    return tree_stopped


def run_bounded(arguments, deadline, log_path, cwd, display_output=False):
    """Use one monotonic deadline; kill the owned process tree on timeout."""
    remaining = deadline - time.monotonic()
    if remaining <= 0: raise BudgetExhausted('Experiment wall-clock budget exhausted before launch')
    log_path = Path(log_path); log_path.parent.mkdir(parents=True, exist_ok=True)
    options = {'cwd': str(cwd), 'stdin': subprocess.DEVNULL}
    if os.name == 'nt': options['creationflags'] = subprocess.CREATE_NO_WINDOW
    else: options['start_new_session'] = True
    with log_path.open('wb') as output:
        process = subprocess.Popen(arguments, stdout=output, stderr=subprocess.STDOUT, **options)
        try:
            if display_output:
                with log_path.open('rb') as live_output:
                    while True:
                        chunk = live_output.read(65536)
                        if chunk:
                            if hasattr(sys.stdout, 'buffer'):
                                sys.stdout.buffer.write(chunk)
                            else:
                                sys.stdout.write(chunk.decode('utf-8', errors='replace'))
                            sys.stdout.flush()
                        elif process.poll() is not None:
                            break
                        else:
                            time.sleep(.1)
                        if time.monotonic() >= deadline and process.poll() is None:
                            raise subprocess.TimeoutExpired(arguments, remaining)
                code = process.wait()
            else:
                code = process.wait(timeout=max(.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as error:
            tree_stopped = stop_process_tree(process)
            detail = ('its process tree was stopped' if tree_stopped
                      else 'its subprocess was stopped and descendant cleanup is unverified')
            raise BudgetExhausted('Experiment exceeded wall-clock budget; ' + detail) from error
        except BaseException:
            stop_process_tree(process)
            raise
    if code:
        raise RuntimeError(f'Experiment child exited with code {code}; inspect {log_path.name}')
    return code
