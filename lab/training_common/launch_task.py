"""Start or resume one V4 model with the Windows launcher defaults."""
import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import sys


TASK_FEATURES = {'person': 'person', 'e_bike': 'e_bike', 'fire': 'fire',
                 'building_name_paper': 'fire', 'sign': 'sign', 'waste_bin_state': 'waste_bin',
                 'waste_sticker': 'waste_bin', 'meter': 'meter'}
DEFAULTS = {'train_config': None, 'wall_seconds': None, 'tensorboard': True,
            'open_browser': True, 'port': 0, 'keep_tensorboard_open': True}


def _finished(run, epochs):
    from monitor_training import _json

    artifacts = _json(run / 'metadata/artifacts.json')
    completed = artifacts.get('completed_epochs')
    return (_json(run / 'metadata/train_status.json').get('status') == 'complete'
            or (type(completed) is int and completed >= epochs)
            or artifacts.get('stop_reason') == 'validation_early_stop')


def targets(root, task):
    if not task:
        return root / 'scripts/train.py', 'config/train.yaml'
    training = root / 'lab/features' / TASK_FEATURES[task] / 'training'
    model = 'fire_sticker' if task == 'fire' else task
    return training / 'scripts' / f'train_{model}_v4.py', (
        training / 'config' / f'{model}_v4.yaml').relative_to(root).as_posix()


def _positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Wall-clock seconds must be finite and positive.')
    return seconds


def launcher_settings(path, task):
    import yaml
    from engine.config import read_yaml

    try:
        settings = read_yaml(path)
    except yaml.YAMLError as error:
        raise ValueError(f'Invalid launcher YAML {path}: {error}') from error
    tasks = settings.pop('tasks', {})
    if not isinstance(tasks, dict) or set(tasks) - TASK_FEATURES.keys():
        raise ValueError(f'Unknown or invalid tasks in {path}')
    for section in (settings, *tasks.values()):
        if not isinstance(section, dict) or set(section) - DEFAULTS.keys():
            raise ValueError(f'Unknown or invalid launcher settings in {path}')
    settings = {**DEFAULTS, **settings, **tasks.get(task, {})}
    if settings['train_config'] is not None and not isinstance(settings['train_config'], str):
        raise ValueError(f'train_config must be a relative path in {path}')
    wall = settings['wall_seconds']
    if wall is not None and (type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0):
        raise ValueError(f'wall_seconds must be a positive number in {path}')
    if type(settings['port']) is not int or not 0 <= settings['port'] <= 65535:
        raise ValueError(f'port must be an integer between 0 and 65535 in {path}')
    for key in ('tensorboard', 'open_browser', 'keep_tensorboard_open'):
        if type(settings[key]) is not bool:
            raise ValueError(f'{key} must be true or false in {path}')
    return settings


def latest_checkpoint(model_root, model_key):
    from monitor_training import _json

    if model_root.is_dir():
        candidates = []
        for run in model_root.iterdir():
            if run.name.startswith('_') or run.resolve().parent != model_root:
                continue
            checkpoint = run / 'training/weights/last.pt'
            config = _json(run / 'config/effective.json')
            if (checkpoint.is_file() and str(config.get('version', '')).lower() == 'v4'
                    and config.get('model_key') == model_key and type(config.get('epochs')) is int
                    and not _finished(run, config['epochs'])):
                candidates.append(checkpoint)
        if candidates:
            return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path))).parents[2]
    raise ValueError(f'No unfinished V4 run with last.pt under {model_root}.')


def _previous_limit(root, run):
    from monitor_training import _json

    key = run.relative_to(root).as_posix()
    candidates = []
    local = run / 'metadata/wall_budget.json'
    if local.is_file():
        state = _json(local)
        try:
            candidates.append((local.stat().st_mtime_ns, _positive_seconds(state['wall_seconds'])))
        except (KeyError, OSError, TypeError, ValueError, OverflowError):
            pass
    for receipt in (root / 'runs/_launcher').glob('*/status.json'):
        state = _json(receipt)
        if state.get('run') == key and state.get('train_pid') and state.get('wall_seconds'):
            try:
                candidates.append((receipt.stat().st_mtime_ns, _positive_seconds(state['wall_seconds'])))
            except (OSError, TypeError, ValueError, OverflowError):
                continue
    return max(candidates, default=(0, None))[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'resume'))
    parser.add_argument('--task', choices=TASK_FEATURES)
    parser.add_argument('--run', help='Specific run directory instead of the latest V4 run')
    parser.add_argument('--plan', action='store_true', help='Print entry and configuration without starting anything')
    args = parser.parse_args(argv)
    if args.run and args.action != 'resume':
        parser.error('Use --run with resume.')
    here = Path(__file__).resolve()
    root = here.parents[2] if args.task else here.parents[1]
    sys.path.insert(0, str(here.parent))
    if not args.task:
        sys.path.insert(0, str(here.parent / 'common'))
    from engine.config import load_config, resolve, validate

    launcher_path = here.with_name('launcher.yaml') if args.task else root / 'config/launcher.yaml'
    settings = launcher_settings(launcher_path, args.task)
    entry, default_config = targets(root, args.task)
    config_path = settings['train_config'] or default_config
    if not entry.is_file() or not resolve(root, config_path).is_file():
        parser.error(f'Missing entry or config: {entry} | {config_path}')
    config = load_config(root, config_path)
    model_key = config['model_key']
    expected = 'fire_sticker' if args.task == 'fire' else args.task
    if config['version'].lower() != 'v4' or (expected and model_key != expected):
        raise ValueError(f'Configuration must target V4 {expected or model_key}.')
    if args.plan:
        print(f'{args.action}: {entry}\nTraining config: {config_path}\nLauncher config: {launcher_path}')
        print(f'Model: {model_key}\nEpochs: {config.get("epochs")}\nWall seconds: {settings["wall_seconds"]}\n'
              f'Training mode: {config.get("training_policy", {}).get("mode", "legacy")}\n'
              f'Post-training validation: {config.get("evaluate_after_training", False)}\n'
              f'TensorBoard: {settings["tensorboard"]}')
        return 0

    from engine.runner import recorded_resume_config
    from engine.runtime import training_lock
    if args.task:
        from launch_v3 import main as launch
    else:
        from launch import main as launch

    model_root = resolve(root, config['output_root'])
    args_for_launch = ['--root', str(root), '--port', str(settings['port'])]
    if settings['open_browser']:
        args_for_launch.append('--open-browser')
    seconds = settings['wall_seconds']
    lock_root = model_root / '_one_click'
    (lock_root / 'metadata').mkdir(parents=True, exist_ok=True)
    with training_lock(lock_root):
        if args.action == 'resume':
            if args.run:
                value = Path(args.run)
                if value.name == args.run and args.run not in {'.', '..'}:
                    value = model_root / value
                run = resolve(root, value, allow_absolute=True)
            else:
                run = latest_checkpoint(model_root, model_key)
            if not run.is_dir():
                raise ValueError(f'Expected a run directory: {run}')
            recorded = recorded_resume_config(root, run)
            if recorded['version'].lower() != 'v4' or recorded['model_key'] != model_key:
                raise ValueError(f'Run does not match the V4 {model_key} model: {run}')
            if _finished(run, recorded['epochs']):
                raise ValueError(f'This run has already finished: {run}')
            print(f'Resume {run} (original budget: {recorded["epochs"]} epochs)', flush=True)
            args_for_launch += ['--resume', str(run)]
            seconds = seconds if seconds is not None else _previous_limit(root, run)
        else:
            if 'training_policy' not in config:
                raise ValueError(f'{config_path} needs a reviewed V4 training_policy. '
                                 f'Set train_config in {launcher_path} to select another config.')
            data = resolve(root, config['data'])
            if not data.is_file():
                raise ValueError(f'Dataset is missing: {data}')
            if config.get('epochs') is None:
                raise ValueError(f'Set epochs in the training config: {config_path}')
            validate(config, require_epochs=True)
            name = datetime.now().strftime('v4_%Y%m%d_%H%M%S_%f')
            args_for_launch += ['--config', config_path, '--run-name', name]
        if seconds is None:
            raise ValueError(f'Set wall_seconds in the launcher config: {launcher_path}')
        args_for_launch += ['--wall-seconds', str(seconds)]
        if not settings['tensorboard']:
            args_for_launch.append('--no-tensorboard')
        if not settings['keep_tensorboard_open']:
            args_for_launch.append('--no-wait')
        os.environ['AEC_TRAIN_PROGRESS'] = 'blocks'
        return launch(args_for_launch, entry=entry, model_root=model_root)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (EOFError, KeyboardInterrupt):
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
