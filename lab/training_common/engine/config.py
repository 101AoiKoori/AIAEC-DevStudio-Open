"""Validate explicit training options and workspace-relative paths."""
from pathlib import Path
import json
import math
import yaml

TRAIN_KEYS = {'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs',
              'warmup_momentum', 'warmup_bias_lr', 'cos_lr', 'close_mosaic', 'amp',
              'deterministic', 'box', 'cls', 'dfl', 'pose', 'kobj', 'dropout', 'freeze'}
VALIDATING_MODES = ('fixed_validation', 'early_stopping')
AUGMENT_KEYS = {'degrees', 'translate', 'scale', 'shear', 'perspective', 'flipud',
                'fliplr', 'mosaic', 'mixup', 'copy_paste', 'hsv_h', 'hsv_s', 'hsv_v'}
BASE_KEYS = {'schema_version', 'task_id', 'model_key', 'version', 'task', 'data', 'model',
             'output_root', 'imgsz', 'epochs', 'batch', 'workers', 'patience', 'seed',
             'device', 'train_limit', 'eval_limit', 'augment', 'data_overrides',
             'source_notes', 'time_hours', 'max_det', 'diagnostic_conf', 'match_iou',
             'preview_limit', 'evaluate_after_training', 'tune', 'description',
             'amp_check_model', 'keypoint_names', 'class_names', 'nms_iou', 'test_unavailable',
             'training_policy'}


def read_yaml(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig')
    # Frozen run configurations are JSON. YAML 1.1 treats JSON's 5e-05 as a
    # string, which otherwise changes configuration types during freeze/resume.
    value = json.loads(text) if path.suffix.lower() == '.json' else yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError('Expected YAML mapping: ' + str(path))
    return value


def resolve(root, value, allow_absolute=False):
    root = Path(root).resolve()
    value = Path(value)
    if value.is_absolute() and not allow_absolute:
        raise ValueError('Expected root-relative path: ' + str(value))
    result = (root / value).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Path escapes workspace/package: ' + str(value))
    return result


def _validate_training_policy(config):
    policy = config['training_policy']
    if config['version'].lower() != 'v4' or not isinstance(policy, dict):
        raise ValueError('training_policy requires a V4 configuration and a mapping')
    mode = policy.get('mode')
    if mode not in ('fixed', *VALIDATING_MODES):
        raise ValueError('training_policy.mode must be fixed, fixed_validation or early_stopping')
    keys = {'mode', 'save_period'}
    if mode == 'early_stopping':
        keys.update(('min_epochs', 'patience', 'min_delta'))
    if set(policy) != keys:
        raise ValueError('training_policy requires exactly: ' + ', '.join(sorted(keys)))
    if type(policy['save_period']) is not int or policy['save_period'] < 1:
        raise ValueError('training_policy.save_period must be a positive integer')
    if config.get('patience', 0) != 0:
        raise ValueError('training_policy replaces top-level patience')
    if mode == 'fixed' and config.get('evaluate_after_training', False):
        raise ValueError('evaluate_after_training requires a training policy with validation')
    if mode in VALIDATING_MODES and config.get('eval_limit') is not None:
        raise ValueError('Validation training policies use the full validation split')
    if mode == 'early_stopping':
        for key in ('min_epochs', 'patience'):
            if type(policy[key]) is not int or policy[key] < 1:
                raise ValueError('training_policy.' + key + ' must be a positive integer')
        if config.get('epochs') is not None and policy['min_epochs'] > config['epochs']:
            raise ValueError('training_policy.min_epochs cannot exceed epochs')
        delta = policy['min_delta']
        if type(delta) not in (int, float) or not 0 <= delta < 1:
            raise ValueError('training_policy.min_delta must be in [0, 1)')


def validate(config, *, require_epochs=False):
    unknown = set(config) - BASE_KEYS - TRAIN_KEYS
    if unknown:
        raise ValueError('Unsupported configuration keys: ' + ', '.join(sorted(unknown)))
    if config.get('schema_version') != 3 or config.get('task') not in {'detect', 'pose'}:
        raise ValueError('Require schema_version: 3 and task: detect/pose')
    for key in ('task_id', 'model_key', 'version', 'data', 'model', 'output_root'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError('Missing nonempty string: ' + key)
    output = Path(config['output_root'])
    if output.is_absolute() or output.drive or '..' in output.parts:
        raise ValueError('output_root must be relative and cannot contain ..')
    if 'runs' not in output.parts:
        raise ValueError('output_root must be inside a runs directory')
    for key in ('imgsz', 'batch'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    epochs = config.get('epochs')
    if epochs is None and 'training_policy' in config:
        if require_epochs:
            raise ValueError('Training budget is unset. Supply --epochs or set epochs in the configuration')
    elif type(epochs) is not int or epochs < 1:
        raise ValueError('epochs must be a positive integer')
    if config['imgsz'] % 32:
        raise ValueError('imgsz must be divisible by 32')
    for key, default in (('workers', 0), ('patience', 30), ('seed', 42), ('preview_limit', 64)):
        if type(config.get(key, default)) is not int or config.get(key, default) < 0:
            raise ValueError(key + ' must be a nonnegative integer')
    for key in ('train_limit', 'eval_limit'):
        if config.get(key) is not None and (type(config[key]) is not int or config[key] < 1):
            raise ValueError(key + ' must be null or a positive integer')
    for key in ('diagnostic_conf', 'match_iou', 'nms_iou'):
        if key in config and not 0 < float(config[key]) <= 1:
            raise ValueError(key + ' must be in (0, 1]')
    for key in ('evaluate_after_training', 'amp', 'cos_lr', 'deterministic'):
        if key in config and type(config[key]) is not bool:
            raise ValueError(key + ' must be boolean')
    if type(config.get('test_unavailable', False)) is not bool:
        raise ValueError('test_unavailable must be boolean')
    if config.get('test_unavailable') and str(config.get('version', '')).lower() != 'v4':
        raise ValueError('test_unavailable is supported only for V4')
    if config.get('time_hours') is not None and float(config['time_hours']) <= 0:
        raise ValueError('time_hours must be positive')
    if str(config.get('version', '')).lower() == 'v4' and config.get('time_hours') is not None:
        raise ValueError('V4 uses an external --wall-seconds limit; internal time_hours changes the schedule')
    device = str(config.get('device', '0'))
    if ',' in device or isinstance(config.get('device'), list):
        raise ValueError('This portable runner supports one CPU or CUDA device per process')
    augment = config.get('augment', {})
    if not isinstance(augment, dict) or set(augment) - AUGMENT_KEYS:
        raise ValueError('Unsupported augmentation options')
    for key, value in augment.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid augmentation value: ' + key)
    if config['task'] == 'pose' and any(augment.get(key, 0) for key in ('flipud', 'fliplr')):
        raise ValueError('Pose keypoint order requires flips disabled')
    if 'training_policy' in config:
        _validate_training_policy(config)
    return config


def load_config(root, path):
    return validate(read_yaml(resolve(root, path)))
