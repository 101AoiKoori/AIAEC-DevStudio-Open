"""Freeze a validated V4 candidate before explicitly evaluating the test split.

The gate compares files directly and does not calculate or persist hashes. It
freezes the run-local data as the actual test input. It cannot establish label
truth or protect its frozen directory from intentional modification.
"""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

from .config import resolve, read_yaml
from .data import EXTENSIONS, relocate_staged_data


def _run(root, run_dir):
    root = Path(root).resolve()
    run = resolve(root, run_dir, allow_absolute=True)
    if 'runs' not in run.relative_to(root).parts or not run.is_dir():
        raise ValueError('An existing run inside runs is required')
    return root, run


def _read(path):
    with Path(path).open(encoding='utf-8') as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError('Expected JSON mapping: ' + str(path))
    return result


def _same_bytes(first, second):
    first, second = Path(first), Path(second)
    if not first.is_file() or not second.is_file() or first.stat().st_size != second.stat().st_size:
        return False
    with first.open('rb') as one, second.open('rb') as two:
        while True:
            block = one.read(1024 * 1024)
            if block != two.read(1024 * 1024):
                return False
            if not block:
                return True


def _copy_exclusive(source, destination):
    with Path(source).open('rb') as inp, Path(destination).open('xb') as out:
        shutil.copyfileobj(inp, out, 1024 * 1024)


def _name(value):
    if not isinstance(value, str) or not value or Path(value).name != value or value in ('.', '..'):
        raise ValueError('Evaluation name must be one directory name')
    return value


def _staged_files(run):
    """Require the staged split directories and sample list to describe the same files."""
    data = run / 'data'
    if not (data / 'data.yaml').is_file():
        raise ValueError('Staged dataset YAML is missing')
    spec = read_yaml(data / 'data.yaml')
    if 'download' in spec or any(spec.get(split) != 'images/' + split for split in ('train', 'val', 'test')):
        raise ValueError('Staged dataset YAML must point to its local train/val/test directories')
    records = json.loads((data / 'samples.json').read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError('Staged sample manifest is empty')
    images, labels = set(), set()
    counts = {split: 0 for split in ('train', 'val', 'test')}
    for row in records:
        split = row['split']
        if split not in counts:
            raise ValueError('Unexpected staged split')
        image, label = resolve(run, row['run_image']), resolve(run, row['run_label'])
        expected_images, expected_labels = data / 'images' / split, data / 'labels' / split
        if (not image.is_relative_to(expected_images) or not label.is_relative_to(expected_labels) or
                not image.is_file() or not label.is_file() or image.suffix.lower() not in EXTENSIONS or
                label.suffix.lower() != '.txt' or image in images or label in labels):
            raise ValueError('Missing, duplicated, or misplaced staged image/label')
        images.add(image); labels.add(label); counts[split] += 1
    if not all(counts.values()):
        raise ValueError('Each staged split needs at least one sample')
    actual_images = {p.resolve() for p in (data / 'images').rglob('*')
                     if p.is_file() and p.suffix.lower() in EXTENSIONS}
    actual_labels = {p.resolve() for p in (data / 'labels').rglob('*.txt') if p.is_file()}
    if actual_images != images or actual_labels != labels:
        raise ValueError('Staged files disagree with the sample manifest')
    if any(path.is_symlink() for path in data.rglob('*')):
        raise ValueError('Staged dataset must not contain symlinks')
    return sorted(images | labels)


def _same_dataset(first, second):
    """Compare actual snapshot files; permit only YAML's relocatable path to differ."""
    first, second = Path(first), Path(second)
    _staged_files(first.parent); _staged_files(second.parent)
    first_yaml, second_yaml = read_yaml(first / 'data.yaml'), read_yaml(second / 'data.yaml')
    first_yaml.pop('path', None); second_yaml.pop('path', None)
    if first_yaml != second_yaml:
        return False
    def files(base):
        return {p.relative_to(base) for p in base.rglob('*') if p.is_file() and
                p.relative_to(base) != Path('data.yaml') and p.suffix.lower() != '.cache'}
    names = files(first)
    return names == files(second) and all(_same_bytes(first / name, second / name) for name in names)


def _validation(root, run, config, weights, name):
    name = _name(name)
    if str(config.get('version', '')).lower() != 'v4':
        raise ValueError('Freeze requires a V4 configuration')
    checkpoint = resolve(root, weights, allow_absolute=True)
    if not checkpoint.is_file() or checkpoint.suffix.lower() != '.pt':
        raise ValueError('Candidate checkpoint missing')
    summary_path = run / 'evaluation' / name / 'summary.json'
    status_path = run / 'metadata' / ('evaluate_' + name + '_status.json')
    config_path = run / 'config' / ('evaluate_' + name + '.json')
    snapshot_home = run / 'metadata/evaluation_inputs' / name
    summary, status, evaluated_config = _read(summary_path), _read(status_path), _read(config_path)
    if status.get('phase') != 'evaluate_' + name or status.get('status') != 'complete':
        raise ValueError('Validation evaluation did not complete')
    if evaluated_config != config:
        raise ValueError('Current configuration differs from validation configuration')
    if (summary.get('split') != 'val' or summary.get('version') != config['version'] or
            summary.get('model_key') != config['model_key'] or summary.get('task') != config['task']):
        raise ValueError('Validation task, version, or split does not match')
    metrics = summary.get('metrics')
    if (type(summary.get('images')) is not int or summary['images'] < 1 or not isinstance(metrics, dict) or
            not any(type(value) in (int, float) and math.isfinite(value) for value in metrics.values())):
        raise ValueError('Validation summary lacks evaluated images or metrics')
    if resolve(root, summary['checkpoint'], allow_absolute=True) != checkpoint:
        raise ValueError('Candidate differs from validation checkpoint')
    snapshot = resolve(root, summary['checkpoint_snapshot'], allow_absolute=True)
    expected = run / 'metadata/evaluation_inputs' / name / 'checkpoint.pt'
    if snapshot != expected.resolve() or not _same_bytes(snapshot, checkpoint):
        raise ValueError('Validation checkpoint snapshot does not match candidate')
    dataset_snapshot = resolve(root, summary['dataset_snapshot'], allow_absolute=True)
    if dataset_snapshot != (snapshot_home / 'data/data.yaml').resolve():
        raise ValueError('Validation dataset snapshot is outside this evaluation')
    sample_path = snapshot_home / 'data/samples.json'
    if not sample_path.is_file():
        raise ValueError('Run dataset manifest is missing')
    _staged_files(snapshot_home)
    return checkpoint, snapshot, summary_path, status_path, config_path, sample_path, snapshot_home


def freeze_selection(root, config, run_dir, weights, validation_name, selection_reason):
    """Freeze a completed explicit val evaluation; return root-relative selected.pt."""
    if config.get('test_unavailable'):
        raise ValueError('Final test is unavailable; add an audited test set before freezing')
    if not isinstance(selection_reason, str) or not selection_reason.strip() or len(selection_reason) > 2000:
        raise ValueError('A nonempty selection reason of at most 2000 characters is required')
    root, run = _run(root, run_dir)
    checkpoint, snapshot, summary, status, evaluated_config, samples, snapshot_home = _validation(
        root, run, config, weights, validation_name)
    destination = run / 'frozen_selection'
    final_status = run / 'metadata/final_test_status.json'
    previous_status = _read(final_status) if final_status.exists() else {'status': 'not_executed'}
    if previous_status.get('status') != 'not_executed':
        raise ValueError('Final test has already started or completed')
    destination.mkdir(exist_ok=False)
    copies = ((snapshot, destination / 'selected.pt'),
              (summary, destination / 'validation_summary.json'),
              (status, destination / 'validation_status.json'),
              (evaluated_config, destination / 'validation_config.json'),
              (samples, destination / 'samples.json'))
    for source, target in copies:
        _copy_exclusive(source, target)
    shutil.copytree(snapshot_home / 'data', destination / 'data', ignore=shutil.ignore_patterns('*.cache'))
    if not _same_dataset(snapshot_home / 'data', destination / 'data'):
        raise RuntimeError('Validation dataset changed during freezing')
    if not _same_bytes(checkpoint, destination / 'selected.pt') or any(
            not _same_bytes(source, target) for source, target in copies):
        raise RuntimeError('Validation inputs changed during freezing')
    record = {'schema_version': 1, 'status': 'frozen', 'model_key': config['model_key'],
              'version': config['version'], 'task': config['task'], 'validation_name': validation_name,
              'selection_reason': selection_reason.strip(),
              'selected_checkpoint': 'frozen_selection/selected.pt',
              'source_checkpoint': checkpoint.relative_to(root).as_posix(),
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Frozen selection and copied staged inputs are used for test; label truth and tampering are not established.'}
    if final_status.exists() and _read(final_status).get('status') != 'not_executed':
        raise ValueError('Final test has already started or completed')
    previous_status['selection'] = 'frozen_selection/record.json'
    with final_status.open('w', encoding='utf-8') as stream:
        json.dump(previous_status, stream, indent=2)
        stream.write('\n')
    with (destination / 'record.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2); stream.write('\n')
    return (destination / 'selected.pt').relative_to(root).as_posix()


def require_frozen_selection(root, config, run_dir, weights):
    """Reject test when selection, configuration, checkpoint or evidence differ."""
    root, run = _run(root, run_dir)
    selection = run / 'frozen_selection'
    record = _read(selection / 'record.json')
    if (record.get('schema_version') != 1 or record.get('status') != 'frozen' or
            record.get('model_key') != config.get('model_key') or record.get('version') != config.get('version') or
            record.get('task') != config.get('task') or str(config.get('version', '')).lower() != 'v4' or
            not isinstance(record.get('selection_reason'), str) or not record['selection_reason'].strip()):
        raise ValueError('Frozen selection does not match V4 configuration')
    selected = (selection / 'selected.pt').resolve()
    if resolve(root, weights, allow_absolute=True) != selected or not selected.is_file():
        raise ValueError('Test must use the frozen selected checkpoint')
    name = _name(record.get('validation_name'))
    checks = ((run / 'evaluation' / name / 'summary.json', selection / 'validation_summary.json'),
              (run / 'metadata' / ('evaluate_' + name + '_status.json'), selection / 'validation_status.json'),
              (run / 'config' / ('evaluate_' + name + '.json'), selection / 'validation_config.json'),
              (run / 'metadata/evaluation_inputs' / name / 'data/samples.json', selection / 'samples.json'),
              (run / 'metadata/evaluation_inputs' / name / 'checkpoint.pt', selected))
    if any(not _same_bytes(source, target) for source, target in checks):
        raise ValueError('Frozen validation evidence or selected checkpoint changed')
    if _read(selection / 'validation_config.json') != config:
        raise ValueError('Test configuration differs from frozen validation configuration')
    if not _same_dataset(run / 'metadata/evaluation_inputs' / name / 'data', selection / 'data'):
        raise ValueError('Frozen data differs from validation dataset')
    return record


def frozen_dataset_path(root, config, run_dir, weights):
    """Return the selected run's frozen dataset YAML after relocatable path refresh."""
    require_frozen_selection(root, config, run_dir, weights)
    _, run = _run(root, run_dir)
    return relocate_staged_data(run / 'frozen_selection')
