"""Portable YOLOv8 detection/pose entry point. No training without `train`.

Run `python train.py check` first to inspect the data and local model.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
SPLITS = ('train', 'val', 'test')


def within(root, value):
    """Resolve a package-relative path; reject absolute or escaping paths."""
    path = Path(value)
    if path.is_absolute():
        raise ValueError('Expected a relative package path: {}'.format(value))
    result = (root / path).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError:
        raise ValueError('Path escapes package: {}'.format(value))
    return result


def load_yaml(path):
    with path.open('r', encoding='utf-8') as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError('Expected YAML mapping: {}'.format(path))
    return result


def check_label(path, class_count, keypoints):
    rows = path.read_text(encoding='utf-8').splitlines()
    for number, row in enumerate(rows, 1):
        try:
            values = [float(value) for value in row.split()]
            expected = 5 + (keypoints * 3)
            if len(values) != expected or not all(math.isfinite(v) for v in values):
                raise ValueError('bad field count or nonfinite value')
            class_id, x, y, w, h = values[:5]
            if int(class_id) != class_id or not 0 <= int(class_id) < class_count:
                raise ValueError('invalid class')
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError('invalid normalized bounding box')
            if x-w/2 < -1e-6 or x+w/2 > 1+1e-6 or y-h/2 < -1e-6 or y+h/2 > 1+1e-6:
                raise ValueError('bounding box exceeds image')
            for index in range(keypoints):
                px, py, visibility = values[5+index*3:8+index*3]
                if not (0 <= px <= 1 and 0 <= py <= 1 and visibility in (0, 1, 2)):
                    raise ValueError('invalid keypoint')
        except (ValueError, OverflowError) as error:
            raise ValueError('{}:{}: {}'.format(path, number, error)) from error


def check_dataset(dataset_path, task):
    data = load_yaml(dataset_path)
    if 'path' in data or 'download' in data:
        raise ValueError('Dataset YAML must omit path and download for portability')
    names = data.get('names')
    if isinstance(names, list):
        class_count = len(names)
    elif isinstance(names, dict) and set(names) == set(range(len(names))):
        class_count = len(names)
    else:
        raise ValueError('names must be a list or consecutive integer-keyed mapping')
    if not class_count:
        raise ValueError('No classes in dataset YAML')
    keypoints = 0
    if task == 'pose':
        shape = data.get('kpt_shape')
        if not isinstance(shape, list) or len(shape) != 2 or shape[1] != 3 or not isinstance(shape[0], int) or shape[0] < 1:
            raise ValueError('Pose data requires kpt_shape: [N, 3]')
        keypoints = shape[0]
    counts = {}
    for split in SPLITS:
        expected = 'images/' + split
        if data.get(split) != expected:
            raise ValueError('Dataset {} must be {}'.format(split, expected))
        image_dir = within(dataset_path.parent, expected)
        label_dir = within(dataset_path.parent, 'labels/' + split)
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError('Missing image/label directory for {}'.format(split))
        images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError('No images in {}'.format(split))
        stems = set()
        for image in images:
            if image.stem in stems:
                raise ValueError('Duplicate image stem in {}: {}'.format(split, image.stem))
            stems.add(image.stem)
            label = label_dir / (image.stem + '.txt')
            if not label.is_file():
                raise ValueError('Missing label: {}'.format(label))
            check_label(label, class_count, keypoints)
        orphan = {p.stem for p in label_dir.glob('*.txt')} - stems
        if orphan:
            raise ValueError('Labels without images in {}: {}'.format(split, sorted(orphan)[:5]))
        counts[split] = len(images)
    return counts


def preflight(root=ROOT):
    config = load_yaml(within(root, 'config/train.yaml'))
    task = config.get('task')
    if task not in ('detect', 'pose'):
        raise ValueError('task must be detect or pose')
    if config.get('imgsz') != 640:
        raise ValueError('imgsz must be 640')
    if not isinstance(config.get('epochs'), int) or config['epochs'] < 1:
        raise ValueError('epochs must be a positive integer')
    if not isinstance(config.get('test_after_training', True), bool):
        raise ValueError('test_after_training must be true or false')
    data = within(root, config['data'])
    model = within(root, config['model'])
    if model.suffix.lower() != '.pt' or not model.is_file():
        raise ValueError('Local pretrained .pt model missing: {}'.format(model))
    counts = check_dataset(data, task)
    return config, data, model, counts


def _number(value):
    """Convert library/numpy values to finite JSON numbers or None."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _class_metrics(metric):
    """Index Ultralytics Metric results by class id, retaining missing values."""
    result = {}
    if metric is None:
        return result
    indices = getattr(metric, 'ap_class_index', [])
    precision, recall, f1 = (getattr(metric, name, []) for name in ('p', 'r', 'f1'))
    ap50, ap = getattr(metric, 'ap50', []), getattr(metric, 'ap', [])
    for position, class_id in enumerate(indices):
        values = []
        for sequence in (precision, recall, f1, ap50, ap):
            values.append(_number(sequence[position]) if position < len(sequence) else None)
        result[int(class_id)] = values
    return result


def _test_support(dataset_path, class_count):
    """Count annotated instances and images, including empty-label negatives."""
    label_dir = dataset_path.parent / 'labels' / 'test'
    instances, images = [0] * class_count, [0] * class_count
    negatives = total = 0
    for label in sorted(label_dir.glob('*.txt')):
        total += 1
        seen = set()
        for line in label.read_text(encoding='utf-8').splitlines():
            if line.strip():
                class_id = int(float(line.split()[0]))
                instances[class_id] += 1
                seen.add(class_id)
        if not seen:
            negatives += 1
        for class_id in seen:
            images[class_id] += 1
    return total, negatives, instances, images


def _save_class_plot(rows, prefix, destination):
    """Plot per-class precision, recall, AP50 and AP50-95 when available."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fields = ('precision', 'recall', 'ap50', 'map50_95')
    if not any(row.get(prefix + '_' + field) is not None for row in rows for field in fields):
        return False
    height = max(4.0, len(rows) * .34 + 1.5)
    fig, axes = plt.subplots(1, len(fields), figsize=(14, height), sharey=True)
    labels = ['{} ({})'.format(row['name'], row['instances']) for row in rows]
    positions = list(range(len(rows)))
    for axis, field in zip(axes, fields):
        values = [row.get(prefix + '_' + field) for row in rows]
        axis.barh(positions, [v if v is not None else 0 for v in values], color='#3478ae')
        for y, value in enumerate(values):
            axis.text((value if value is not None else 0) + .015, y,
                      '{:.2f}'.format(value) if value is not None else 'n/a', va='center', fontsize=7)
        axis.set_xlim(0, 1.18)
        axis.set_title(field.upper())
        axis.grid(axis='x', alpha=.2)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    fig.suptitle('{} per class (ground-truth instances in parentheses)'.format(prefix))
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return True


def _negative_false_positives(data_path, destination, coverage, threshold=0.25):
    """Count predictions on empty-label images only when coverage is known."""
    image_dir = data_path.parent / 'images' / 'test'
    label_dir = data_path.parent / 'labels' / 'test'
    prediction_dir = destination / 'labels'
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    complete = (coverage.get('seen') == len(images) and coverage.get('dataset_size') == len(images)
                and prediction_dir.is_dir())
    rows = []
    for image in images:
        if (label_dir / (image.stem + '.txt')).read_text(encoding='utf-8').strip():
            continue
        prediction = prediction_dir / (image.stem + '.txt')
        count = None
        if prediction.is_file():
            count = 0
            for line in prediction.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                try:
                    fields = line.split()
                    if len(fields) < 6:
                        raise ValueError('prediction row has fewer than six fields')
                    confidence = float(fields[-1])
                except ValueError as error:
                    raise ValueError('Invalid prediction confidence: {}'.format(prediction)) from error
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError('Invalid prediction confidence: {}'.format(prediction))
                count += int(confidence >= threshold)
        elif complete:
            count = 0
        rows.append({'image': image.name, 'predicted_boxes': count,
                     'has_false_positive': (count > 0) if count is not None else None})
    with (destination / 'negative_false_positives.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'predicted_boxes', 'has_false_positive'])
        writer.writeheader()
        writer.writerows(rows)
    known = sum(row['predicted_boxes'] is not None for row in rows)
    all_known = complete and known == len(rows)
    return {
        'confidence_threshold': threshold,
        'prediction_export_confidence_threshold': 0.001,
        'negative_images': len(rows), 'evaluated_negative_images': known,
        'images_with_false_positives': sum(bool(row['has_false_positive']) for row in rows) if all_known else None,
        'false_positive_boxes': sum(row['predicted_boxes'] for row in rows) if all_known else None,
        'coverage_confirmed': complete, 'validator_seen': coverage.get('seen'),
        'validator_dataset_size': coverage.get('dataset_size'),
    }


def _write_test_tensorboard(summary, rows, destination, writer_class):
    """Write available test metrics under this run's test TensorBoard log."""
    writer = writer_class(log_dir=str(destination))
    try:
        for key, value in summary['metrics'].items():
            number = _number(value)
            if number is not None:
                writer.add_scalar('test/overall/' + key, number, 0)
        for row in rows:
            for key, value in row.items():
                if not (key.startswith('box_') or key.startswith('pose_')):
                    continue
                number = _number(value)
                if number is not None:
                    writer.add_scalar('test/class_{}/{}'.format(row['class_id'], key), number, 0)
        negative = summary['negative_false_positives']
        for key in ('negative_images', 'evaluated_negative_images', 'images_with_false_positives',
                    'false_positive_boxes'):
            number = _number(negative.get(key))
            if number is not None:
                writer.add_scalar('test/negative_fp_025/' + key, number, 0)
        writer.flush()
    finally:
        writer.close()


def write_test_report(metrics, data_path, task, destination, coverage, writer_class=None):
    """Write test-only summary and per-class results from the validator."""
    destination = Path(destination)
    data = load_yaml(data_path)
    names = data['names']
    if isinstance(names, list):
        names = dict(enumerate(names))
    total, negatives, instances, image_counts = _test_support(data_path, len(names))
    box = _class_metrics(getattr(metrics, 'box', None))
    pose = _class_metrics(getattr(metrics, 'pose', None)) if task == 'pose' else {}
    columns = ('precision', 'recall', 'f1', 'ap50', 'map50_95')
    rows = []
    for class_id, name in names.items():
        row = {'class_id': int(class_id), 'name': str(name),
               'instances': instances[class_id], 'images_with_class': image_counts[class_id]}
        for prefix, values in (('box', box), ('pose', pose)):
            if prefix == 'pose' and task != 'pose':
                continue
            result = values.get(class_id, [None] * len(columns))
            row.update({prefix + '_' + field: value for field, value in zip(columns, result)})
        rows.append(row)
    fields = ['class_id', 'name', 'instances', 'images_with_class']
    fields += ['box_' + field for field in columns]
    if task == 'pose':
        fields += ['pose_' + field for field in columns]
    with (destination / 'per_class_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plots = {'box': _save_class_plot(rows, 'box', destination / 'box_per_class.png')}
    if task == 'pose':
        plots['pose'] = _save_class_plot(rows, 'pose', destination / 'pose_per_class.png')
    all_metrics = {}
    for key, value in getattr(metrics, 'results_dict', {}).items():
        if not key.startswith('metrics/'):
            continue
        available = bool(pose) if key.endswith('(P)') else bool(box)
        all_metrics[key] = _number(value) if available else None
    summary = {
        'split': 'test', 'task': task, 'checkpoint': '../weights/best.pt',
        'images': total, 'negative_images': negatives, 'annotated_instances': sum(instances),
        'metrics_available': {'box': bool(box), **({'pose': bool(pose)} if task == 'pose' else {})},
        'metrics': all_metrics,
        'speed_ms_per_image': {key: _number(value) for key, value in getattr(metrics, 'speed', {}).items()},
        'per_class_plots_written': plots,
        'negative_false_positives': _negative_false_positives(data_path, destination, coverage),
        'metric_definitions': {
            'precision_recall_f1': 'Ultralytics 8.2.103 selects a shared confidence index maximizing '
                                   'the smoothed mean F1 curve over the evaluated test set; '
                                   'these values are not fixed-threshold 0.25 results and the '
                                   'test-derived index must not be used to choose deployment parameters.',
            'ap50': 'Average precision integrated over the precision-recall curve at IoU 0.50.',
            'map50_95': 'Mean average precision over IoU thresholds 0.50 through 0.95 in steps of 0.05.',
        },
        'notes': ['Null means no metric available; empty-label images are counted as negatives.',
                  'Test uses the selected validation best checkpoint and is not used to select or tune it.',
                  'Ultralytics 8.2.103 confusion matrices may omit false positives on negative images; '
                  'see negative_false_positives.csv and its explicitly stated confidence threshold.',
                  'When some classes have no predictions, use the ID-indexed class CSV rather than '
                  'relying on the labels of the native precision-recall plot.']
                 + (['Confusion matrices describe object detection boxes, not individual keypoints.'] if task == 'pose' else []),
    }
    (destination / 'test_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2,
        allow_nan=False) + '\n', encoding='utf-8')
    if writer_class is not None:
        _write_test_tensorboard(summary, rows, destination, writer_class)
    return summary


def train(root=ROOT):
    config, data, model, counts = preflight(root)
    print('Counts:', counts, flush=True)
    # Import only after validation; never request a model by downloadable name.
    from ultralytics import YOLO
    from ultralytics.utils.callbacks import tensorboard as tensorboard_callback
    if tensorboard_callback.SummaryWriter is None:
        raise RuntimeError('TensorBoard logging unavailable; install requirements and enable Ultralytics tensorboard integration')
    options = dict(data=str(data), imgsz=640, epochs=config['epochs'], plots=True,
                   batch=config.get('batch', 16), workers=config.get('workers', 4),
                   patience=config.get('patience', 30), seed=config.get('seed', 42),
                   project=str(within(root, config.get('project', 'runs'))),
                   name=config.get('name', 'train'), exist_ok=False)
    if config.get('device') is not None:
        options['device'] = config['device']
    if config['task'] == 'pose':
        options['fliplr'] = 0.0
        options['flipud'] = 0.0
    trained = YOLO(str(model), task=config['task'])
    trained.train(**options)
    if not config.get('test_after_training', True):
        print('Test evaluation disabled by configuration.', flush=True)
        return
    run_dir = Path(trained.trainer.save_dir).resolve()
    try:
        run_dir.relative_to(Path(options['project']).resolve())
    except ValueError:
        raise RuntimeError('Training output directory is outside configured project')
    best = run_dir / 'weights' / 'best.pt'
    if not best.is_file():
        raise RuntimeError('Training finished without a best.pt checkpoint; test was not run')
    test_options = dict(data=str(data), split='test', imgsz=640, batch=config.get('batch', 16), conf=0.001,
                        workers=config.get('workers', 4), project=str(run_dir), name='test',
                        exist_ok=False, plots=True, save_json=True, save_txt=True, save_conf=True)
    if config.get('device') is not None:
        test_options['device'] = config['device']
    print('Evaluating the selected best checkpoint on test...', flush=True)
    evaluator = YOLO(str(best), task=config['task'])
    coverage = {}

    def record_coverage(validator):
        coverage['seen'] = getattr(validator, 'seen', None)
        try:
            coverage['dataset_size'] = len(validator.dataloader.dataset)
        except (AttributeError, TypeError):
            coverage['dataset_size'] = None

    evaluator.add_callback('on_val_end', record_coverage)
    result = evaluator.val(**test_options)
    test_dir = Path(result.save_dir).resolve()
    try:
        test_dir.relative_to(run_dir)
    except ValueError:
        raise RuntimeError('Test output directory is outside the training run')
    write_test_report(result, data, config['task'], test_dir, coverage, tensorboard_callback.SummaryWriter)
    print('Test metrics and visualizations:', test_dir, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'train'))
    args = parser.parse_args(argv)
    try:
        if args.command == 'check':
            _, _, _, counts = preflight()
            print('Data and local model checks passed:', counts)
        else:
            train()
    except (OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError) as error:
        print('ERROR:', error, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
