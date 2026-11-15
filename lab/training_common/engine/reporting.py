"""Detector/Pose metrics, fixed-threshold diagnostics and inspectable predictions."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import importlib.util
from itertools import combinations
import json
import math
import os
from functools import lru_cache
from .runtime import environment, write_json
from .config import read_yaml


@lru_cache(maxsize=1)
def legacy():
    """Reuse existing checked label and per-class plotting utilities."""
    path = Path(__file__).resolve().parents[1] / 'train.py'
    spec = importlib.util.spec_from_file_location('_aec_training_report_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows, fields):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def artifact_manifest(evaluation_dir, task=None, *, postprocess_name=None):
    """Inventory emitted figures without modifying evaluation results or native plots.

    This can also annotate a frozen historical evaluation. Only a new directory
    under its postprocessing/ directory is written. Missing figures get a text
    card, never a reconstructed curve or a substituted metric.
    """
    from . import __version__
    evaluation_dir = Path(evaluation_dir).resolve()
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(evaluation_dir)
    summary_path = evaluation_dir / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
    task = task or summary.get('task')
    if task not in {'detect', 'pose'}:
        raise ValueError('Artifact inventory requires task detect or pose')
    if summary.get('task') is not None and summary['task'] != task:
        raise ValueError('Artifact inventory task differs from recorded evaluation')
    name = postprocess_name or ('artifact_availability_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    if Path(name).name != name or name in {'.', '..'}:
        raise ValueError('Invalid postprocessing directory name')
    destination = evaluation_dir / 'postprocessing' / name
    if not destination.resolve().is_relative_to(evaluation_dir):
        raise ValueError('Postprocessing output must stay inside the evaluation directory')
    destination.mkdir(parents=True, exist_ok=False)
    expected = []
    for kind in ('box', 'pose') if task == 'pose' else ('box',):
        for curve in ('PR', 'F1', 'P', 'R'):
            candidates = [kind.title() + curve + '_curve.png']
            if task == 'detect':
                candidates.append(curve + '_curve.png')
            expected.append((kind + '_' + curve.lower() + '_curve', kind.title() + ' ' + curve + ' curve',
                             'ultralytics_native', candidates))
        expected.append((kind + '_per_class', kind.title() + ' per-class metrics',
                         'engine_per_class', [kind + '_per_class.png']))
    for item in ('confusion_matrix', 'confusion_matrix_normalized', 'diagnostic_counts'):
        expected.append((item, item.replace('_', ' ').capitalize(),
                         'engine_diagnostics' if item == 'diagnostic_counts' else 'ultralytics_native', [item + '.png']))
    entries = []
    for identifier, title, producer, candidates in expected:
        present = [candidate for candidate in candidates if (evaluation_dir / candidate).is_file()]
        entries.append({'id': identifier, 'title': title, 'producer': producer,
                        'expected_files': candidates, 'files': present,
                        'status': 'present' if present else 'missing'})
    missing = [entry for entry in entries if entry['status'] == 'missing']
    explanation = None
    if missing:
        explanation = 'unavailable_artifacts.png'
        cache = destination / 'cache/matplotlib'
        cache.mkdir(parents=True)
        old_cache = os.environ.get('MPLCONFIGDIR')
        os.environ['MPLCONFIGDIR'] = str(cache)
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            figure = Figure(figsize=(10, 3.1 + len(missing) * .30), facecolor='white')
            FigureCanvasAgg(figure)
            lines = ['The recorded evaluation does not contain these figures:', '',
                     *['  - ' + entry['title'] for entry in missing], '',
                     'Figure availability: missing. Cause: not established by this inventory.',
                     'No curve or metric was reconstructed. Original results remain unchanged.',
                     'See summary.json and per_class_metrics.csv for recorded values.']
            figure.text(.05, .94, 'Evaluation artifact availability', fontsize=17, va='top', weight='bold')
            figure.text(.05, .85, '\n'.join(lines), fontsize=11, va='top', linespacing=1.6)
            figure.savefig(destination / explanation, dpi=140)
        finally:
            if old_cache is None:
                os.environ.pop('MPLCONFIGDIR', None)
            else:
                os.environ['MPLCONFIGDIR'] = old_cache
        for entry in missing:
            entry['explanation_file'] = explanation
    source = environment(Path(__file__).parent)
    manifest = {'schema_version': 1, 'task': task, 'split': summary.get('split'),
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'evaluation_directory': '../..',
                'tool': {'module': 'engine.reporting.artifact_manifest', 'revision': 'artifact_availability_v1',
                         'engine_version': __version__, 'git_commit': source.get('git_commit'),
                         'git_worktree_modified': source.get('git_worktree_modified')},
                'artifacts': entries, 'missing_count': len(missing), 'explanation_file': explanation,
                'interpretation': 'File presence only; this inventory does not infer the reason for missing plots. '
                                  'Fixed-threshold diagnostic TP counts do not determine native curve availability. '
                                  'An explanation image is not a metric visualization.'}
    path = destination / 'artifact_manifest.json'
    write_json(path, manifest)
    return path


def metric_report(metrics, data_path, split, config, destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    header = read_yaml(data_path)
    names = header['names']
    if isinstance(names, list): names = dict(enumerate(names))
    labels = sorted((Path(data_path).parent / 'labels' / split).glob('*.txt'))
    instances, image_counts = [0] * len(names), [0] * len(names)
    negatives = 0
    for label in labels:
        classes = [int(float(line.split()[0])) for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
        negatives += int(not classes)
        for cls in classes: instances[cls] += 1
        for cls in set(classes): image_counts[cls] += 1
    box = legacy()._class_metrics(getattr(metrics, 'box', None))
    pose = legacy()._class_metrics(getattr(metrics, 'pose', None)) if config['task'] == 'pose' else {}
    rows = []
    columns = ('precision', 'recall', 'f1', 'ap50', 'map50_95')
    for cls, name in names.items():
        row = {'class_id': cls, 'name': name, 'instances': instances[cls], 'images_with_class': image_counts[cls]}
        for prefix, values in [('box', box)] + ([('pose', pose)] if config['task'] == 'pose' else []):
            found = values.get(cls, [None] * len(columns)) if instances[cls] else [None] * len(columns)
            row.update({prefix + '_' + key: value for key, value in zip(columns, found)})
        rows.append(row)
    write_csv(destination / 'per_class_metrics.csv', rows, list(rows[0]))
    for kind in ('box', 'pose') if config['task'] == 'pose' else ('box',):
        legacy()._save_class_plot(rows, kind, destination / (kind + '_per_class.png'))
    result = {key: legacy()._number(value) for key, value in getattr(metrics, 'results_dict', {}).items()}
    summary = {'split': split, 'task': config['task'], 'model_key': config['model_key'],
               'version': config['version'], 'images': len(labels), 'negative_images': negatives,
               'annotated_instances': sum(instances), 'metrics': result,
               'speed_ms_per_image': {key: legacy()._number(value) for key, value in getattr(metrics, 'speed', {}).items()},
               'source_notes': config.get('source_notes', []),
               'metric_definitions': {
                   'ap': 'Precision-recall integral from the installed Ultralytics validator; see recorded version.',
                   'precision_recall_f1': 'Native curve-derived values, separate from fixed-threshold diagnostics.',
                   'missing_class': 'A class absent from ground truth has null per-class metrics.',
                   'pose': 'Pose AP measures keypoints; confusion matrices concern detection boxes.'},
               'task_chain_evaluation': {'status': 'not_evaluated',
                   'reason': 'This detector/Pose evaluation has no OCR, calibrated readings or cross-frame counting ground truth.'}}
    write_json(destination / 'summary.json', summary)
    return summary, rows


def box_iou(a, b):
    left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right-left) * max(0, bottom-top)
    union = max(0, a[2]-a[0]) * max(0, a[3]-a[1]) + max(0, b[2]-b[0]) * max(0, b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(ground_truth, predictions, threshold):
    """Greedy IoU matches, class-aware, one GT and prediction per pair."""
    candidates = []
    for gi, truth in enumerate(ground_truth):
        for pi, pred in enumerate(predictions):
            iou = box_iou(truth['xyxy'], pred['xyxy'])
            if truth['class_id'] == pred['class_id'] and iou >= threshold:
                candidates.append((iou, gi, pi))
    matches, used_g, used_p = [], set(), set()
    for iou, gi, pi in sorted(candidates, reverse=True):
        if gi not in used_g and pi not in used_p:
            used_g.add(gi); used_p.add(pi); matches.append((gi, pi, iou))
    return matches, [i for i in range(len(ground_truth)) if i not in used_g], [i for i in range(len(predictions)) if i not in used_p]


def reviewed_memberships(row, record):
    """Return this image's reviewed strata and any exclusion reason."""
    review = record.get('reviewed_strata')
    if not isinstance(review, dict) or review.get('supervision') != 'complete':
        return [], 'supervision_not_confirmed'
    state = review.get('label_state')
    truth_count = row['ground_truth']
    if ((state == 'complete_positive' and truth_count <= 0) or
            (state == 'complete_negative' and truth_count != 0) or
            state not in ('complete_positive', 'complete_negative')):
        return [], 'label_state_disagrees_with_staged_label'
    result = []
    for dimension in ('source_group', 'source_kind', 'generation_method', 'label_state',
                      'difficulty', 'conditions', 'target_states'):
        value = review.get(dimension)
        if value is None:
            values = [('unknown', 'missing')]
        elif isinstance(value, list):
            values = [(item, 'confirmed') for item in value] if value else [('none', 'confirmed_empty')]
        else:
            values = [(value, 'confirmed')]
        if any(not isinstance(item, str) or not item.strip() for item, _ in values) or len(values) != len(set(values)):
            raise ValueError('Invalid or duplicated reviewed stratum: ' + dimension)
        result.extend((dimension, item, status) for item, status in values)
    return result, None


def stratify_image_metrics(rows, records):
    """Aggregate fixed-threshold image counts using the staged review snapshot.

    A condition list may put one image into multiple groups. An image without
    complete review is excluded rather than inferred to be a negative.
    """
    if len(rows) != len(records):
        raise ValueError('Each diagnostic image requires one staged record')
    totals, seen, excluded = {}, set(), []
    for row, record in zip(rows, records):
        name = row['image']
        if name in seen or name != Path(record['run_image']).name:
            raise ValueError('Duplicate or mismatched diagnostic image identity')
        seen.add(name)
        memberships, reason = reviewed_memberships(row, record)
        if reason:
            excluded.append({'image': name, 'reason': reason})
            continue
        state = record['reviewed_strata']['label_state']
        for membership in memberships:
            group = totals.setdefault(membership, {
                'images': 0, 'ground_truth': 0, 'predictions': 0, 'tp': 0, 'fp': 0, 'fn': 0,
                'negative_images': 0, 'negative_images_with_false_positives': 0})
            group['images'] += 1
            for key, source in (('ground_truth', 'ground_truth'), ('predictions', 'predictions'),
                                ('tp', 'true_positives'), ('fp', 'false_positives'), ('fn', 'false_negatives')):
                group[key] += row[source]
            group['negative_images'] += int(state == 'complete_negative')
            group['negative_images_with_false_positives'] += int(state == 'complete_negative' and row['false_positives'] > 0)
    result = []
    for (dimension, value, status), group in sorted(totals.items()):
        tp, fp, fn = group['tp'], group['fp'], group['fn']
        result.append({'dimension': dimension, 'value': value, 'value_status': status, **group,
                       'precision': tp/(tp+fp) if tp+fp else None,
                       'recall': tp/(tp+fn) if tp+fn else None})
    return result, excluded


def class_detection_counts(truths, predictions, matches, missed, false, class_count):
    """Use the same class-aware matches as image diagnostics for class counts."""
    if type(class_count) is not int or class_count <= 0:
        raise ValueError('A nonempty class table is required')
    totals = [{'ground_truth': 0, 'predictions': 0, 'tp': 0, 'fp': 0, 'fn': 0}
              for _ in range(class_count)]
    for items, key in ((truths, 'ground_truth'), (predictions, 'predictions')):
        for value in items:
            cls = value['class_id']
            if type(cls) is not int or not 0 <= cls < class_count:
                raise ValueError('Detection class is outside the dataset class table')
            totals[cls][key] += 1
    truth_indices = [gi for gi, _, _ in matches] + list(missed)
    prediction_indices = [pi for _, pi, _ in matches] + list(false)
    if (sorted(truth_indices) != list(range(len(truths))) or
            sorted(prediction_indices) != list(range(len(predictions)))):
        raise ValueError('Detection match partition is inconsistent')
    for gi, pi, _ in matches:
        if truths[gi]['class_id'] != predictions[pi]['class_id']:
            raise ValueError('Matched detections have different class IDs')
        totals[truths[gi]['class_id']]['tp'] += 1
    for gi in missed:
        totals[truths[gi]['class_id']]['fn'] += 1
    for pi in false:
        totals[predictions[pi]['class_id']]['fp'] += 1
    if any(item['tp'] + item['fn'] != item['ground_truth'] or
           item['tp'] + item['fp'] != item['predictions'] for item in totals):
        raise ValueError('Detection match partition is inconsistent')
    return totals


def stratify_class_metrics(rows, records, image_classes, names):
    """Report each class across reviewed image groups, including absent classes."""
    if len(rows) != len(records) or len(rows) != len(image_classes):
        raise ValueError('Each diagnostic image requires staged and class records')
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError('A nonempty ordered class table is required')
    totals, seen = {}, set()
    for row, record, counts in zip(rows, records, image_classes):
        name = row['image']
        if name in seen or name != Path(record['run_image']).name:
            raise ValueError('Duplicate or mismatched diagnostic image identity')
        seen.add(name)
        memberships, reason = reviewed_memberships(row, record)
        if reason:
            continue
        if len(counts) != len(names):
            raise ValueError('Per-image counts differ from class table')
        for dimension, value, status in memberships:
            for class_id, item in enumerate(counts):
                accumulate_class_group(totals, (dimension, value, status, class_id), item)
    result = []
    for (dimension, value, status, class_id), group in sorted(totals.items()):
        result.append({'dimension': dimension, 'value': value, 'value_status': status,
                       'class_id': class_id, 'class_name': names[class_id], **finalize_class_group(group)})
    return result


def accumulate_class_group(totals, key, item):
    group = totals.setdefault(key, {'images': 0, 'images_with_class': 0, 'images_without_class': 0,
        'images_without_class_with_false_positives': 0,
        'ground_truth': 0, 'predictions': 0, 'tp': 0, 'fp': 0, 'fn': 0})
    group['images'] += 1
    group['images_with_class'] += int(item['ground_truth'] > 0)
    group['images_without_class'] += int(item['ground_truth'] == 0)
    group['images_without_class_with_false_positives'] += int(item['ground_truth'] == 0 and item['fp'] > 0)
    for field in ('ground_truth', 'predictions', 'tp', 'fp', 'fn'):
        group[field] += item[field]


def finalize_class_group(group):
    tp, fp, fn = group['tp'], group['fp'], group['fn']
    return {**group, 'precision_denominator': tp+fp, 'recall_denominator': tp+fn,
            'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None}


def joint_domain_condition_metrics(rows, records, image_classes, names):
    """Count one- and two-condition intersections separately for each image domain."""
    if len(rows) != len(records) or len(rows) != len(image_classes):
        raise ValueError('Each diagnostic image requires staged and class records')
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError('A nonempty ordered class table is required')
    totals, seen = {}, set()
    for row, record, counts in zip(rows, records, image_classes):
        name = row['image']
        if name in seen or name != Path(record['run_image']).name:
            raise ValueError('Duplicate or mismatched diagnostic image identity')
        seen.add(name)
        _, reason = reviewed_memberships(row, record)
        if reason:
            continue
        if len(counts) != len(names):
            raise ValueError('Per-image counts differ from class table')
        review = record['reviewed_strata']
        domain = review.get('image_domain')
        if domain is not None and (not isinstance(domain, str) or not domain.strip()):
            raise ValueError('Invalid reviewed image domain')
        domain_value, domain_status = (domain, 'confirmed') if domain else ('unknown', 'missing')
        conditions = review.get('conditions')
        if conditions is None:
            condition_groups = [((), 'missing')]
        elif not isinstance(conditions, list) or len(conditions) > 64:
            raise ValueError('Reviewed conditions must be a list with at most 64 entries')
        elif not conditions:
            condition_groups = [((), 'confirmed_empty')]
        else:
            ordered = sorted(conditions)
            condition_groups = [(tuple([condition]), 'confirmed') for condition in ordered]
            condition_groups.extend((pair, 'confirmed') for pair in combinations(ordered, 2))
        for condition_set, condition_status in condition_groups:
            for class_id, item in enumerate(counts):
                accumulate_class_group(totals, (domain_value, domain_status, condition_set, condition_status, class_id), item)
    output = []
    for (domain, domain_status, condition_set, condition_status, class_id), group in sorted(totals.items()):
        output.append({'image_domain': domain, 'image_domain_status': domain_status,
                       'conditions_all_of': json.dumps(condition_set, ensure_ascii=False),
                       'condition_status': condition_status, 'class_id': class_id, 'class_name': names[class_id],
                       **finalize_class_group(group)})
    return output


def diagnostics(model, data_path, split, config, destination):
    """Save every prediction and compute diagnostics at a declared fixed threshold."""
    destination = Path(destination)
    prediction_dir = destination / 'predictions'
    for directory in ('labels', 'previews', 'failures'):
        (prediction_dir / directory).mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(data_path).parent
    source_records = json.loads((dataset_dir / 'samples.json').read_text(encoding='utf-8'))
    lookup = {Path(record['run_image']).name: record for record in source_records if record['split'] == split}
    v4 = str(config.get('version', '')).lower() == 'v4'
    names = None
    if v4:
        names = read_yaml(data_path).get('names')
        if isinstance(names, dict) and set(names) == set(range(len(names))):
            names = [names[index] for index in range(len(names))]
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
            raise ValueError('Staged data requires an ordered class table')
    threshold = config.get('diagnostic_conf', .25)
    match_iou = config.get('match_iou', .5)
    options = {'source': str(dataset_dir / 'images' / split), 'stream': True,
               'imgsz': config['imgsz'], 'conf': threshold, 'iou': config.get('nms_iou', .7),
               'max_det': config.get('max_det', 300), 'verbose': False, 'save': False,
               'device': config.get('device', '0'), 'save_txt': False, 'save_crop': False, 'show': False}
    rows, keypoint_rows, groups, diagnostic_records, image_classes = [], [], {}, [], []
    previews = failures = 0
    with (prediction_dir / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for result in model.predict(**options):
            name = Path(result.path).name
            record = lookup[name]
            height, width = result.orig_shape
            truths = []
            for line in (dataset_dir / 'labels' / split / (Path(name).stem + '.txt')).read_text(encoding='utf-8').splitlines():
                if not line.strip(): continue
                values = list(map(float, line.split()))
                cls, x, y, w, h = values[:5]
                truths.append({'class_id': int(cls), 'xyxy': [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height],
                               'keypoints': [values[i:i+3] for i in range(5, len(values), 3)]})
            predictions = []
            boxes = result.boxes
            for index in range(len(boxes)):
                prediction = {'class_id': int(boxes.cls[index].item()), 'confidence': float(boxes.conf[index].item()),
                              'xyxy': boxes.xyxy[index].cpu().tolist()}
                if result.keypoints is not None:
                    prediction['keypoints'] = result.keypoints.data[index].cpu().tolist()
                predictions.append(prediction)
            matched, missed, false = match_detections(truths, predictions, match_iou)
            if v4:
                image_classes.append(class_detection_counts(truths, predictions, matched, missed, false, len(names)))
            confused = sum(any(truths[gi]['class_id'] != predictions[pi]['class_id'] and
                              box_iou(truths[gi]['xyxy'], predictions[pi]['xyxy']) >= match_iou for pi in false) for gi in missed)
            row = {'image': name, 'source_image': record['image'], 'source_group': record['source_group'],
                   'ground_truth': len(truths), 'predictions': len(predictions), 'true_positives': len(matched),
                   'false_positives': len(false), 'false_negatives': len(missed), 'class_confusion_candidates': confused,
                   'negative_image': not truths, 'negative_false_positive': not truths and bool(predictions)}
            rows.append(row)
            diagnostic_records.append(record)
            group = groups.setdefault(record['source_group'], {'images': 0, 'tp': 0, 'fp': 0, 'fn': 0})
            group['images'] += 1; group['tp'] += len(matched); group['fp'] += len(false); group['fn'] += len(missed)
            for gi, pi, iou in matched:
                predicted_points = predictions[pi].get('keypoints', [])
                for index, truth_point in enumerate(truths[gi]['keypoints']):
                    if truth_point[2] <= 0 or index >= len(predicted_points): continue
                    point = predicted_points[index]
                    error = math.hypot(point[0] - truth_point[0]*width, point[1] - truth_point[1]*height)
                    keypoint_rows.append({'image': name, 'keypoint_index': index, 'visibility': truth_point[2],
                                          'error_pixels': error, 'error_image_diagonal_fraction': error/math.hypot(width, height)})
            out = {'image': 'data/images/' + split + '/' + name, 'source_image': record['image'],
                   'shape_hw': [height, width], 'ground_truth': truths, 'predictions': predictions,
                   'matches': matched, 'missed_gt_indices': missed, 'false_prediction_indices': false}
            stream.write(json.dumps(out, ensure_ascii=False, allow_nan=False) + '\n')
            label_path = prediction_dir / 'labels' / (Path(name).stem + '.txt')
            result.save_txt(str(label_path), save_conf=True)
            if not label_path.exists(): label_path.write_text('', encoding='utf-8')
            if previews < config.get('preview_limit', 64):
                result.save(filename=str(prediction_dir / 'previews' / name)); previews += 1
            if (missed or false) and failures < config.get('preview_limit', 64):
                result.save(filename=str(prediction_dir / 'failures' / name)); failures += 1
    if len(rows) != len(lookup):
        raise RuntimeError(f'Prediction coverage incomplete: {len(rows)}/{len(lookup)}')
    write_csv(destination / 'per_image.csv', rows, list(rows[0]))
    write_csv(destination / 'negative_false_positives.csv', [r for r in rows if r['negative_image']], list(rows[0]))
    write_csv(destination / 'keypoint_errors.csv', keypoint_rows,
              ['image', 'keypoint_index', 'visibility', 'error_pixels', 'error_image_diagonal_fraction'])
    group_rows = []
    for source, group in groups.items():
        tp, fp, fn = group['tp'], group['fp'], group['fn']
        group_rows.append({'source_group': source, **group,
                           'precision': tp/(tp+fp) if tp+fp else None,
                           'recall': tp/(tp+fn) if tp+fn else None})
    write_csv(destination / 'by_source.csv', group_rows, ['source_group', 'images', 'tp', 'fp', 'fn', 'precision', 'recall'])
    strata_summary = None
    if v4:
        strata, excluded = stratify_image_metrics(rows, diagnostic_records)
        per_class = stratify_class_metrics(rows, diagnostic_records, image_classes, names)
        joint = joint_domain_condition_metrics(rows, diagnostic_records, image_classes, names)
        fields = ['dimension', 'value', 'value_status', 'images', 'ground_truth', 'predictions', 'tp', 'fp', 'fn',
                  'negative_images', 'negative_images_with_false_positives', 'precision', 'recall']
        write_csv(destination / 'by_reviewed_stratum.csv', strata, fields)
        class_fields = ['dimension', 'value', 'value_status', 'class_id', 'class_name', 'images',
                        'images_with_class', 'images_without_class', 'images_without_class_with_false_positives',
                        'ground_truth', 'predictions', 'tp', 'fp', 'fn',
                        'precision_denominator', 'recall_denominator', 'precision', 'recall']
        write_csv(destination / 'by_reviewed_stratum_class.csv', per_class, class_fields)
        joint_fields = ['image_domain', 'image_domain_status', 'conditions_all_of', 'condition_status',
                        'class_id', 'class_name', 'images', 'images_with_class', 'images_without_class',
                        'images_without_class_with_false_positives', 'ground_truth', 'predictions', 'tp', 'fp', 'fn',
                        'precision_denominator', 'recall_denominator', 'precision', 'recall']
        write_csv(destination / 'by_domain_conditions_class.csv', joint, joint_fields)
        strata_summary = {'included_images': len(rows)-len(excluded), 'excluded_images': excluded,
                          'scope': ('Fixed-threshold box counts on completely reviewed staged labels; '
                                    'stratum attributes are recorded review claims, not independently verified truth. '
                                    'Multi-valued groups overlap; missing attributes are unknown. '
                                    'Each group includes all objects in its images, not only objects with that attribute. '
                                    'Class recall uses class-aware matched objects in those images; it is not per-object '
                                    'condition or state recall. No-GT class recall and no-prediction precision are null. '
                                    'Domain-condition rows include each confirmed single condition and each pair; '
                                    'groups overlap. An empty confirmed condition list records no listed tags, '
                                    'not proof that all conditions are absent.')}
        write_json(destination / 'reviewed_strata.json', strata_summary)
    summary = {'confidence_threshold': threshold, 'match_iou': match_iou,
               'nms_iou': config.get('nms_iou', .7), 'images': len(rows),
               'true_positives': sum(r['true_positives'] for r in rows),
               'false_positives': sum(r['false_positives'] for r in rows),
               'false_negatives': sum(r['false_negatives'] for r in rows),
               'negative_images': sum(r['negative_image'] for r in rows),
               'negative_images_with_false_positives': sum(r['negative_false_positive'] for r in rows),
               'keypoint_pairs': len(keypoint_rows),
               'mean_keypoint_pixel_error': sum(r['error_pixels'] for r in keypoint_rows)/len(keypoint_rows) if keypoint_rows else None,
               'keypoint_scope': 'Visible labeled points on class/IoU-matched detections only; missed objects counted separately.',
               'preview_images': previews, 'failure_previews': failures}
    if strata_summary is not None:
        summary['reviewed_strata'] = strata_summary
    write_json(destination / 'diagnostics.json', summary)
    _diagnostic_plots(destination, rows, keypoint_rows)
    return summary


def _diagnostic_plots(destination, rows, keypoints):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(['TP', 'FP', 'FN'], [sum(row[field] for row in rows) for field in ('true_positives', 'false_positives', 'false_negatives')])
    ax.set_title('Fixed-threshold box diagnostics'); ax.set_ylabel('Instances')
    fig.tight_layout(); fig.savefig(destination / 'diagnostic_counts.png', dpi=150); plt.close(fig)
    if keypoints:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist([row['error_pixels'] for row in keypoints], bins=30)
        ax.set_xlabel('Keypoint error (pixels, matched visible points)'); ax.set_ylabel('Points')
        fig.tight_layout(); fig.savefig(destination / 'keypoint_error_histogram.png', dpi=150); plt.close(fig)


def tensorboard_report(run_dir, name, summary, rows, diagnostic):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(Path(run_dir) / 'tensorboard' / name))
    try:
        for key, value in summary['metrics'].items():
            if isinstance(value, (int, float)) and math.isfinite(value): writer.add_scalar(key, value, 0)
        for row in rows:
            for key, value in row.items():
                if (key.startswith('box_') or key.startswith('pose_')) and value is not None:
                    writer.add_scalar(f'class_{row["class_id"]}/{key}', value, 0)
        for key, value in diagnostic.items():
            if isinstance(value, (int, float)) and math.isfinite(value): writer.add_scalar('diagnostics/' + key, value, 0)
        writer.flush()
    finally:
        writer.close()
