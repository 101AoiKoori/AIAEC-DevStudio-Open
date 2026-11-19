"""Whole-bin localization and lid-state diagnostics from recorded predictions."""
from collections import defaultdict
import json
import math
from pathlib import Path

from .reporting import box_iou


STATES = ('open', 'closed')
AXES = ('view', 'occlusion', 'tone', 'position', 'size')
OUTCOMES = ('correct_state', 'opposite_state', 'localization_miss')


def checked_box(item):
    values = item.get('xyxy')
    if (item.get('class_id') not in (0, 1) or type(item['class_id']) is not int
            or not isinstance(values, list) or len(values) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
            or values[0] >= values[2] or values[1] >= values[3]):
        raise ValueError('Invalid whole-bin class or box')
    return values


def match_whole_bins(truths, predictions, threshold):
    """Match highest-IoU pairs greedily, independent of predicted lid state."""
    if not 0 < threshold <= 1:
        raise ValueError('Invalid whole-bin IoU threshold')
    for item in truths + predictions:
        checked_box(item)
    pairs = []
    for gi, truth in enumerate(truths):
        for pi, prediction in enumerate(predictions):
            overlap = box_iou(truth['xyxy'], prediction['xyxy'])
            if overlap >= threshold:
                pairs.append((overlap, gi, pi))
    matched, seen_gt, seen_prediction = [], set(), set()
    for overlap, gi, pi in sorted(pairs, key=lambda row: (-row[0], row[1], row[2])):
        if gi not in seen_gt and pi not in seen_prediction:
            matched.append((gi, pi, overlap))
            seen_gt.add(gi)
            seen_prediction.add(pi)
    missed = [index for index in range(len(truths)) if index not in seen_gt]
    extra = [index for index in range(len(predictions)) if index not in seen_prediction]
    duplicates = [index for index in extra if any(
        box_iou(predictions[index]['xyxy'], truth['xyxy']) >= threshold for truth in truths)]
    return matched, missed, extra, duplicates


def manifest_rows(path, root, split):
    """Read exported bin metadata and keep the declaring manifest's path base."""
    document = json.loads(path.read_text(encoding='utf-8-sig'))
    if document.get('task', document.get('model_key')) != 'waste_bin_state':
        raise ValueError('Dataset manifest is not waste_bin_state')
    names = document.get('class_names')
    if names is not None and names != list(STATES):
        raise ValueError('Bin-state dataset class order differs from predictions')
    if 'records' in document:
        for row in document['records']:
            if row.get('split', document.get('split')) == split:
                yield path, row
        return
    part = document.get('splits', {}).get(split)
    if not isinstance(part, dict) or not isinstance(part.get('manifest'), str):
        raise ValueError('Dataset manifest has no requested split records')
    child = (path.parent / part['manifest']).resolve()
    if not child.is_relative_to(root):
        raise ValueError('Dataset split manifest escapes workspace or package')
    yield from manifest_rows(child, root, split)


def image_aliases(path, record, root):
    values = set()
    for field in ('image', 'source_image', 'workspace_image'):
        value = record.get(field)
        if not isinstance(value, str) or not value:
            continue
        value = value.replace('\\', '/')
        values.add(value)
        relative = (path.parent / value).resolve()
        if relative.is_relative_to(root):
            values.add(relative.relative_to(root).as_posix())
    return values


def verified_metadata(prediction_row, record):
    """Bind every color and state to the original coordinates and GT row order."""
    truths, shape = prediction_row['ground_truth'], prediction_row['shape_hw']
    size = record.get('image_size')
    bins = record.get('bins')
    if (not isinstance(shape, list) or len(shape) != 2 or not isinstance(size, list)
            or size != [shape[1], shape[0]] or not isinstance(bins, list) or len(bins) != len(truths)):
        raise ValueError('Bin metadata dimensions or instance count differ from evaluated ground truth')
    for truth, item in zip(truths, bins):
        checked_box(truth)
        color = item.get('color', 'unknown')
        original = item.get('bbox_xyxy')
        if (not isinstance(color, str) or not color.strip() or item.get('state') != STATES[truth['class_id']]
                or not isinstance(original, list) or len(original) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in original)
                or max(abs(a-b) for a, b in zip(original, truth['xyxy'])) > 1):
            raise ValueError('Bin color/state metadata does not bind to the evaluated GT box')
    return bins


def counts():
    return {'instances': 0, **{key: 0 for key in OUTCOMES}}


def rates(group):
    instances = group['instances']
    located = group['correct_state'] + group['opposite_state']
    return {**group, 'located': located,
            'localization_recall': located / instances if instances else None,
            'correct_state_recall': group['correct_state'] / instances if instances else None,
            'state_accuracy_when_located': group['correct_state'] / located if located else None}


def bin_state_report(rows, metadata, threshold, *, imgsz=640):
    """Create exclusive per-bin outcomes while retaining duplicate predictions."""
    if not rows or type(imgsz) is not int or imgsz < 1:
        raise ValueError('Expected nonempty predictions and a positive input size')
    totals = counts()
    groups = defaultdict(lambda: defaultdict(counts))
    images = []
    seen = set()
    extras = duplicates = negatives = negative_fp_images = exactly_correct = 0
    for row in rows:
        identity = row['source_image'].replace('\\', '/')
        if identity in seen:
            raise ValueError('Duplicate evaluated source image')
        seen.add(identity)
        choices = metadata.get(identity, [])
        if len(choices) != 1:
            raise ValueError('Evaluated image has missing or ambiguous bin metadata: ' + identity)
        record = choices[0]
        bins = verified_metadata(row, record)
        truths, predictions = row['ground_truth'], row['predictions']
        matched, missed, extra, duplicate = match_whole_bins(truths, predictions, threshold)
        located = {gi: (pi, overlap) for gi, pi, overlap in matched}
        item_outcomes = []
        for gi, (truth, item) in enumerate(zip(truths, bins)):
            pair = located.get(gi)
            outcome = ('localization_miss' if pair is None else 'correct_state'
                       if truth['class_id'] == predictions[pair[0]]['class_id'] else 'opposite_state')
            height = (truth['xyxy'][3] - truth['xyxy'][1]) * imgsz / max(row['shape_hw'])
            tier = 'under_75' if height < 75 else '75_to_under_90' if height < 90 else '90_to_under_110' if height < 110 else '110_or_more'
            color = item.get('color', 'unknown')
            dimensions = {'color': color, 'state': item['state'],
                          'source_kind': record.get('source_kind') or 'unknown',
                          'parent_group': record.get('physical_parent_group') or record.get('parent_group') or record.get('source_id') or 'unknown',
                          'whole_bin_height_px': tier,
                          **{axis: (record.get('allocation_axes') or {}).get(axis, 'unknown') for axis in AXES}}
            totals['instances'] += 1
            totals[outcome] += 1
            for dimension, value in dimensions.items():
                group = groups[dimension][str(value)]
                group['instances'] += 1
                group[outcome] += 1
            item_outcomes.append({'gt_index': gi, 'color': color, 'true_state': item['state'],
                                  'outcome': outcome, 'prediction_index': pair[0] if pair else None,
                                  'iou': pair[1] if pair else None, 'whole_bin_height_at_imgsz': height})
        extras += len(extra)
        duplicates += len(duplicate)
        negatives += not bool(truths)
        negative_fp_images += not truths and bool(predictions)
        exactly_correct += not extra and all(item['outcome'] == 'correct_state' for item in item_outcomes)
        images.append({'source_image': identity, 'bins': item_outcomes,
                       'unmatched_prediction_indices': extra, 'duplicate_prediction_indices': duplicate,
                       'localization_miss_gt_indices': missed})
    return {'images': len(rows), 'match_iou': threshold, 'imgsz': imgsz, 'totals': rates(totals),
            'unmatched_predictions': extras, 'duplicate_predictions': duplicates,
            'other_unmatched_predictions': extras - duplicates,
            'negative_images': negatives, 'negative_images_with_predictions': negative_fp_images,
            'exactly_correct_images': exactly_correct,
            'by_dimension': {key: {value: rates(group) for value, group in sorted(items.items())}
                             for key, items in groups.items()}, 'per_image': images,
            'scope': 'Fixed-threshold whole-bin matches use highest-IoU one-to-one pairing independent of state. '
                     'Each GT has exactly one correct-state, opposite-state or localization-miss outcome. '
                     'Unmatched predictions overlapping any GT at the threshold are duplicates. '
                     'Other unmatched predictions include background errors and poorly localized boxes. '
                     'Standard detector metrics are retained in the existing evaluation files. '
                     'Dimensions come from the dataset manifest; unspecified attributes remain unknown.'}


def write_bin_state_report(evaluation_dir, dataset_manifest, *, root):
    """Write whole-bin/state diagnostics for an existing saved evaluation."""
    root = Path(root).resolve()
    evaluation_dir, dataset_manifest = Path(evaluation_dir).resolve(), Path(dataset_manifest).resolve()
    if not evaluation_dir.is_relative_to(root) or not dataset_manifest.is_relative_to(root):
        raise ValueError('Report inputs must be inside the workspace or package')
    summary = json.loads((evaluation_dir / 'summary.json').read_text(encoding='utf-8'))
    if summary.get('model_key') != 'waste_bin_state':
        raise ValueError('Evaluation is not waste_bin_state')
    split = summary['split']
    metadata = defaultdict(list)
    records = list(manifest_rows(dataset_manifest, root, split))
    for source, record in records:
        for alias in image_aliases(source, record, root):
            metadata[alias].append(record)
    predictions = evaluation_dir / 'predictions/predictions.jsonl'
    rows = [json.loads(line) for line in predictions.read_text(encoding='utf-8').splitlines() if line.strip()]
    config = summary.get('evaluation_config') or {}
    report = bin_state_report(rows, metadata, float(summary['diagnostics']['match_iou']),
                              imgsz=config.get('imgsz', 640))
    report.update(split=split, dataset_manifest=dataset_manifest.relative_to(root).as_posix(),
                  metadata_records=len(records), evaluated_metadata_records=len(rows),
                  confidence_threshold=summary['diagnostics'].get('confidence_threshold'),
                  nms_iou=summary['diagnostics'].get('nms_iou'),
                  evaluation_summary=(evaluation_dir / 'summary.json').relative_to(root).as_posix(),
                  prediction_source=predictions.relative_to(root).as_posix())
    path = evaluation_dir / 'bin_state_report.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != report:
            raise ValueError('Existing bin-state report differs from predictions or dataset metadata')
    else:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path
