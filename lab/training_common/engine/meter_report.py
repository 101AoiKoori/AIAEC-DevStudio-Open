"""Eight named meter points and separate projected needle precision diagnostics."""
import json
import math
from pathlib import Path

from .reporting import write_csv
from .runtime import write_json

POINTS = ('temperature_pivot', 'temperature_tip', 'humidity_pivot', 'humidity_tip',
          'temperature_min', 'temperature_max', 'humidity_min', 'humidity_max')
NEEDLES = {'temperature': (0, 1), 'humidity': (2, 3)}
POINT_CONFIDENCE = 0.5


def _stats(values):
    values = sorted(values)
    def quantile(q):
        position = (len(values) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return {'count': len(values), 'mean': sum(values) / len(values) if values else None,
            **{name: quantile(q) if values else None for name, q in
               (('p50', .5), ('p90', .9), ('p95', .95))},
            'max': max(values) if values else None}


def _bucket():
    return {'eligible': 0, 'measured': 0, 'box_missed': 0, 'point_unavailable': 0,
            'degenerate_truth': 0, 'degenerate_prediction': 0, 'angles': [], 'tip_errors': []}


def _needle_summary(bucket):
    result = {key: value for key, value in bucket.items() if key not in ('angles', 'tip_errors')}
    result['unmeasured'] = bucket['eligible'] - bucket['measured']
    result['coverage'] = bucket['measured'] / bucket['eligible'] if bucket['eligible'] else None
    result['angle_error_degrees'] = _stats(bucket['angles'])
    result['tip_error_pixels'] = _stats(bucket['tip_errors'])
    result['within_angle_degrees'] = {
        str(limit): {'count': sum(value <= limit for value in bucket['angles']),
                     'fraction_of_measured': (sum(value <= limit for value in bucket['angles']) /
                                              bucket['measured'] if bucket['measured'] else None),
                     'fraction_of_eligible': (sum(value <= limit for value in bucket['angles']) /
                                             bucket['eligible'] if bucket['eligible'] else None)}
        for limit in (2, 5, 10)}
    return result


def _points(value, *, truth):
    if not truth and not value:
        return []
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError('Meter records require eight ordered keypoints')
    for point in value:
        if (not isinstance(point, list) or len(point) != 3 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
            raise ValueError('Invalid meter point triplet')
        if truth and (point[2] not in (0, 1, 2) or not all(0 <= v <= 1 for v in point[:2])):
            raise ValueError('Truth points require normalized XY and visibility 0/1/2')
        if not truth and not 0 <= point[2] <= 1:
            raise ValueError('Prediction point confidence must be in [0, 1]')
    return value


def meter_pose_report(rows):
    """Consume saved diagnostic rows; truth XY is normalized, predictions are pixels."""
    points = [dict(index=i, name=name, eligible=0, measured=0, box_missed=0,
                   point_unavailable=0, visibility_1=0, visibility_2=0, errors=[], normalized=[])
              for i, name in enumerate(POINTS)]
    needles = {name: _bucket() for name in NEEDLES}
    length_groups = {name: {key: _bucket() for key in ('under10', '10to20', '20plus')}
                     for name in NEEDLES}
    size_groups = {key: {name: _bucket() for name in NEEDLES}
                   for key in ('under48', '48to80', '80plus')}
    seen = set()
    counts = dict(images=0, positive_images=0, negative_images=0, negative_images_with_false_positives=0,
                  ground_truth=0, matched_boxes=0, missed_boxes=0, false_predictions=0)
    details = []
    for row in rows:
        image = row['image']
        if image in seen:
            raise ValueError('Duplicate meter evaluation image')
        seen.add(image)
        height, width = row['shape_hw']
        if any(type(v) is not int or v <= 0 for v in (height, width)):
            raise ValueError('Invalid image size')
        truths, predictions = row['ground_truth'], row['predictions']
        matches = row['matches']
        paired = {pair[0]: pair[1] for pair in matches}
        if (len(paired) != len(matches) or len(set(paired.values())) != len(matches) or
                any(type(g) is not int or type(p) is not int or
                    not 0 <= g < len(truths) or not 0 <= p < len(predictions)
                    for g, p in paired.items())):
            raise ValueError('Invalid or repeated meter match')
        missed, false = row['missed_gt_indices'], row['false_prediction_indices']
        if (len(missed) != len(set(missed)) or len(false) != len(set(false)) or
                set(missed) != set(range(len(truths))) - set(paired) or
                set(false) != set(range(len(predictions))) - set(paired.values())):
            raise ValueError('Incomplete meter match partition')
        counts['images'] += 1
        counts['positive_images' if truths else 'negative_images'] += 1
        counts['negative_images_with_false_positives'] += int(not truths and bool(false))
        counts['ground_truth'] += len(truths)
        counts['matched_boxes'] += len(matches)
        counts['missed_boxes'] += len(missed)
        counts['false_predictions'] += len(false)
        for prediction in predictions:
            if prediction['class_id'] != 0:
                raise ValueError('Meter report requires the single meter class')
            _points(prediction.get('keypoints', []), truth=False)
        for gi, truth in enumerate(truths):
            if truth['class_id'] != 0:
                raise ValueError('Meter report requires the single meter class')
            gt = _points(truth['keypoints'], truth=True)
            x1, y1, x2, y2 = truth['xyxy']
            side = min(x2 - x1, y2 - y1)
            if not math.isfinite(side) or side <= 0:
                raise ValueError('Invalid truth dial box')
            gt_xy = [(p[0] * width, p[1] * height) for p in gt]
            pred = predictions[paired[gi]].get('keypoints', []) if gi in paired else []
            usable = [bool(pred) and pred[i][2] >= POINT_CONFIDENCE for i in range(8)]
            for i, point in enumerate(gt):
                if point[2] == 0:
                    continue
                item = points[i]
                item['eligible'] += 1
                item['visibility_' + str(int(point[2]))] += 1
                if gi not in paired:
                    item['box_missed'] += 1
                elif not usable[i]:
                    item['point_unavailable'] += 1
                else:
                    error = math.dist(gt_xy[i], pred[i][:2])
                    item['measured'] += 1
                    item['errors'].append(error)
                    item['normalized'].append(error / side)
            size = 'under48' if side < 48 else '48to80' if side < 80 else '80plus'
            for name, (pivot, tip) in NEEDLES.items():
                if not gt[pivot][2] or not gt[tip][2]:
                    continue
                length = math.dist(gt_xy[pivot], gt_xy[tip])
                group = 'under10' if length < 10 else '10to20' if length < 20 else '20plus'
                angle = tip_error = None
                if length <= 1e-9:
                    state = 'degenerate_truth'
                elif gi not in paired:
                    state = 'box_missed'
                elif not usable[pivot] or not usable[tip]:
                    state = 'point_unavailable'
                elif math.dist(pred[pivot][:2], pred[tip][:2]) <= 1e-9:
                    state = 'degenerate_prediction'
                else:
                    a = math.atan2(gt_xy[tip][1] - gt_xy[pivot][1], gt_xy[tip][0] - gt_xy[pivot][0])
                    b = math.atan2(pred[tip][1] - pred[pivot][1], pred[tip][0] - pred[pivot][0])
                    angle = abs(math.degrees(math.atan2(math.sin(a-b), math.cos(a-b))))
                    tip_error = math.dist(gt_xy[tip], pred[tip][:2])
                    state = 'measured'
                for bucket in (needles[name], length_groups[name][group], size_groups[size][name]):
                    bucket['eligible'] += 1
                    bucket[state] += 1
                    if state == 'measured':
                        bucket['angles'].append(angle)
                        bucket['tip_errors'].append(tip_error)
                details.append(dict(image=image, gt_index=gi, needle=name, state=state,
                                    dial_short_side_pixels=side, truth_needle_length_pixels=length,
                                    angle_error_degrees=angle, tip_error_pixels=tip_error))
    if not seen:
        raise ValueError('No meter diagnostic rows')
    point_results = []
    for item in points:
        errors, normalized = item.pop('errors'), item.pop('normalized')
        point_results.append({**item, 'unmeasured': item['eligible'] - item['measured'],
                              'error_pixels': _stats(errors), 'error_dial_short_side_fraction': _stats(normalized)})
    return {'point_confidence_threshold': POINT_CONFIDENCE, 'counts': counts,
            'negative_image_false_positive_rate': (counts['negative_images_with_false_positives'] /
                                                   counts['negative_images'] if counts['negative_images'] else None),
            'points': point_results, 'needles': {k: _needle_summary(v) for k, v in needles.items()},
            'by_needle_length_pixels': {k: {g: _needle_summary(v) for g, v in groups.items()}
                                        for k, groups in length_groups.items()},
            'by_dial_short_side_pixels': {k: {g: _needle_summary(v) for g, v in groups.items()}
                                          for k, groups in size_groups.items()},
            'scope': ('Class/IoU matches and saved fixed-threshold predictions. Truth visibility 1 and 2 '
                      'are supervised; 0 is unknown. Predicted point confidence must be >=0.5. '
                      'Errors exclude unmeasured entries; eligible counts retain missed boxes and unavailable points. '
                      'Angles compare projected 2D pivot-to-tip directions, not perspective-corrected physical '
                      'readings. Angle thresholds are diagnostics, not task acceptance thresholds.')}, details


def write_meter_report(evaluation):
    """Write meter_report.json and per-needle errors from a completed Pose evaluation."""
    evaluation = Path(evaluation)
    summary = json.loads((evaluation / 'summary.json').read_text(encoding='utf-8'))
    if summary.get('model_key') != 'meter' or summary.get('task') != 'pose':
        raise ValueError('Evaluation is not meter Pose')
    with (evaluation / 'predictions/predictions.jsonl').open(encoding='utf-8') as stream:
        report, rows = meter_pose_report(json.loads(line) for line in stream if line.strip())
    if summary['images'] != report['counts']['images']:
        raise ValueError('Meter prediction coverage differs from evaluation summary')
    report.update(split=summary['split'], diagnostic_config=summary.get('diagnostics', {}),
                  keypoint_order=list(POINTS))
    write_csv(evaluation / 'meter_needle_errors.csv', rows,
              ['image', 'gt_index', 'needle', 'state', 'dial_short_side_pixels',
               'truth_needle_length_pixels', 'angle_error_degrees', 'tip_error_pixels'])
    path = evaluation / 'meter_report.json'
    write_json(path, report)
    return path
