"""Run frozen V3 inference and score independently reviewed field candidates."""

import argparse
from collections import Counter, defaultdict
import csv
import filecmp
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'lab/training_common'))

SCORED = {'visual_candidate', 'visual_negative'}
REVIEW_STATUSES = {'visual_candidate', 'visual_negative', 'partial', 'partial_unlabeled', 'unknown'}
STATUS_ALIASES = {'partial_unlabeled': 'partial'}
CONFIDENCE_FLOOR = .01
MATCH_IOU = .5
CAPTURE_IMAGE_ROOTS = ('datasets/raw', 'Archive/Assets/captures')
SELECTION_IMAGE_ROOTS = (*CAPTURE_IMAGE_ROOTS, 'Archive/Assets/CarRealPicture', 'datasets/annotated')
SOURCE_COPY_MANIFEST = 'datasets/source_copy_manifest.jsonl'


def workspace_path(value, *parents):
    path = (ROOT / value).resolve()
    for parent in parents:
        base = (ROOT / parent).resolve()
        if base != ROOT and base.is_relative_to(ROOT) and path != base and path.is_relative_to(base):
            return path
    raise ValueError(f'路径须在 {" 或 ".join(parents)} 下：{value}')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def source_images(batch):
    photos = {
        workspace_path(row['current_path'], *CAPTURE_IMAGE_ROOTS): 'vehicle_camera'
        for row in csv_rows(batch / 'source_inventory.csv') if row['media_type'] == 'image'
    }
    extract = batch / 'extract_v1'
    frames = {
        workspace_path(extract / row['frame'], 'datasets/annotated'): Path(row['source_video']).stem
        for row in csv_rows(extract / 'frames.csv')
    }
    sources = photos | frames
    if not sources or not all(path.is_file() for path in sources):
        raise ValueError('原图或抽帧缺失')
    return {path.relative_to(ROOT).as_posix(): group for path, group in sources.items()}


def selected_images(selection):
    manifest = read_json(selection)
    group = manifest['batch_id']
    if not isinstance(group, str) or not group:
        raise ValueError('选择清单缺少有效 batch_id')
    sources = {}
    for row in manifest['items']:
        raw = Path(row['path'])
        if raw.is_absolute():
            raise ValueError(f'选择原图须位于工作空间数据或素材区：{raw}')
        path = workspace_path(raw, *SELECTION_IMAGE_ROOTS)
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.ppm'}:
            raise ValueError(f'选择原图缺失或格式不支持：{raw}')
        relative = path.relative_to(ROOT).as_posix()
        if relative in sources:
            raise ValueError(f'选择清单重复原图：{relative}')
        sources[relative] = group
    if not sources:
        raise ValueError('选择清单为空')
    return sources


def checkpoint_specs(path):
    specs = read_json(path)
    for key, spec in specs.items():
        if not key.isidentifier() or spec['task'] not in {'detect', 'pose'}:
            raise ValueError(f'无效模型键或类型：{key}')
        weight = workspace_path(spec['weights'], 'runs')
        if weight.name != 'best.pt' or not weight.is_file():
            raise ValueError(f'{key} 缺少冻结 best.pt')
    if not specs:
        raise ValueError('检查点列表为空')
    return specs


def audit_label_files(label_files):
    """Validate current field-label records and derive the review-summary counts."""

    records = []
    seen = set()
    errors = []
    for path in label_files:
        for index, row in enumerate(read_json(path)):
            location = f'{path.relative_to(ROOT).as_posix()}[{index}]'
            if not isinstance(row, dict):
                errors.append(f'{location}: 记录必须是对象')
                continue
            model_key = row.get('model_key')
            image = row.get('image')
            status = row.get('review_status')
            boxes = row.get('boxes')
            size = row.get('image_size')
            if not isinstance(model_key, str) or not model_key:
                errors.append(f'{location}: model_key 缺失')
                continue
            if not isinstance(image, str) or not image:
                errors.append(f'{location}: image 缺失')
                continue
            key = (model_key, image)
            if key in seen:
                errors.append(f'{location}: 重复记录 {model_key}/{image}')
            seen.add(key)
            if status not in REVIEW_STATUSES:
                errors.append(f'{location}: 未知 review_status={status!r}')
            if not isinstance(boxes, list):
                errors.append(f'{location}: boxes 必须是列表')
                boxes = []
            if (
                not isinstance(size, list)
                or len(size) != 2
                or any(type(value) is not int or value <= 0 for value in size)
            ):
                errors.append(f'{location}: image_size 必须是两个正整数')
                width = height = None
            else:
                width, height = size
            if status == 'visual_candidate' and not boxes:
                errors.append(f'{location}: visual_candidate 必须至少有一个框')
            if status == 'visual_negative' and boxes:
                errors.append(f'{location}: visual_negative 不得包含框')
            for box_index, box in enumerate(boxes):
                if not isinstance(box, dict) or type(box.get('class_id')) is not int or box['class_id'] < 0:
                    errors.append(f'{location}: boxes[{box_index}] class_id 无效')
                    continue
                xyxy = box.get('xyxy')
                if (
                    not isinstance(xyxy, list)
                    or len(xyxy) != 4
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in xyxy)
                ):
                    errors.append(f'{location}: boxes[{box_index}].xyxy 无效')
                    continue
                x1, y1, x2, y2 = xyxy
                if width is not None and not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    errors.append(f'{location}: boxes[{box_index}].xyxy 越界或无序')
            records.append(row)
    if errors:
        raise ValueError('；'.join(errors))
    statuses = Counter(STATUS_ALIASES.get(row['review_status'], row['review_status']) for row in records)
    return {
        'annotation_records': len(records),
        'annotated_images': len({row['image'] for row in records}),
        'known_candidate_boxes': sum(len(row['boxes']) for row in records),
        'status_counts': dict(sorted(statuses.items())),
    }


def audit_review_summary(label_files, summary_path, *, write=False):
    """Compare or explicitly refresh derived counts in an existing review summary."""

    actual = audit_label_files(label_files)
    summary = read_json(summary_path)
    mismatches = {
        key: {'summary': summary.get(key), 'labels': value}
        for key, value in actual.items()
        if summary.get(key) != value
    }
    if write:
        summary.update(actual)
        summary['count_semantics'] = {
            'annotation_records': 'all current records across the supplied label files',
            'reviewed_records': 'records examined by the named review pass; independent of current label-file totals',
            'status_counts': 'review_status values, with partial_unlabeled normalized to partial',
        }
        write_json(summary_path, summary)
    return {'summary': summary_path.relative_to(ROOT).as_posix(), 'actual': actual, 'mismatches': mismatches}


def predict(batch, checkpoint_file, run, device, selection=None):
    if device != 'cpu' and not device.isdecimal():
        raise ValueError('设备须是 cpu 或单张 CUDA 卡编号')
    sources = selected_images(selection) if selection is not None else source_images(batch)
    specs = checkpoint_specs(checkpoint_file)
    run.mkdir(parents=True, exist_ok=False)
    (run / 'predictions').mkdir()
    (run / 'runtime').mkdir()
    write_json(run / 'input_sources.json', sources)
    write_json(run / 'checkpoint_specs.json', specs)
    if selection is not None:
        write_json(run / 'selection_snapshot.json', read_json(selection))
    os.environ['YOLO_CONFIG_DIR'] = str(run / 'runtime')
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    if device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，请显式选择 cpu')
    summary = {
        'batch': batch.relative_to(ROOT).as_posix() if batch is not None else None,
        'selection': selection.relative_to(ROOT).as_posix() if selection is not None else None,
        'environment': {'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
                        'device': device, 'gpu': torch.cuda.get_device_name(int(device)) if device != 'cpu' else None},
        'source_counts': dict(Counter(sources.values())),
        'inference': {'imgsz': 640, 'rect': False, 'conf_floor': CONFIDENCE_FLOOR,
                      'nms_iou': .7, 'max_det': 100},
        'models': {},
    }
    paths = list(sources)
    failures = 0
    for key, spec in specs.items():
        model_record = summary['models'][key] = {'status': 'running', 'weights': spec['weights'], 'images': 0}
        write_json(run / 'prediction_summary.json', summary)
        try:
            model = YOLO(str(ROOT / spec['weights']), task=spec['task'])
            names = [model.names[index] for index in range(len(model.names))]
            if names != spec['class_names']:
                raise ValueError(f'权重类别顺序与冻结配置不符：{names}')
            model_record['class_names'] = names
            with (run / 'predictions' / f'{key}.jsonl').open('x', encoding='utf-8') as out:
                for start in range(0, len(paths), 4):
                    chunk = paths[start:start + 4]
                    images = [cv2.imdecode(np.fromfile(ROOT / image, np.uint8), cv2.IMREAD_COLOR) for image in chunk]
                    if any(image is None for image in images):
                        raise ValueError(f'图像解码失败：{chunk}')
                    results = model.predict(images, imgsz=640, rect=False, device=device, conf=CONFIDENCE_FLOOR,
                                            iou=.7, max_det=100, save=False, verbose=False)
                    for path, result in zip(chunk, results, strict=True):
                        boxes = []
                        for index, (x1, y1, x2, y2, confidence, class_id) in enumerate(result.boxes.data.cpu().tolist()):
                            box = {'class_id': int(class_id), 'confidence': confidence, 'xyxy': [x1, y1, x2, y2]}
                            if result.keypoints is not None:
                                box['keypoints'] = result.keypoints.data[index].cpu().tolist()
                            boxes.append(box)
                        out.write(json.dumps({'image': path, 'source': sources[path],
                                              'shape_hw': list(result.orig_shape), 'boxes': boxes}) + '\n')
                        model_record['images'] += 1
            model_record['status'] = 'complete'
        except (OSError, RuntimeError, ValueError) as exc:
            model_record.update(status='failed', error=str(exc))
            failures += 1
        finally:
            write_json(run / 'prediction_summary.json', summary)
            print(f'{key}: {model_record["status"]}, {model_record["images"]} 张', flush=True)
    return 1 if failures else 0


def score_candidate(candidate, prediction, confidence, class_names):
    from engine.reporting import box_iou, match_detections

    truth = candidate['boxes']
    height, width = prediction['shape_hw']
    for box in truth:
        x1, y1, x2, y2 = box['xyxy']
        if (type(box['class_id']) is not int or not 0 <= box['class_id'] < len(class_names)
                or not all(math.isfinite(value) for value in (x1, y1, x2, y2))
                or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)):
            raise ValueError(f'候选框或类别越界：{candidate}')
    if (candidate['review_status'] == 'visual_negative') != (not truth):
        raise ValueError(f'候选正负状态与标签不符：{candidate}')
    raw = prediction['boxes']
    active = [box for box in raw if box['confidence'] >= confidence]
    matches, missed, extra = match_detections(truth, active, MATCH_IOU)
    misses = []
    for index in missed:
        box = truth[index]
        same = [item for item in raw if item['class_id'] == box['class_id']]
        wrong = [item for item in active if item['class_id'] != box['class_id']]
        best = max(same, key=lambda item: box_iou(box['xyxy'], item['xyxy']), default=None)
        best_iou = box_iou(box['xyxy'], best['xyxy']) if best else 0
        if any(box_iou(box['xyxy'], item['xyxy']) >= MATCH_IOU for item in wrong):
            reason = 'wrong_class_overlap'
        elif best_iou >= MATCH_IOU and best['confidence'] < confidence:
            reason = 'below_confidence_threshold'
        elif best_iou >= .1:
            reason = 'localization_or_assignment'
        else:
            reason = 'no_overlapping_prediction'
        misses.append({'truth': box, 'reason': reason, 'closest_same_class': best, 'closest_iou': best_iou})
    return {'image': candidate['image'], 'source': prediction['source'], 'model_key': candidate['model_key'],
            'notes': candidate.get('notes', ''), 'review_status': candidate['review_status'],
            'tp': len(matches), 'fn': len(misses), 'fp': len(extra),
            'misses': misses, 'extras': [active[index] for index in extra],
            'truth': truth, 'active_predictions': active}


def file_copy_aliases():
    aliases = defaultdict(set)
    manifest = ROOT / SOURCE_COPY_MANIFEST
    if manifest.is_file():
        with manifest.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('generation_method') == 'file_copy':
                    image, original = row['path'], row['original_path']
                    aliases[image].add(original)
                    aliases[original].add(image)
    return aliases


def same_copied_image(image, alias):
    if Path(image).is_absolute() or Path(alias).is_absolute():
        return False
    try:
        source = workspace_path(image, *SELECTION_IMAGE_ROOTS)
        copy = workspace_path(alias, *SELECTION_IMAGE_ROOTS)
    except ValueError:
        return False
    return source.is_file() and copy.is_file() and filecmp.cmp(source, copy, shallow=False)


def score(run, label_files, output, confidence):
    if not CONFIDENCE_FLOOR <= confidence <= 1:
        raise ValueError('置信度超出已保存的推理范围')
    base = read_json(run / 'prediction_summary.json')
    labels = {}
    for path in label_files:
        for row in read_json(path):
            key = (row['model_key'], row['image'])
            if key in labels:
                raise ValueError(f'候选标注重复：{key}')
            labels[key] = row
    report = {'confidence': confidence, 'match_iou': MATCH_IOU,
              'scope': 'independent visual candidates; descriptive box diagnostics',
              'inputs': [p.relative_to(ROOT).as_posix() for p in label_files], 'models': {}}
    aliases = None
    per_image = []
    for key in sorted({name for name, _ in labels} | base['models'].keys()):
        subset = {image: row for (name, image), row in labels.items() if name == key}
        excluded = Counter(row['review_status'] for row in subset.values() if row['review_status'] not in SCORED)
        complete = {image: row for image, row in subset.items() if row['review_status'] in SCORED}
        record = base['models'].get(key, {'status': 'checkpoint_missing'})
        summary = {'status': record['status'], 'eligible_images': len(complete),
                   'unscored_review_statuses': dict(excluded)}
        report['models'][key] = summary
        if record['status'] != 'complete':
            continue
        with (run / 'predictions' / f'{key}.jsonl').open(encoding='utf-8') as stream:
            predictions = {item['image']: item for item in map(json.loads, stream)}
        matched = {image: image for image in complete.keys() & predictions.keys()}
        missing = complete.keys() - matched.keys()
        if missing and aliases is None:
            aliases = file_copy_aliases()
        for image in sorted(missing):
            candidates = [alias for alias in aliases[image] & predictions.keys()
                          if same_copied_image(image, alias)]
            if len(candidates) != 1:
                raise ValueError(f'{key} 的候选图缺少唯一的冻结推理结果：{image}')
            matched[image] = candidates[0]
        if len(set(matched.values())) != len(matched):
            raise ValueError(f'{key} 的多条候选标注对应同一冻结推理结果')
        results = []
        for image, row in complete.items():
            prediction_image = matched[image]
            result = score_candidate(row, predictions[prediction_image], confidence, record['class_names'])
            if prediction_image != image:
                result['prediction_image'] = prediction_image
                result['prediction_match'] = {'method': 'verified_byte_identical_file_copy',
                                              'manifest': SOURCE_COPY_MANIFEST}
            results.append(result)
        per_image.extend(results)
        sources = defaultdict(Counter)
        for row in results:
            sources[row['source']].update(images=1, tp=row['tp'], fp=row['fp'], fn=row['fn'])
        summary.update(scored_images=len(results), negative_images=sum(not row['truth'] for row in results),
                       **{metric: sum(row[metric] for row in results) for metric in ('tp', 'fp', 'fn')},
                       by_source=dict(sources),
                       miss_reasons=dict(Counter(miss['reason'] for row in results for miss in row['misses'])))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'label_snapshot.json', list(labels.values()))
    write_json(output / 'per_image.json', per_image)
    write_json(output / 'summary.json', report)
    print(f'{len(per_image)} 条图像与任务的候选对照。')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    inference = actions.add_parser('predict')
    inputs = inference.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--batch')
    inputs.add_argument('--selection')
    inference.add_argument('--checkpoints', required=True)
    inference.add_argument('--run', required=True)
    inference.add_argument('--device', default='0')
    scoring = actions.add_parser('score')
    scoring.add_argument('--run', required=True)
    scoring.add_argument('--output', required=True)
    scoring.add_argument('--labels', nargs='+', required=True)
    scoring.add_argument('--confidence', type=float, default=.25)
    audit = actions.add_parser('audit-labels')
    audit.add_argument('--labels', nargs='+', required=True)
    audit.add_argument('--review-summary', required=True)
    audit.add_argument('--write', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'predict':
            run = workspace_path(args.run, 'runs')
            return predict(workspace_path(args.batch, 'datasets/annotated') if args.batch else None,
                           workspace_path(args.checkpoints, 'datasets/annotated'), run, args.device,
                           workspace_path(args.selection, 'datasets/annotated') if args.selection else None)
        if args.action == 'score':
            run = workspace_path(args.run, 'runs')
            return score(run, [workspace_path(path, 'datasets/annotated') for path in args.labels],
                         workspace_path(args.output, 'runs'), args.confidence)
        report = audit_review_summary(
            [workspace_path(path, 'datasets/annotated') for path in args.labels],
            workspace_path(args.review_summary, 'datasets/annotated'),
            write=args.write,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if args.write or not report['mismatches'] else 1
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f'失败：{exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
