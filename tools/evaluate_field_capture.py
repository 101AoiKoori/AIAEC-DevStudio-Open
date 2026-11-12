"""Run frozen V3 inference and score independently reviewed field candidates."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lab/training_common'))

SCORED = {'visual_candidate', 'visual_negative'}
CONFIDENCE_FLOOR = .01
MATCH_IOU = .5


def workspace_path(value, parent):
    path = (ROOT / value).resolve()
    base = (ROOT / parent).resolve()
    if not base.is_relative_to(ROOT) or path == base or not path.is_relative_to(base):
        raise ValueError(f'路径须在 {parent} 下：{value}')
    return path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def source_images(batch):
    photos = {
        workspace_path(row['current_path'], 'datasets/raw'): 'vehicle_camera'
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


def predict(batch, checkpoint_file, run, device):
    if device != 'cpu' and not device.isdecimal():
        raise ValueError('设备须是 cpu 或单张 CUDA 卡编号')
    sources = source_images(batch)
    specs = checkpoint_specs(checkpoint_file)
    run.mkdir(parents=True, exist_ok=False)
    (run / 'predictions').mkdir()
    (run / 'runtime').mkdir()
    os.environ['YOLO_CONFIG_DIR'] = str(run / 'runtime')
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    if device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，请显式选择 cpu')
    summary = {
        'batch': batch.relative_to(ROOT).as_posix(),
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
        if complete.keys() - predictions.keys():
            raise ValueError(f'{key} 的候选图缺少冻结推理结果')
        results = [score_candidate(row, predictions[image], confidence, record['class_names'])
                   for image, row in complete.items()]
        per_image.extend(results)
        sources = defaultdict(Counter)
        for row in results:
            sources[row['source']].update(images=1, tp=row['tp'], fp=row['fp'], fn=row['fn'])
        summary.update(scored_images=len(results), negative_images=sum(not row['truth'] for row in results),
                       **{metric: sum(row[metric] for row in results) for metric in ('tp', 'fp', 'fn')},
                       by_source=dict(sources),
                       miss_reasons=dict(Counter(miss['reason'] for row in results for miss in row['misses'])))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'per_image.json', per_image)
    write_json(output / 'summary.json', report)
    print(f'{len(per_image)} 条图像与任务的候选对照。')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    inference = actions.add_parser('predict')
    inference.add_argument('--batch', required=True)
    inference.add_argument('--checkpoints', required=True)
    inference.add_argument('--run', required=True)
    inference.add_argument('--device', default='0')
    scoring = actions.add_parser('score')
    scoring.add_argument('--run', required=True)
    scoring.add_argument('--output', required=True)
    scoring.add_argument('--labels', nargs='+', required=True)
    scoring.add_argument('--confidence', type=float, default=.25)
    args = parser.parse_args()
    try:
        run = workspace_path(args.run, 'runs')
        if args.action == 'predict':
            return predict(workspace_path(args.batch, 'datasets/annotated'),
                           workspace_path(args.checkpoints, 'datasets/annotated'), run, args.device)
        return score(run, [workspace_path(path, 'datasets/annotated') for path in args.labels],
                     workspace_path(args.output, 'runs'), args.confidence)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f'失败：{exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
