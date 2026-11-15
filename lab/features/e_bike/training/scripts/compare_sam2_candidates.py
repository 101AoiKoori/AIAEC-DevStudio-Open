"""Compare box-prompted SAM2 masks with existing GrabCut review candidates."""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard


def local_path(value):
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError('路径必须位于工作空间内') from exc
    return path


def read_image(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if img is None:
        raise ValueError(f'无法读取图片：{path}')
    return img


def validate_annotation(review, annotation):
    """Only consume a reviewed frame from this review's images directory."""
    if not isinstance(annotation, dict) or annotation.get('class') not in ('upright', 'fallen'):
        raise ValueError('候选类别必须是 upright 或 fallen')
    image_name = annotation.get('image')
    if not isinstance(image_name, str) or not image_name:
        raise ValueError('候选帧路径缺失')
    images_root = (review / 'images').resolve()
    try:
        images_root.relative_to(review.resolve())
        frame = (review / image_name).resolve()
        frame.relative_to(images_root)
    except ValueError as exc:
        raise ValueError('候选帧必须位于复核目录的 images 内') from exc
    if not frame.is_file():
        raise ValueError('候选帧不存在')
    box = annotation.get('bbox_xyxy_normalized')
    if (not isinstance(box, list) or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in box)
            or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
        raise ValueError('候选框必须是有效的归一化 xyxy 坐标')
    return frame


def write_manifest(output, weights, records, status, device, error=None):
    manifest = {
        'purpose': 'limited mask comparison; not verified ground truth',
        'model': weights.relative_to(ROOT).as_posix(),
        'model_source': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/sam2_t.pt',
        'model_license': 'SAM2 checkpoints Apache-2.0; ultralytics package AGPL-3.0',
        'device': device, 'status': status, 'records': records,
    }
    if error is not None:
        manifest['error'] = error
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='对少量平视候选帧比较 SAM2 与 GrabCut')
    parser.add_argument('--review', required=True, help='包含 box_candidates.json 的复核目录')
    parser.add_argument('--model', required=True, help='工作空间内 SAM2 权重路径')
    parser.add_argument('--output', required=True, help='不存在的输出目录')
    parser.add_argument('--indices', nargs='+', type=int, required=True, help='候选框的 1 起始编号')
    parser.add_argument('--device', choices=('cpu', '0'), default='0', help='默认 CUDA 设备 0；CPU 须显式指定')
    args = parser.parse_args()
    review, weights, output = map(local_path, (args.review, args.model, args.output))
    selection = review / 'box_candidates.json'
    annotations = json.loads(selection.read_text(encoding='utf-8'))['annotations']
    if not args.indices or len(args.indices) > 6 or len(set(args.indices)) != len(args.indices):
        parser.error('请选择 1 到 6 个不重复编号')
    if any(index < 1 or index > len(annotations) for index in args.indices):
        parser.error('候选编号超出范围')
    media = []
    for index in args.indices:
        annotation = annotations[index - 1]
        media.extend([validate_annotation(review, annotation), review / 'cutout_candidates' /
                      f'candidate_{index:02d}_{annotation["class"]}.png'])
    media = list(dict.fromkeys(media))
    with generation_guard('e_bike.compare_sam2:' + str(output), [*media, selection, weights],
                          [output], {'indices': args.indices, 'device': args.device}, code_paths=[Path(__file__)],
                          procedural_inputs=[selection, weights]) as run:
        if run.reused:
            return
        return _compare(args, parser)


def _compare(args, parser):
    output = None
    records = []
    started = False
    weights = None
    try:
        review, weights, output = map(local_path, (args.review, args.model, args.output))
        annotations = json.loads((review / 'box_candidates.json').read_text(encoding='utf-8'))['annotations']
        if output.exists() or not weights.is_file():
            raise ValueError('输出目录已存在或模型权重不存在')
        if not args.indices or len(args.indices) > 6 or len(set(args.indices)) != len(args.indices):
            raise ValueError('请选择 1 到 6 个不重复编号')
        if any(index < 1 or index > len(annotations) for index in args.indices):
            raise ValueError('候选编号超出范围')
        frames = {index: validate_annotation(review, annotations[index - 1]) for index in args.indices}
        output.mkdir(parents=True)
        started = True
        write_manifest(output, weights, records, 'running', args.device)
        model = SAM(str(weights))
        sheets = []
        for index in args.indices:
            annotation = annotations[index - 1]
            img = read_image(frames[index])
            h, w = img.shape[:2]
            scale = min(1.0, 1024 / max(h, w))
            small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            sh, sw = small.shape[:2]
            x1, y1, x2, y2 = annotation['bbox_xyxy_normalized']
            box = [x1 * sw, y1 * sh, x2 * sw, y2 * sh]
            result = model.predict(small, bboxes=[box], device=args.device, imgsz=1024, verbose=False)[0]
            if result.masks is None or len(result.masks.data) != 1:
                raise RuntimeError(f'候选 {index} 未产生唯一掩码')
            mask = result.masks.data[0].cpu().numpy().astype(np.uint8) * 255
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            file = f'sam2_{index:02d}_{annotation["class"]}.png'
            cv2.imencode('.png', np.dstack((img, mask)))[1].tofile(str(output / file))
            grab = read_image(review / 'cutout_candidates' /
                              f'candidate_{index:02d}_{annotation["class"]}.png', cv2.IMREAD_UNCHANGED)
            tiles = []
            for rgba in (grab, np.dstack((img, mask))):
                alpha = rgba[:, :, 3:4].astype(np.float32) / 255
                comp = (rgba[:, :, :3] * alpha + 225 * (1 - alpha)).astype(np.uint8)
                tiles.append(cv2.resize(comp, (270, 480), interpolation=cv2.INTER_AREA))
            sheets.append(np.hstack(tiles))
            records.append({'source_frame': annotation['image'], 'class': annotation['class'],
                            'bbox_xyxy_normalized': annotation['bbox_xyxy_normalized'],
                            'sam2_candidate': file, 'status': 'unreviewed'})
            write_manifest(output, weights, records, 'running', args.device)
        cv2.imencode('.jpg', np.vstack(sheets))[1].tofile(str(output / 'comparison.jpg'))
        write_manifest(output, weights, records, 'complete', args.device)
        print(f'完成 {len(records)} 帧对照；掩码待人工复核')
    except KeyboardInterrupt:
        if started:
            write_manifest(output, weights, records, 'interrupted', args.device, '用户或进程中断')
        raise
    except Exception as exc:
        if started:
            write_manifest(output, weights, records, 'failed', args.device, f'{type(exc).__name__}: {exc}')
        parser.exit(2, f'失败：{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
