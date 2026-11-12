"""Prepare unsplit single-fixture box candidates from the provided sign video.

The HSV heuristic is specific to the recorded red sign, yellow pole and green
base on this indoor video. It is not a detector or proof of label correctness.
Candidates must be reviewed before YOLO training/evaluation; uncertain frames
have no label file and must not be interpreted as negatives.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SOURCE = ROOT / 'Archive/Assets/禁止直行牌/视频/禁止直行牌视频.mp4'
OUTPUT_ROOT = ROOT / 'datasets/annotated/sign'


def component_mask(hsv, low, high):
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    mask[:80, :] = 0
    mask[1450:, :] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    kept = np.zeros_like(mask)
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area > 300 and 120 < x < 900 and 0 < y < 1400:
            kept[labels == index] = 255
    return kept


def candidate_box(image):
    if image is None or image.shape != (1920, 1080, 3):
        raise ValueError('expected 1080×1920 BGR source frame')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bands = [((17, 130, 100), (40, 255, 255)),
             ((40, 120, 65), (90, 255, 255)),
             ((0, 140, 80), (10, 255, 255)),
             ((170, 140, 80), (179, 255, 255))]
    masks = [component_mask(hsv, low, high) for low, high in bands]
    if np.count_nonzero(masks[0]) < 300 or np.count_nonzero(masks[1]) < 300:
        return None
    y, x = np.nonzero(np.maximum.reduce(masks))
    # The pale sign back and its narrow edge can extend above the color mask.
    # The fixed top bound is conservative for this one recorded setup.
    x1 = max(0, int(x.min()) - 12)
    y1 = max(0, min(60, int(y.min()) - 12))
    x2 = min(image.shape[1], int(x.max()) + 13)
    y2 = min(image.shape[0], int(y.max()) + 13)
    if not (0 <= x1 < x2 <= 1080 and 0 <= y1 < y2 <= 1920):
        return None
    return [x1, y1, x2, y2]


def yolo_line(box, width=1080, height=1920):
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('invalid box')
    return f'0 {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=2)
    args = parser.parse_args()
    source = (ROOT / args.source).resolve()
    out = (ROOT / args.out).resolve()
    if source != DEFAULT_SOURCE.resolve() or not source.is_file():
        parser.error('该颜色阈值只针对用户提供的禁止直行牌视频')
    if not out.is_relative_to(OUTPUT_ROOT.resolve()) or out == OUTPUT_ROOT.resolve() or out.exists():
        parser.error('输出须是 datasets/annotated/sign 下不存在的目录')
    if not 0 < args.fps <= 4:
        parser.error('抽帧 fps 必须大于 0 且不超过 4')
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError('无法打开视频')
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or count <= 0:
        raise RuntimeError('视频帧率或帧数无效')
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('images', 'labels_candidate', 'overlays'):
        (out / folder).mkdir()
    report = {'status': 'running', 'source_video': source.relative_to(ROOT).as_posix(),
              'source_group': 'single_provided_sign_video', 'split': None,
              'class_id': 0, 'class_meaning': 'single sign fixture including board, pole, and base',
              'annotations': 'HSV-derived approximate boxes; AI visual review and further review required',
              'excluded_are_not_negatives': True, 'source_fps': source_fps,
              'source_frame_count': count, 'sampling_fps_requested': args.fps, 'samples': []}
    path = out / 'manifest.json'
    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save()
    try:
        for number in range(int((count - 1) / source_fps * args.fps) + 1):
            frame_index = min(count - 1, round(number * source_fps / args.fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f'无法读取视频帧 {frame_index}')
            box = candidate_box(image)
            name = f'sign_{frame_index:04d}'
            image_path = out / 'images' / f'{name}.png'
            success, encoded = cv2.imencode('.png', image)
            if not success:
                raise RuntimeError(f'无法编码帧 {frame_index}')
            encoded.tofile(str(image_path))
            label = None
            if box is not None:
                label = f'labels_candidate/{name}.txt'
                (out / label).write_text(yolo_line(box) + '\n', encoding='utf-8')
                overlay = image.copy()
                cv2.rectangle(overlay, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 5)
                ok, encoded = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    raise RuntimeError(f'无法编码预览 {frame_index}')
                encoded.tofile(str(out / 'overlays' / f'{name}.jpg'))
            report['samples'].append({'image': f'images/{name}.png', 'source_frame': frame_index,
                'seconds_approx': frame_index / source_fps, 'bbox_xyxy_candidate': box,
                'candidate_label': label, 'status': 'pending_review', 'human_reviewed': False})
            save()
        report['status'] = 'complete_candidates'
    except BaseException as exc:
        report['status'] = 'failed_or_interrupted'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        capture.release()
        save()
    print(json.dumps({'count': len(report['samples']),
                      'labels_candidate': sum(x['candidate_label'] is not None for x in report['samples']),
                      'output': out.relative_to(ROOT).as_posix()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
