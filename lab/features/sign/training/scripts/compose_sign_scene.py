"""Compose whole sign-fixture candidates using split-specific original media.

Sign masks are SAM candidates, not ground truth. The validation sign is
truncated at the top in its source; test has only one complete camera view.
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
ANNOTATED = ROOT / 'datasets/annotated/sign/sam2_fixture_candidates_20261110'
BACKGROUNDS = ROOT / 'datasets/raw/person/teammate_submission_20261110/dataset'
EXPORTS = ROOT / 'datasets/exports/sign'
VAL_VIDEO = ROOT / 'Archive/Assets/电动车与人物/视频/人物立牌正背面_电动车直立_倒伏和遮挡_场地扫描_012.mp4'
TEST_BACKGROUND = ROOT / 'Archive/Assets/红绿灯/照片/小车摄像头实拍_近似平视_信号灯正面与道路_001.ppm'
SIGN_VIDEO = 'Archive/Assets/禁止直行牌/视频/禁止直行牌视频.mp4'
TEST_SIGN = 'Archive/Assets/电动车与人物/照片/小车摄像头实拍_近似平视_人物正反面和直立车_007.ppm'


def read_image(path, mode):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), mode)
    if image is None:
        raise ValueError('unreadable image: ' + str(path))
    return image


def cutout(name):
    path = ANNOTATED / name
    image = read_image(path, cv2.IMREAD_UNCHANGED)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError('RGBA sign candidate required')
    # SAM may exclude dark printed areas inside the sign face. Fill only holes
    # enclosed by an existing mask contour so the background does not show
    # through printed arrows or red paint.
    mask = (image[:, :, 3] > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
    image[:, :, 3] = mask
    ys, xs = np.where(image[:, :, 3] > 127)
    if len(xs) < 100:
        raise ValueError('empty sign mask')
    return image[ys.min():ys.max()+1, xs.min():xs.max()+1]


def train_backgrounds():
    with (BACKGROUNDS / 'reports/backgrounds_manifest.csv').open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    result = []
    for row in rows:
        if 0 <= int(row['frame']) <= 179:
            p = (BACKGROUNDS / 'backgrounds' / row['file']).resolve()
            if not p.is_relative_to((BACKGROUNDS / 'backgrounds').resolve()) or not p.is_file():
                raise ValueError('bad background path')
            result.append((p, row))
    if not result:
        raise ValueError('no inspected train ground frames')
    return result


def val_frame():
    cap = cv2.VideoCapture(str(VAL_VIDEO))
    try:
        if not cap.isOpened():
            raise ValueError('cannot open validation video')
        cap.set(cv2.CAP_PROP_POS_MSEC, 4500)
        ok, image = cap.read()
        if not ok:
            raise ValueError('cannot decode validation background')
        return image
    finally:
        cap.release()


def make_image(background, sprite, rng, split):
    image = background.copy()
    if sprite is None:
        return image, None
    if split == 'val':
        height = rng.randint(310, 465)  # top-truncated source at the image edge
        top = 0
    else:
        height = rng.randint(125, 340)
        top = rng.randint(max(5, (340 if split == 'test' else 260)-height), 475-height)
    width = max(2, round(sprite.shape[1] * height / sprite.shape[0]))
    sprite = cv2.resize(sprite, (width, height), interpolation=cv2.INTER_AREA)
    left = rng.randint(5, 955-width)
    shadow = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(shadow, (left+width//2, min(479, top+height)),
                (max(4, width//4), max(2, height//50)), 0, 0, 360, 110, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 4).astype(np.float32)[:, :, None] / 255
    image = np.clip(image.astype(np.float32)*(1-shadow), 0, 255).astype(np.uint8)
    region = image[top:top+height, left:left+width]
    alpha = sprite[:, :, 3:4].astype(np.float32)/255
    region[:] = np.clip(sprite[:, :, :3]*alpha + region*(1-alpha), 0, 255).astype(np.uint8)
    ys, xs = np.where(sprite[:, :, 3] > 127)
    return image, [int(left+xs.min()), int(top+ys.min()), int(left+xs.max()+1), int(top+ys.max()+1)]


def yolo(box):
    x1, y1, x2, y2 = box
    if not 0 <= x1 < x2 <= 960 or not 0 <= y1 < y2 <= 480:
        raise ValueError('out of bounds')
    return f'0 {(x1+x2)/1920:.6f} {(y1+y2)/960:.6f} {(x2-x1)/960:.6f} {(y2-y1)/480:.6f}'


def generate(output, split, count, seed):
    output = output.resolve()
    if output.exists() or not output.is_relative_to(EXPORTS.resolve()):
        raise ValueError('output must be new and under sign exports')
    if not 10 <= count <= 10000 or count % 5:
        raise ValueError('count must be a multiple of 5, between 10 and 10000')
    rng = random.Random(seed)
    if split == 'train':
        backgrounds = train_backgrounds()
        sprites = [(cutout(f'train_{i}.png'), f'train_{i}.png') for i in (8, 14, 20, 26)]
    elif split == 'val':
        backgrounds = [(val_frame(), {'source_video': VAL_VIDEO.relative_to(ROOT).as_posix(), 'seconds': 4.5})]
        sprites = [(cutout('val_frame15.png'), 'val_frame15.png')]
    else:
        backgrounds = [(read_image(TEST_BACKGROUND, cv2.IMREAD_COLOR), {'source_image': TEST_BACKGROUND.relative_to(ROOT).as_posix()})]
        sprites = [(cutout('test_photo007.png'), 'test_photo007.png')]
    (output / 'images').mkdir(parents=True)
    (output / 'labels').mkdir()
    scenarios = Counter()
    summary = {'status': 'running', 'split_candidate': split, 'count': count, 'completed': 0,
               'source_scope': 'all sign views show the same competition prop; capture sessions and masks not independently verified',
               'validation_limits': 'val uses top-truncated sign from one source frame' if split == 'val' else None,
               'seed': seed}
    try:
        with (output / 'manifest.jsonl').open('w', encoding='utf8') as records:
            for index in range(count):
                positive = index % 5 != 0
                entry, info = rng.choice(backgrounds)
                if split == 'train':
                    raw = read_image(entry, cv2.IMREAD_COLOR)
                    background = raw[600:1080, 480:1440]
                    background_source = {'source_video': 'Archive/Assets/场景/综合视频/空场与场景物料_001.mp4',
                                         'source_frame': int(info['frame']), 'source_group': 'scene_video_49s_shared'}
                    foreground_source = SIGN_VIDEO
                else:
                    background = cv2.resize(entry, (960, 480), interpolation=cv2.INTER_AREA)
                    background_source = info
                    foreground_source = VAL_VIDEO.relative_to(ROOT).as_posix() if split == 'val' else TEST_SIGN
                sprite, name = rng.choice(sprites)
                result, box = make_image(background, sprite if positive else None, rng, split)
                level = ('dim', 'normal', 'bright')[index % 3]
                gain = min(1.14, max(.84, rng.gauss({'dim':.88, 'normal':1., 'bright':1.10}[level], .025)))
                result = np.clip(result.astype(np.float32)*gain, 0, 255).astype(np.uint8)
                stem = f'sign_{index+1:06d}'
                ok, encoded = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise OSError('image encode failed')
                encoded.tofile(str(output / 'images' / (stem + '.jpg')))
                (output / 'labels' / (stem + '.txt')).write_text((yolo(box)+'\n') if box else '', encoding='utf8')
                records.write(json.dumps({'id':stem, 'split_candidate':split, 'positive':positive,
                    'bbox_xyxy':box, 'background':background_source, 'foreground_source':foreground_source if positive else None,
                    'foreground_mask':name if positive else None, 'gain':gain, 'light':level,
                    'truncated':positive and split=='val', 'status':'AI_candidate_requires_review'}, ensure_ascii=False)+'\n')
                scenarios['positive' if positive else 'negative'] += 1
                summary['completed'] = index+1
        summary['status'] = 'complete_candidates'
        summary['scenarios'] = dict(scenarios)
    except BaseException as exc:
        summary.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--seed', type=int, default=1918)
    args = parser.parse_args()
    print(json.dumps(generate(ROOT / args.output, args.split, args.count, args.seed), ensure_ascii=False))
