"""Compose whole sign-fixture candidates using split-specific original media.

Sign masks are SAM candidates, not ground truth. The validation sign is
truncated at the top in its source; test has only one complete camera view.
"""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/sign/.")

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
from datasets.scripts.common.config import source_path
ANNOTATED = source_path('sign', 'foreground_root')
BACKGROUNDS = ROOT / 'datasets/assets/person/dataset'
EXPORTS = ROOT / 'datasets/samples/sign'
VAL_VIDEO = ROOT / 'Archive/Assets/working_sources/legacy/e_bike/视频/人物立牌正背面_电动车直立_倒伏和遮挡_场地扫描_012.mp4'
TEST_BACKGROUND = ROOT / 'Archive/Assets/working_sources/legacy/traffic_light/照片/小车摄像头实拍_近似平视_信号灯正面与道路_001.png'
SIGN_VIDEO = source_path('sign', 'foreground_video').relative_to(ROOT).as_posix()
TEST_SIGN = 'Archive/Assets/working_sources/legacy/e_bike/照片/小车摄像头实拍_近似平视_人物正反面和直立车_007.png'


def read_image(path, mode):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), mode)
    if image is None:
        raise ValueError('unreadable image: ' + str(path))
    return image


def cutout(name, *, working=None):
    path = ANNOTATED / name
    if working:
        path = working.path(path)
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


def _prepare_plan(declaration, split, backgrounds, mask_data):
    if not isinstance(declaration, dict) or declaration.get('model_key') != 'sign':
        raise ValueError('Explicit sign production plan is required')
    if split != 'train' or declaration.get('target_split') != split:
        raise ValueError('Only the train background source is supported')
    constraints = {'image_size': [960, 480], 'background_crop_xyxy': [480, 600, 1440, 1080],
                   'label_kind': 'visible_fixture_box'}
    if declaration.get('constraints') != constraints:
        raise ValueError('Unsupported sign scene constraints')
    states = declaration.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('Sign production requires explicit states')
    available = {path.relative_to(ROOT).as_posix(): (path, row) for path, row in backgrounds}
    prepared, seen = {}, set()
    for state in states:
        if not isinstance(state, dict) or not isinstance(state.get('background'), dict) or set(state['background']) != {'image'}:
            raise ValueError('Sign scene background must specify exactly image')
        name = state['background']['image']
        if name not in available:
            raise ValueError('Background missing from inspected source index')
        target = state.get('target')
        if not isinstance(target, dict) or set(target) != {'kind', 'mask'} or target['kind'] != 'positive':
            raise ValueError('Only positive sign states are supported')
        mask = target['mask']
        if mask not in ('train_8.png', 'train_14.png', 'train_20.png', 'train_26.png'):
            raise ValueError('Unsupported sign mask')
        if state.get('labels') != {'class_id': 0, 'kind': 'visible_fixture_box'}:
            raise ValueError('Unsupported sign label contract')
        aug = state.get('augmentation')
        if (not isinstance(aug, dict) or set(aug) != {'gain'} or
                type(aug['gain']) not in (int, float) or not 0.84 <= aug['gain'] <= 1.14):
            raise ValueError('Brightness gain must be within 0.84 to 1.14')
        params = state.get('parameters')
        if not isinstance(params, dict) or set(params) != {'height', 'left', 'top'}:
            raise ValueError('Placement requires exactly height, left and top')
        if any(type(params[k]) is not int for k in ('height', 'left', 'top')):
            raise ValueError('Placement values must be integers')
        height, left, top = (params[k] for k in ('height', 'left', 'top'))
        if not (125 <= height <= 340 and 5 <= left < 955 and 0 <= top and top + height <= 480):
            raise ValueError('Infeasible sign scene placement')
        if state.get('conditions') != ['positive']:
            raise ValueError('Unsupported scene conditions')
        if type(state.get('max_count')) is not int or not 0 <= state['max_count'] <= 1:
            raise ValueError('Each exact placement state permits at most one sample')
        origins = [r for r in mask_data['records'] if 'index' in r and f"train_{r['index']}.png" == mask]
        if len(origins) != 1:
            raise ValueError('Foreground mask source missing or ambiguous')
        source_video = 'Archive/Assets/working_sources/legacy/site/场景/综合视频/空场与场景物料_001.mp4'
        expected = list(dict.fromkeys([name, source_video, (ANNOTATED / mask).relative_to(ROOT).as_posix(),
                                       origins[0]['source'], SIGN_VIDEO]))
        if state.get('sources') != expected:
            raise ValueError('Declared sources differ from scene inputs')
        identity = json.dumps([state['background'], target, float(aug['gain']), params], sort_keys=True)
        if state['id'] in prepared or identity in seen:
            raise ValueError('Duplicate sign composition state')
        seen.add(identity)
        prepared[state['id']] = (available[name], origins[0], source_video)
    return prepared


def _planned_image(background, sprite, params, gain):
    height, left, top = (params[k] for k in ('height', 'left', 'top'))
    width = max(2, round(sprite.shape[1] * height / sprite.shape[0]))
    if left + width > 960:
        raise ValueError('Sign placement exceeds image width')
    sprite = cv2.resize(sprite, (width, height), interpolation=cv2.INTER_AREA)
    image = background.copy()
    shadow = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(shadow, (left + width // 2, min(479, top + height)),
                (max(4, width // 4), max(2, height // 50)), 0, 0, 360, 110, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 4).astype(np.float32)[:, :, None] / 255
    image = np.clip(image.astype(np.float32) * (1 - shadow), 0, 255).astype(np.uint8)
    alpha = sprite[:, :, 3:4].astype(np.float32) / 255
    image[top:top+height, left:left+width] = np.clip(
        sprite[:, :, :3] * alpha + image[top:top+height, left:left+width] * (1-alpha), 0, 255).astype(np.uint8)
    ys, xs = np.where(sprite[:, :, 3] > 127)
    if not len(xs):
        raise ValueError('Resized sign mask is empty')
    box = [int(left+xs.min()), int(top+ys.min()), int(left+xs.max()+1), int(top+ys.max()+1)]
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8), box


def _generate(output, plan, run, prepared, sampling_seed=None):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(EXPORTS.resolve()) or output == EXPORTS.resolve():
        raise ValueError('Output must be a new directory under sign exports')
    rendered = {}
    for state in plan['strata']:
        if not state['planned_count']:
            continue
        (path, info), origin, source_video = prepared[state['id']]
        raw = read_image(path, cv2.IMREAD_COLOR)
        background = raw[600:1080, 480:1440]
        if background.shape != (480, 960, 3):
            raise ValueError('Background source cannot provide the declared crop')
        image, box = _planned_image(background, cutout(state['target']['mask']),
                                    state['parameters'], state['augmentation']['gain'])
        rendered[state['id']] = (info, origin, source_video, image, box)
    (output / 'images').mkdir(parents=True)
    (output / 'labels_candidate').mkdir()
    result = {'status': 'running', 'split_candidate': plan['target_split'], 'samples': [],
              'scope': 'shared real parents; background absence and masks remain candidate evidence'}
    try:
        for state in plan['strata']:
            if not state['planned_count']:
                continue
            info, origin, source_video, image, box = rendered[state['id']]
            stem = f"sign_{len(result['samples'])+1:06d}"
            image_path = output / 'images' / (stem + '.jpg')
            label_path = output / 'labels_candidate' / (stem + '.txt')
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise OSError('image encode failed')
            encoded.tofile(str(image_path))
            label_path.write_text(yolo(box) + '\n', encoding='utf8')
            actual = {'background': state['background']['image'], 'background_video': source_video,
                      'background_frame': int(info['frame']), 'background_crop_xyxy': [480, 600, 1440, 1080],
                      'foreground_mask': state['target']['mask'], 'foreground_source': origin['source'],
                      'foreground_video': SIGN_VIDEO, 'bbox_xyxy': box,
                      'placement': state['parameters'], 'gain': state['augmentation']['gain'],
                      'background_sign_absence': 'unknown', 'sampling_seed': sampling_seed}
            run.record_sample(state['id'], output=image_path, sources=state['sources'],
                              parameters=state['parameters'], actual=actual)
            result['samples'].append({'id': stem, 'state_id': state['id'],
                'image': image_path.relative_to(output).as_posix(), 'label': label_path.relative_to(output).as_posix(),
                'sources': state['sources'], 'actual': actual, 'status': 'candidate_requires_review'})
        result['status'] = 'complete_candidates'
    except BaseException as exc:
        result.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    return result


def generate(output, split, count, seed, declaration=None):
    raise RuntimeError("Retired production entry; use the task scripts in datasets/scripts with its source.yaml")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--seed', type=int, default=1918)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan_path = (ROOT / args.plan).resolve()
    if not plan_path.is_relative_to(ROOT):
        raise ValueError('Plan must be inside the workspace')
    declaration = json.loads(plan_path.read_text(encoding='utf8'))
    print(json.dumps(generate(ROOT / args.output, args.split, args.count, args.seed, declaration), ensure_ascii=False))
