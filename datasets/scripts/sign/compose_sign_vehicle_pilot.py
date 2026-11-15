"""Create a small, review-only sign-view comparison on exposed car images.

The backgrounds were already used for diagnostics. Every output is therefore
development regression, not a training split or independent test.
"""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/sign/. Production remains paused.")

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from datasets.scripts.sign import compose_sign_scene as base

ROOT = base.ROOT
EXPORTS = ROOT / 'datasets/samples/sign'
MASK_DIR = base.ANNOTATED
SAM_MANIFEST = MASK_DIR / 'manifest.json'

# Shared placement makes the two P12 views directly comparable.
CASES = (
    ('p12_front', 'P12_TL2.png', 'train_20.png', 145, 365, 210),
    ('p12_side', 'P12_TL2.png', 'train_8.png', 145, 365, 210),
    ('p12_back', 'P12_TL2.png', 'train_0.png', 145, 365, 210),
    ('p09_oblique', 'P09_B1.png', 'train_14.png', 353, 408, 185),
    ('p12_negative', 'P12_TL2.png', None, None, None, None),
)

# Manually traced visible outer edge in the 576x1024 SAM source image, after
# inspecting the original frame. This supplements a missing pale back board;
# the original yellow post, orange fasteners and green base remain from SAM.
BACK_BOARD_POLYGON = [
    (193, 81), (181, 76), (169, 75), (158, 78), (147, 86), (139, 97),
    (132, 111), (127, 126), (123, 146), (121, 168), (121, 190), (123, 210),
    (128, 231), (134, 250), (142, 269), (152, 288), (163, 307), (176, 328),
    (190, 348), (205, 366), (214, 374), (226, 373), (234, 365), (239, 349),
    (241, 330), (243, 309), (244, 286), (244, 260), (244, 237), (243, 215),
    (241, 191), (238, 169), (235, 149), (229, 132), (224, 118), (219, 106),
    (213, 96), (205, 88),
]


def repaired_back(output, *, write_artifacts=True):
    original = base.read_image(MASK_DIR / 'train_0.png', cv2.IMREAD_UNCHANGED)
    if original.shape != (1024, 576, 4):
        raise ValueError('unexpected source dimensions for manually traced back')
    mask = (original[:, :, 3] > 127).astype(np.uint8) * 255
    board = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(board, [np.asarray(BACK_BOARD_POLYGON, dtype=np.int32)], 255)
    # Discard isolated pale SAM fragments around the open mask boundary, while
    # preserving the separately segmented post and fasteners inside this strip.
    keep = np.zeros(mask.shape, dtype=np.uint8)
    keep[:380, 193:251] = mask[:380, 193:251]
    mask[:380] = cv2.bitwise_or(board[:380], keep[:380])
    fixed = original.copy()
    fixed[:, :, 3] = mask
    if write_artifacts:
        write_image(output / 'back_mask_repaired.png', fixed)
    review = original[:, :, :3].copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(review, contours, -1, (0, 255, 0), 2)
    cv2.polylines(review, [np.asarray(BACK_BOARD_POLYGON, dtype=np.int32)], True, (255, 0, 0), 2)
    if write_artifacts:
        write_image(output / 'back_mask_overlay.png', review)
    ys, xs = np.where(mask > 127)
    if not len(xs):
        raise ValueError('back mask is empty')
    return fixed[ys.min():ys.max()+1, xs.min():xs.max()+1]


def write_image(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError(f'cannot encode {path}')
    encoded.tofile(str(path))


def _generate(output):
    output = output.resolve()
    if output.exists() or not output.is_relative_to(EXPORTS.resolve()) or output == EXPORTS.resolve():
        raise ValueError('output must be a new directory under runs/sign')
    manifest = json.loads(SAM_MANIFEST.read_text(encoding='utf8'))
    sources = {f"train_{r['index']}.png": r['source'] for r in manifest['records'] if 'index' in r}
    output.mkdir(parents=True)
    (output / 'images').mkdir()
    (output / 'labels_candidate').mkdir()
    (output / 'overlays').mkdir()
    report = {'status': 'running', 'role': 'development_regression',
              'source_scope': 'same physical sign and two already-exposed car-camera backgrounds',
              'class': 'whole sign fixture including visible board, pole and base',
              'image_size': [640, 480], 'review': 'pending', 'samples': []}
    try:
        back_sprite = repaired_back(output)
        for name, background_name, sprite_name, left, bottom, height in CASES:
            background_path = ROOT / 'datasets/raw/car_real_capture_20261112' / background_name
            frame = base.read_image(background_path, cv2.IMREAD_COLOR)
            if frame.shape != (480, 640, 3):
                raise ValueError('expected unscaled 640x480 car background')
            box = None
            source = None
            if sprite_name:
                sprite = back_sprite if sprite_name == 'train_0.png' else base.cutout(sprite_name)
                width = max(2, round(sprite.shape[1] * height / sprite.shape[0]))
                sprite = cv2.resize(sprite, (width, height), interpolation=cv2.INTER_AREA)
                top, right = bottom - height, left + width
                if top < 0 or left < 0 or right > 640 or bottom > 480:
                    raise ValueError('sign placement is outside background')
                alpha = sprite[:, :, 3].astype(np.float32) / 255
                ys, xs = np.where(alpha >= .5)
                if not len(xs):
                    raise ValueError('sign mask is empty after resizing')
                # Contact shadow is an approximation, not a measured camera property.
                shadow = np.zeros((480, 640), dtype=np.uint8)
                cv2.ellipse(shadow, (left + width // 2, bottom),
                            (max(3, width // 3), 5), 0, 0, 360, 75, -1)
                shadow = cv2.GaussianBlur(shadow, (15, 15), 4).astype(np.float32) / 255
                frame = (frame.astype(np.float32) * (1 - shadow[:, :, None])).clip(0, 255).astype(np.uint8)
                roi = frame[top:bottom, left:right].astype(np.float32)
                foreground = sprite[:, :, :3].astype(np.float32) * .85
                frame[top:bottom, left:right] = (foreground * alpha[:, :, None] +
                    roi * (1 - alpha[:, :, None])).clip(0, 255).astype(np.uint8)
                box = [left + int(xs.min()), top + int(ys.min()),
                       left + int(xs.max()) + 1, top + int(ys.max()) + 1]
                source = {'mask': (MASK_DIR / sprite_name).relative_to(ROOT).as_posix(),
                          'source_frame': sources[sprite_name],
                          'source_video': base.SIGN_VIDEO,
                          'placement_xyxy': [left, top, right, bottom],
                          'brightness_factor': .85, 'shadow': 'elliptical, approximate'}
                if sprite_name == 'train_0.png':
                    source.update(repaired_mask='back_mask_repaired.png',
                                  mask_overlay='back_mask_overlay.png',
                                  mask_method='manually traced visible back board union original post/fasteners/base',
                                  polygon_xy=BACK_BOARD_POLYGON, polygon_canvas=[576, 1024])
            write_image(output / 'images' / (name + '.png'), frame)
            label = ''
            if box:
                x1, y1, x2, y2 = box
                label = f'0 {(x1+x2)/1280:.6f} {(y1+y2)/960:.6f} {(x2-x1)/640:.6f} {(y2-y1)/480:.6f}\n'
            (output / 'labels_candidate' / (name + '.txt')).write_text(label, encoding='utf8')
            overlay = frame.copy()
            if box:
                cv2.rectangle(overlay, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
            write_image(output / 'overlays' / (name + '.png'), overlay)
            report['samples'].append({'id': name, 'background': background_path.relative_to(ROOT).as_posix(),
                'foreground': source, 'bbox_xyxy_candidate': box,
                'image': f'images/{name}.png', 'candidate_label': f'labels_candidate/{name}.txt',
                'status': 'pending_review', 'human_reviewed': False, 'split': 'regression'})
        report['status'] = 'complete_candidates'
    except BaseException as exc:
        report.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return report


def generate(output):
    raise RuntimeError("Retired production entry; use the task scripts in datasets/scripts with its source.yaml")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = generate(ROOT / args.output)
    print(json.dumps({'status': result['status'], 'count': len(result['samples'])}, ensure_ascii=False))
