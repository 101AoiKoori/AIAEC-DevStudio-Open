"""Create one inspectable red-bin/fire coexistence candidate on a diagnostic frame.

The output inherits the diagnostic background's development-regression role.
It is not a training export, independent test, or proof of annotation quality.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/waste_bin_state or waste_sticker/.")

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bin_render_contract import decode_mask

BACKGROUND = ROOT / 'Archive/Assets/captures/car_real_capture_20261112/P09_B1.png'
ANNOTATIONS = ROOT / 'datasets/annotated/car_real_capture_20261112/field_eval_v1/annotations_root.json'
RENDER = ROOT / 'datasets/exports/waste_bin/v4_photo_markings_preview'
STICKER = ROOT / 'Archive/Assets/working_sources/digital/waste_sticker/toolkit/SourcePic/纽扣电池.png'
TASKS = ROOT / 'config/perception_tasks.json'


def image(path, flags=cv2.IMREAD_COLOR):
    result = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if result is None:
        raise ValueError('Cannot decode image: ' + str(path))
    return result


def write_image(path, value):
    ok, encoded = cv2.imencode('.png', value)
    if not ok:
        raise OSError('Could not encode image')
    encoded.tofile(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--include-sticker', action='store_true',
                        help='add a single button-cell picture to the red bin')
    args = parser.parse_args()
    # Resolve all consumed media before reserving any output.
    output = args.output.resolve()
    owner = (ROOT / 'datasets/exports/waste_bin').resolve()
    if not output.is_relative_to(owner) or output == owner:
        parser.error('output must be a directory under datasets/exports/waste_bin')
    render_manifest = RENDER / 'manifest.json'
    sample = json.loads(render_manifest.read_text(encoding='utf8'))['samples'][0]
    media = [BACKGROUND, RENDER / sample['image'], RENDER / sample['mask_visible']]
    if args.include_sticker:
        media.append(STICKER)
    procedural = [render_manifest, ANNOTATIONS] + ([TASKS] if args.include_sticker else [])
    parameters = {'include_sticker': args.include_sticker}
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    declared = list(dict.fromkeys(media + procedural))
    with generation_guard('waste_bin/compose_fire_bin_pilot.py/' + output.relative_to(ROOT).as_posix(),
                          declared, [output], parameters,
                          code_paths=[Path(__file__), ROOT / 'lab/features/waste_bin/training/scripts/bin_render_contract.py'], procedural_inputs=procedural) as run:
        if run.reused:
            return
        return _produce(args)


def _produce(args):
    output = args.output.resolve()
    owner = (ROOT / 'datasets/exports/waste_bin').resolve()
    if not output.is_relative_to(owner) or output == owner:
        raise ValueError('output must be a directory under datasets/exports/waste_bin')
    render = json.loads((RENDER / 'manifest.json').read_text(encoding='utf8'))
    if render.get('status') != 'complete_candidates' or len(render['samples']) != 1:
        raise ValueError('Expected the one-frame reviewed render candidate')
    sample = render['samples'][0]
    if sample['states'][1] != 'closed':
        raise ValueError('The selected red-bin source is not closed')
    src = image(RENDER / sample['image'])
    mask = image(RENDER / sample['mask_visible'], cv2.IMREAD_UNCHANGED)
    if src.shape != (480, 640, 3) or mask.shape != (480, 640, 4):
        raise ValueError('Expected 640x480 RGB and RGBA mask')
    rgba = mask[:, :, [2, 1, 0, 3]].astype(np.float32) / 255.0
    binary = decode_mask(rgba, 'red')
    ys, xs = np.where(binary)
    if not len(xs):
        raise ValueError('Red-bin mask is empty')
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    sprite = src[y1:y2, x1:x2]
    alpha = (binary[y1:y2, x1:x2].astype(np.uint8) * 255)
    target_width = 67
    target_height = round(sprite.shape[0] * target_width / sprite.shape[1])
    sprite = cv2.resize(sprite, (target_width, target_height), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (target_width, target_height), interpolation=cv2.INTER_AREA)
    alpha = cv2.GaussianBlur(alpha, (3, 3), .5).astype(np.float32) / 255
    left, bottom = 455, 415
    top, right = bottom - target_height, left + target_width
    frame = image(BACKGROUND)
    if frame.shape != (480, 640, 3) or top < 0 or right > 640:
        raise ValueError('Invalid background or placement')
    records = json.loads(ANNOTATIONS.read_text(encoding='utf8'))
    labels = {r['model_key']: r['boxes'] for r in records if r['image'] == BACKGROUND.relative_to(ROOT).as_posix()
              and r['model_key'] in ('fire_sticker', 'building_name_paper')}
    if set(labels) != {'fire_sticker', 'building_name_paper'}:
        raise ValueError('Background task candidate labels are missing')
    for boxes in labels.values():
        for box in boxes:
            a, b, c, d = box['xyxy']
            if max(a, left) < min(c, right) and max(b, top) < min(d, bottom):
                raise ValueError('Placement intersects a background task label')
    shadow = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(shadow, (left + target_width // 2, bottom), (target_width // 2, 6), 0, 0, 360, 90, -1)
    shadow = cv2.GaussianBlur(shadow, (21, 21), 6).astype(np.float32) / 255
    frame = (frame.astype(np.float32) * (1 - shadow[:, :, None])).clip(0, 255).astype(np.uint8)
    roi = frame[top:bottom, left:right].astype(np.float32)
    foreground = np.clip(sprite.astype(np.float32) * .82, 0, 255)
    frame[top:bottom, left:right] = (foreground * alpha[:, :, None] + roi * (1 - alpha[:, :, None])).astype(np.uint8)
    visible_y, visible_x = np.where(alpha >= .5)
    bbox = [left + int(visible_x.min()), top + int(visible_y.min()),
            left + int(visible_x.max()) + 1, top + int(visible_y.max()) + 1]
    sticker_record = None
    if args.include_sticker:
        # One small paper picture on the lower front, below the bin's fixed symbol.
        # The whole paper is the detection target; class order comes from the task contract.
        sticker = image(STICKER)
        sx1, sy1, sx2, sy2 = 480, 385, 501, 405
        if not (bbox[0] <= sx1 < sx2 <= bbox[2] and bbox[1] <= sy1 < sy2 <= bbox[3]):
            raise ValueError('Sticker must fit within the visible red-bin box')
        sticker = cv2.resize(sticker, (sx2-sx1, sy2-sy1), interpolation=cv2.INTER_AREA)
        frame[sy1:sy2, sx1:sx2] = np.clip(sticker.astype(np.float32) * .82, 0, 255).astype(np.uint8)
        task_config = json.loads(TASKS.read_text(encoding='utf8'))
        task = next(item for item in task_config['tasks'] if item['id'] == 'waste_bin')
        class_id = list(task['waste_items']).index('纽扣电池')
        sticker_record = {'source': STICKER.relative_to(ROOT).as_posix(),
                          'source_class_hypothesis': '纽扣电池', 'source_class_id': class_id,
                          'placement_xyxy': [sx1, sy1, sx2, sy2],
                          'generation': 'area downsample and brightness adjustment',
                          'task_state': 'partial',
                          'reason': 'Class cannot be reliably distinguished from the output pixels alone.'}
    output.mkdir(parents=True)
    write_image(output / 'coexistence.png', frame)
    labels['waste_bin_state'] = [{'class_id': 1, 'xyxy': bbox}]
    for task, boxes in labels.items():
        lines = []
        for box in boxes:
            a, b, c, d = box['xyxy']
            lines.append(f"{box['class_id']} {(a+c)/1280:.6f} {(b+d)/960:.6f} {(c-a)/640:.6f} {(d-b)/480:.6f}")
        (output / (task + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf8')
    record = {'status': 'candidate_requires_visual_review', 'role': 'development_regression',
              'background': BACKGROUND.relative_to(ROOT).as_posix(),
              'background_annotations': ANNOTATIONS.relative_to(ROOT).as_posix(),
              'bin_render_manifest': (RENDER / 'manifest.json').relative_to(ROOT).as_posix(),
              'bin_render_image': sample['image'], 'bin_color': 'red', 'bin_state': 'closed',
              'source_mask_crop_xyxy': [x1, y1, x2, y2], 'placement_xyxy': [left, top, right, bottom],
              'bin_visible_bbox_xyxy': bbox, 'foreground_brightness_factor': .82,
              'shadow': {'opacity_max': 90/255, 'ellipse': [target_width//2, 6], 'blur_sigma': 6},
              'image_size': [640, 480], 'tasks': list(labels), 'label_source': 'background candidate labels and transformed render mask'}
    if sticker_record is not None:
        record['sticker'] = sticker_record
        record['tasks'].append('waste_sticker_partial')
        (output / 'waste_sticker_partial.json').write_text(
            json.dumps(sticker_record, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (output / 'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('COEXISTENCE_CANDIDATE', output)


if __name__ == '__main__':
    main()
