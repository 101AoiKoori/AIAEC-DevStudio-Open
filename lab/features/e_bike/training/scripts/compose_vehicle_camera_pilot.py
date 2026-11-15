"""Compose a bounded upright/fallen comparison on exposed car-camera images.

This is a review-only e-bike pilot. It retains background candidate labels and
never treats an unreviewed background as a verified empty image.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/e_bike/. Production remains paused.")

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import preview_person_occlusions as preview

ROOT = Path(__file__).resolve().parents[5]
EXPORTS = ROOT / 'datasets/exports/e_bike'
FIELD_SELECTION = ROOT / 'datasets/annotated/e_bike/field_eval_masks_20261110/selection_review_candidate.json'
REFINED_SELECTION = ROOT / 'datasets/annotated/e_bike/multiview_masks_refined_20261110_v2/selection.json'
P07_REVIEW = ROOT / 'datasets/annotated/site_capture_20261112/person_vehicle_v4_review/review_manifest.json'
BACKGROUNDS = {'p07': ROOT / 'datasets/raw/car_real_capture_20261112/P07_PS3.png',
               'p10': ROOT / 'datasets/raw/car_real_capture_20261112/P10_B2.png'}
CASES = (('p10_upright', 'p10', 'upright'), ('p10_fallen', 'p10', 'fallen'),
         ('p07_upright', 'p07', 'upright'), ('p07_fallen', 'p07', 'fallen'))


def read_image(path, mode=cv2.IMREAD_COLOR):
    result = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), mode)
    if result is None:
        raise ValueError(f'cannot decode {path}')
    return result


def source_entry(selection, name):
    records = selection.get('records', selection.get('vehicles', []))
    matches = [x for x in records if Path(x['path']).name == name]
    if len(matches) != 1:
        raise ValueError(f'expected one source for {name}')
    return matches[0]


def transformed_sprite(entry):
    path = (ROOT / entry['path']).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('source mask must be in the workspace')
    raw = read_image(path, cv2.IMREAD_UNCHANGED)
    if raw.ndim != 3 or raw.shape[2] != 4:
        raise ValueError('source mask must be RGBA')
    ys, xs = np.nonzero(raw[:, :, 3] > 127)
    if not len(xs):
        raise ValueError('empty source mask')
    source_crop = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    sprite = preview.load_rgba(path)
    if sprite.shape[:2] != (source_crop[3] - source_crop[1], source_crop[2] - source_crop[0]):
        raise ValueError('source crop and reused loader disagree')
    sprite = (preview.resized(sprite, height=145) if entry['class'] == 'upright'
              else preview.resized(sprite, width=175))
    return sprite, source_crop


def yolo_row(class_id, box):
    x1, y1, x2, y2 = box
    return f'{class_id} {(x1+x2)/1280:.6f} {(y1+y2)/960:.6f} {(x2-x1)/640:.6f} {(y2-y1)/480:.6f}'


def generate(output):
    output = output.resolve()
    field = json.loads(FIELD_SELECTION.read_text(encoding='utf8'))
    refined = json.loads(REFINED_SELECTION.read_text(encoding='utf8'))
    media = list(BACKGROUNDS.values()) + [
        ROOT / source_entry(field, 'upright_004.png')['path'],
        ROOT / source_entry(refined, 'sam2_03_fallen.png')['path']]
    procedural = [FIELD_SELECTION, REFINED_SELECTION, P07_REVIEW]
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    operation = 'e_bike.vehicle_camera_pilot:' + output.relative_to(ROOT).as_posix()
    with generation_guard(operation, list(dict.fromkeys(procedural + media)), [output],
                          {}, code_paths=[Path(__file__), Path(preview.__file__), ROOT / 'lab/training_common/production_guard.py'],
                          procedural_inputs=procedural, root=ROOT) as run:
        if run.reused:
            return json.loads((output / 'manifest.json').read_text(encoding='utf8'))
        return _generate(output)


def _generate(output):
    if output.exists() or not output.is_relative_to(EXPORTS.resolve()) or output == EXPORTS.resolve():
        raise ValueError('output must be a new directory under datasets/exports/e_bike')
    field = json.loads(FIELD_SELECTION.read_text(encoding='utf8'))
    refined = json.loads(REFINED_SELECTION.read_text(encoding='utf8'))
    sprites = {'upright': source_entry(field, 'upright_004.png'),
               'fallen': source_entry(refined, 'sam2_03_fallen.png')}
    review = json.loads(P07_REVIEW.read_text(encoding='utf8'))
    p07 = next(x for x in review['images'] if x['id'] == 'car_P07_PS3')['tasks']['e_bike']
    if p07['state'] != 'complete_positive':
        raise ValueError('P07 background e-bike candidate labels are not complete')
    background_boxes = [{'class_id': x['class_id'], 'xyxy': x['xyxy']} for x in p07['boxes']]
    output.mkdir(parents=True)
    for folder in ('images', 'labels_candidate', 'foreground_masks', 'overlays'):
        (output / folder).mkdir()
    report = {'status': 'running', 'role': 'development_regression', 'image_size': [640, 480],
              'class_names': ['upright', 'fallen'],
              'source_scope': 'P07/P10 already exposed; source masks are AI candidates, not independent acquisition',
              'background_review': 'P10 visually inspected with no e-bike; P07 carries two separately reviewed candidate boxes',
              'source_selections': [FIELD_SELECTION.relative_to(ROOT).as_posix(), REFINED_SELECTION.relative_to(ROOT).as_posix()],
              'p07_background_annotation': P07_REVIEW.relative_to(ROOT).as_posix(), 'samples': []}
    try:
        for stem, background_id, category in CASES:
            frame = read_image(BACKGROUNDS[background_id])
            if frame.shape != (480, 640, 3):
                raise ValueError('expected original 640x480 car image')
            sprite, crop = transformed_sprite(sprites[category])
            height, width = sprite.shape[:2]
            left, bottom = (445 if background_id == 'p07' else 430), 421
            top, right = bottom - height, left + width
            if top < 0 or right > 640:
                raise ValueError('candidate falls outside the image')
            inherited = background_boxes if background_id == 'p07' else []
            if any(max(left, a) < min(right, c) and max(top, b) < min(bottom, d)
                   for item in inherited for a,b,c,d in [item['xyxy']]):
                raise ValueError('foreground placement intersects a known background e-bike box')
            shadow = np.zeros((480, 640), dtype=np.uint8)
            cv2.ellipse(shadow, (left + width//2, bottom), (max(3, width//2), 5), 0, 0, 360, 50, -1)
            shadow = cv2.GaussianBlur(shadow, (19, 19), 5).astype(np.float32) / 255
            frame = (frame.astype(np.float32) * (1 - shadow[:, :, None])).clip(0, 255).astype(np.uint8)
            alpha = sprite[:, :, 3].astype(np.float32) / 255
            foreground = sprite[:, :, :3].astype(np.float32) * .88
            region = frame[top:bottom, left:right].astype(np.float32)
            frame[top:bottom, left:right] = (foreground * alpha[:, :, None] + region * (1 - alpha[:, :, None])).clip(0, 255).astype(np.uint8)
            mask = np.zeros((480, 640), dtype=np.uint8)
            mask[top:bottom, left:right] = (alpha >= .5).astype(np.uint8)
            bbox = preview.box(mask)
            class_id = 0 if category == 'upright' else 1
            boxes = inherited + [{'class_id': class_id, 'xyxy': bbox}]
            preview.write_image(output/'images'/(stem+'.png'), frame)
            preview.write_image(output/'foreground_masks'/(stem+'.png'), mask * 255)
            (output/'labels_candidate'/(stem+'.txt')).write_text('\n'.join(yolo_row(x['class_id'], x['xyxy']) for x in boxes)+'\n', encoding='utf8')
            overlay = frame.copy()
            for item in boxes:
                a,b,c,d = item['xyxy']
                cv2.rectangle(overlay, (a,b), (c-1,d-1), (0,255,0) if item['class_id']==0 else (0,0,255), 2)
            preview.write_image(output/'overlays'/(stem+'.png'), overlay)
            report['samples'].append({'id': stem, 'image': f'images/{stem}.png', 'split': 'regression',
                'background': BACKGROUNDS[background_id].relative_to(ROOT).as_posix(),
                'background_boxes': inherited, 'foreground': sprites[category], 'source_crop_xyxy': crop,
                'placement_xyxy': [left,top,right,bottom], 'bbox_xyxy_candidate': bbox,
                'foreground_mask': f'foreground_masks/{stem}.png', 'labels': boxes,
                'candidate_label': f'labels_candidate/{stem}.txt', 'overlay': f'overlays/{stem}.png',
                'transform': {'resize': 'aspect-preserving area', 'brightness_factor': .88,
                              'shadow': 'synthetic blurred ellipse; not measured'},
                'status': 'candidate_requires_visual_review', 'human_accepted': False})
        report['status'] = 'complete_candidates'
    except BaseException as exc:
        report.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({'count': len(generate(ROOT/args.output)['samples'])}, ensure_ascii=False))
