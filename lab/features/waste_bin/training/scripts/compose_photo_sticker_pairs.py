"""Compose four inspectable P22 sticker/bin-color pairs on diagnostic P09.

This fixed, bounded pilot has a shared diagnostic background and physical
stickers. Its labels are candidates for development regression only.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/waste_bin_state or waste_sticker/. Production remains paused.")

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from compose_fire_bin_pilot import ANNOTATIONS, BACKGROUND, RENDER, ROOT, TASKS, decode_mask, image, write_image

PHOTO = ROOT / 'datasets/raw/car_real_capture_20261112/P22_TRASH.png'
PHOTO_REVIEW = ROOT / 'datasets/annotated/waste_bin/v4_review_20261112/review_manifest.json'
# The inner paper crops avoid a strip of the original colored bin; their
# source label boxes are recorded separately below.
STICKERS = {
    '油漆': {'class_id': 16, 'label_xyxy': [280, 381, 306, 408], 'crop_xyxy': [282, 383, 304, 404]},
    '纽扣电池': {'class_id': 12, 'label_xyxy': [569, 365, 597, 395], 'crop_xyxy': [572, 371, 594, 392]},
}
PAIRS = [('red', '油漆'), ('green', '油漆'), ('red', '纽扣电池'), ('green', '纽扣电池')]


def label_lines(boxes):
    return ''.join(f"{b['class_id']} {(b['xyxy'][0]+b['xyxy'][2])/1280:.6f} "
                   f"{(b['xyxy'][1]+b['xyxy'][3])/960:.6f} "
                   f"{(b['xyxy'][2]-b['xyxy'][0])/640:.6f} "
                   f"{(b['xyxy'][3]-b['xyxy'][1])/480:.6f}\n" for b in boxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--raw-icon-preview', action='store_true',
                        help='use larger source-icon patches for a separate readability review')
    args = parser.parse_args()
    # Resolve all consumed media before reserving any output.
    output = args.output.resolve()
    owner = (ROOT / 'datasets/exports/waste_bin').resolve()
    if not output.is_relative_to(owner) or output == owner:
        parser.error('output must be a directory under datasets/exports/waste_bin')
    render_manifest = RENDER / 'manifest.json'
    sample = json.loads(render_manifest.read_text(encoding='utf8'))['samples'][0]
    media = [BACKGROUND, RENDER / sample['image'], RENDER / sample['mask_visible']]
    if args.raw_icon_preview:
        media += [ROOT / 'datasets/raw/waste_bin/toolkit/SourcePic' / (name + '.png') for name in STICKERS]
    else:
        media.append(PHOTO)
    procedural = [render_manifest, ANNOTATIONS, TASKS] + ([] if args.raw_icon_preview else [PHOTO_REVIEW])
    parameters = {'raw_icon_preview': args.raw_icon_preview}
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    declared = list(dict.fromkeys(media + procedural))
    with generation_guard('waste_bin/compose_photo_sticker_pairs.py/' + output.relative_to(ROOT).as_posix(),
                          declared, [output], parameters,
                          code_paths=[Path(__file__), ROOT / 'lab/features/waste_bin/training/scripts/compose_fire_bin_pilot.py', ROOT / 'lab/features/waste_bin/training/scripts/bin_render_contract.py'], procedural_inputs=procedural) as run:
        if run.reused:
            return
        return _produce(args)


def _produce(args):
    output = args.output.resolve()
    owner = (ROOT / 'datasets/exports/waste_bin').resolve()
    if not output.is_relative_to(owner) or output == owner:
        parser.error('output must be a new directory below datasets/exports/waste_bin')
    task = next(x for x in json.loads(TASKS.read_text(encoding='utf8'))['tasks'] if x['id'] == 'waste_bin')
    classes = list(task['waste_items'])
    for name, info in STICKERS.items():
        if classes[info['class_id']] != name:
            raise ValueError('Sticker class does not match perception task order: ' + name)
    if not args.raw_icon_preview:
        review = json.loads(PHOTO_REVIEW.read_text(encoding='utf8'))
        row = next(x for x in review['images'] if x['id'] == 'P22_TRASH')
        source_boxes = row['tasks']['waste_sticker']['boxes']
        for info in STICKERS.values():
            if [info['class_id'], info['label_xyxy']] not in source_boxes:
                raise ValueError('P22 candidate label differs from the reviewed crop')
    src = json.loads((RENDER / 'manifest.json').read_text(encoding='utf8'))
    if src['status'] != 'complete_candidates' or len(src['samples']) != 1:
        raise ValueError('Expected one candidate render')
    sample = src['samples'][0]
    render = image(RENDER / sample['image'])
    mask = image(RENDER / sample['mask_visible'], cv2.IMREAD_UNCHANGED)
    photo = None if args.raw_icon_preview else image(PHOTO)
    background = image(BACKGROUND)
    if (render.shape != (480, 640, 3) or mask.shape != (480, 640, 4) or
            background.shape != (480, 640, 3) or (photo is not None and photo.shape != background.shape)):
        raise ValueError('Expected 640x480 source images and RGBA mask')
    labels_bg = {r['model_key']: r['boxes'] for r in json.loads(ANNOTATIONS.read_text(encoding='utf8'))
                 if r['image'] == BACKGROUND.relative_to(ROOT).as_posix()
                 and r['model_key'] in ('fire_sticker', 'building_name_paper')}
    if set(labels_bg) != {'fire_sticker', 'building_name_paper'}:
        raise ValueError('P09 background candidate labels missing')
    output.mkdir(parents=True)
    manifest = {'status': 'candidate_requires_visual_review', 'role': 'development_regression',
                'background': BACKGROUND.relative_to(ROOT).as_posix(),
                'background_annotations': ANNOTATIONS.relative_to(ROOT).as_posix(),
                'sticker_parent': PHOTO.relative_to(ROOT).as_posix(),
                'sticker_source_review': PHOTO_REVIEW.relative_to(ROOT).as_posix(),
                'bin_render_manifest': (RENDER / 'manifest.json').relative_to(ROOT).as_posix(),
                'bin_render_image': sample['image'], 'image_size': [640, 480],
                'reuse': 'One P09 image, one P22 photograph, one rendered scene and two physical sticker appearances reused.',
                'samples': []}
    if args.raw_icon_preview:
        manifest['sticker_parent'] = 'datasets/raw/waste_bin/toolkit/SourcePic'
        manifest['sticker_source_review'] = None
        manifest['reuse'] = 'One P09 image, one rendered scene and two source icons reused.'
    for color, name in PAIRS:
        bin_info = next(x for x in sample['labels'] if x['bin'] == color)
        if bin_info['state'] != 'closed' or not bin_info['target_eligible']:
            raise ValueError('Selected bin must have visible closed-state evidence')
        rgba = mask[:, :, [2, 1, 0, 3]].astype(np.float32) / 255
        selected = decode_mask(rgba, color)
        ys, xs = np.where(selected)
        if not len(xs):
            raise ValueError('Empty bin mask: ' + color)
        x1, x2, y1, y2 = int(xs.min()), int(xs.max())+1, int(ys.min()), int(ys.max())+1
        sprite, alpha = render[y1:y2, x1:x2], selected[y1:y2, x1:x2].astype(np.uint8)*255
        width = 95 if args.raw_icon_preview else 67
        height = round(sprite.shape[0] * width / sprite.shape[1])
        sprite = cv2.resize(sprite, (width, height), interpolation=cv2.INTER_AREA)
        alpha = cv2.GaussianBlur(cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA), (3, 3), .5).astype(np.float32)/255
        left, bottom = (445, 420) if args.raw_icon_preview else (455, 415)
        top, right = bottom-height, left+width
        for boxes in labels_bg.values():
            for box in boxes:
                a, b, c, d = box['xyxy']
                if max(a, left) < min(c, right) and max(b, top) < min(d, bottom):
                    raise ValueError('Bin placement intersects background target')
        frame = background.copy()
        shadow = np.zeros((480, 640), dtype=np.uint8)
        cv2.ellipse(shadow, (left+width//2, bottom), (width//2, 6), 0, 0, 360, 90, -1)
        shadow = cv2.GaussianBlur(shadow, (21, 21), 6).astype(np.float32)/255
        frame = (frame.astype(np.float32)*(1-shadow[:, :, None])).clip(0, 255).astype(np.uint8)
        roi = frame[top:bottom, left:right].astype(np.float32)
        frame[top:bottom, left:right] = (np.clip(sprite.astype(np.float32)*.82, 0, 255)*alpha[:, :, None] + roi*(1-alpha[:, :, None])).astype(np.uint8)
        vy, vx = np.where(alpha >= .5)
        bin_box = [left+int(vx.min()), top+int(vy.min()), left+int(vx.max())+1, top+int(vy.max())+1]
        info = STICKERS[name]
        ax, ay, bx, by = info['crop_xyxy']
        if args.raw_icon_preview:
            icon_source = ROOT / 'datasets/raw/waste_bin/toolkit/SourcePic' / (name + '.png')
            patch = image(icon_source)
            sx1, sy1, sx2, sy2 = 480, 388, 508, 416
        else:
            icon_source = PHOTO
            patch = photo[ay:by, ax:bx]
            sx1, sy1, sx2, sy2 = 480, 400, 495, 413
        if not (bin_box[0] <= sx1 < sx2 <= bin_box[2] and bin_box[1] <= sy1 < sy2 <= bin_box[3]):
            raise ValueError('Sticker paper must fit within bin box')
        if np.any(alpha[sy1-top:sy2-top, sx1-left:sx2-left] < .95):
            raise ValueError('Sticker paper crosses the bin visible-mask edge')
        patch = cv2.resize(patch, (sx2-sx1, sy2-sy1), interpolation=cv2.INTER_AREA)
        frame[sy1:sy2, sx1:sx2] = patch
        sample_id = color + '_' + str(info['class_id'])
        image_path = output / 'images' / (sample_id + '.png')
        image_path.parent.mkdir(exist_ok=True, parents=True)
        write_image(image_path, frame)
        task_labels = dict(labels_bg)
        task_labels['waste_bin_state'] = [{'class_id': 1, 'xyxy': bin_box}]
        # Keep source identity as a hypothesis until each output has been
        # reviewed at its actual size; do not emit a YOLO sticker label.
        paths = {}
        for task_name, boxes in task_labels.items():
            label_path = output / 'labels' / task_name / (sample_id + '.txt')
            label_path.parent.mkdir(exist_ok=True, parents=True)
            label_path.write_text(label_lines(boxes), encoding='utf8')
            paths[task_name] = label_path.relative_to(output).as_posix()
        manifest['samples'].append({'id': sample_id, 'image': image_path.relative_to(output).as_posix(),
            'bin_color': color, 'bin_state': 'closed', 'bin_source_crop_xyxy': [x1, y1, x2, y2],
            'bin_placement_xyxy': [left, top, right, bottom], 'bin_bbox_xyxy': bin_box,
            'source_sticker_class_id': info['class_id'], 'source_sticker_name': name,
            'sticker_source': icon_source.relative_to(ROOT).as_posix(),
            'sticker_source_label_xyxy': None if args.raw_icon_preview else info['label_xyxy'],
            'sticker_source_crop_xyxy': None if args.raw_icon_preview else info['crop_xyxy'],
            'sticker_placement_xyxy': [sx1, sy1, sx2, sy2],
            'sticker_resize': ('area to 28x28' if args.raw_icon_preview else 'area to 15x13, no upsampling'),
            'bin_brightness_factor': .82, 'shadow': f'opacity 90/255, ellipse {width//2}x6, sigma 6',
            'labels': paths, 'waste_sticker': {'state': 'partial', 'class_id': None,
                'candidate_bbox_xyxy': [sx1, sy1, sx2, sy2], 'label': None,
                'reason': ('output class pending visual review' if args.raw_icon_preview else
                           '15x13 output does not establish the 24-way sticker class')},
            'visual_review': 'pending', 'human_accepted': False})
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print('PAIRED_CANDIDATES', output)


if __name__ == '__main__':
    main()
