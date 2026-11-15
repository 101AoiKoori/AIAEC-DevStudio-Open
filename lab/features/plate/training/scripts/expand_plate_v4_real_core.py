"""Generate bounded, labelled training variants of the real plate draft.

Only training parents are transformed.  The development and proposed test
images are copied unchanged.  Output is a candidate, not a frozen dataset.
"""
import itertools
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/training_common'))
from camera_tone import apply_tone, read_image, write_image  # noqa: E402

SOURCE = ROOT / 'datasets/exports/plate/v4_real_core_draft_20261112'
OUT = ROOT / 'datasets/exports/plate/v4_real_batch_draft_20261112'
SEED = 20261112
SCALES = (1.05, 1.15, 1.25)
VALUE_GAINS = (0.9, 0.7)
SATURATION_GAINS = (0.85, 0.6)
BLUR_SIGMAS = (0.0, 0.65)


def rel(path):
    return path.relative_to(ROOT).as_posix()


def transform_box(label, matrix, width, height):
    values = label.split()
    if not values:
        return ''
    if len(values) != 5 or values[0] != '0':
        raise ValueError('Expected a single reviewed plate box')
    x, y, w, h = map(float, values[1:])
    x1, x2 = (x - w / 2) * width, (x + w / 2) * width
    y1, y2 = (y - h / 2) * height, (y + h / 2) * height
    points = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]]) @ matrix.T
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    # Do not turn a cropped or unrecognizably tiny plate into complete truth.
    if left < 0 or top < 0 or right > width or bottom > height or right - left < 24 or bottom - top < 8:
        return None
    return f'0 {(left+right)/(2*width):.8f} {(top+bottom)/(2*height):.8f} ' \
           f'{(right-left)/width:.8f} {(bottom-top)/height:.8f}'


def _generate():
    if OUT.exists():
        raise ValueError('Output already exists; select a new version rather than overwrite')
    prior = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    if prior.get('status') != 'draft_not_deliverable':
        raise ValueError('Expected the reviewed real-camera draft')
    originals = [r for r in prior['records'] if r['split'] == 'train' and r['tone_profile'] == 'original']
    if len(originals) != 7 or sum(r['state'] == 'positive' for r in originals) != 6:
        raise ValueError('Training-parent count or task state has changed')
    rng, records, skipped = np.random.default_rng(SEED), [], []
    counts, combinations = Counter(), Counter()
    try:
        for split in ('train', 'val', 'test'):
            for kind in ('images', 'labels'):
                (OUT / kind / split).mkdir(parents=True, exist_ok=True)
        for row in prior['records']:
            source_image, source_label = ROOT / row['image'], ROOT / row['label']
            image_target = OUT / 'images' / row['split'] / source_image.name
            label_target = OUT / 'labels' / row['split'] / source_label.name
            shutil.copy2(source_image, image_target)
            shutil.copy2(source_label, label_target)
            records.append({**row, 'image': rel(image_target), 'label': rel(label_target),
                            'previous_image': row['image']})
            counts[(row['split'], row['state'])] += 1
        combos = list(itertools.product(SCALES, VALUE_GAINS, SATURATION_GAINS, BLUR_SIGMAS))
        for parent_index, parent in enumerate(originals):
            source_image, source_label = ROOT / parent['image'], ROOT / parent['label']
            image = read_image(source_image)
            height, width = image.shape[:2]
            label = source_label.read_text(encoding='utf-8').strip()
            for combo_index, (scale, value, saturation, blur) in enumerate(combos):
                angle = float(rng.uniform(-1.0, 1.0))
                shift_x, shift_y = (float(v) for v in rng.uniform(-4, 4, size=2))
                matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
                matrix[:, 2] += [shift_x, shift_y]
                result_label = transform_box(label, matrix, width, height)
                if result_label is None:
                    skipped.append({'parent': parent['parent_image'], 'combination': combo_index,
                                    'reason': 'plate cropped or too small'})
                    continue
                frame = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
                frame = apply_tone(frame, value_gain=value, saturation_gain=saturation)
                if blur:
                    frame = cv2.GaussianBlur(frame, (0, 0), blur)
                name = f'aug_{parent_index:02d}_{combo_index:02d}'
                image_target, label_target = OUT / 'images/train' / (name + '.png'), OUT / 'labels/train' / (name + '.txt')
                write_image(image_target, frame)
                label_target.write_text(result_label + ('\n' if result_label else ''), encoding='utf-8')
                settings = {'scale': scale, 'angle_degrees': angle, 'shift_xy_pixels': [shift_x, shift_y],
                            'affine_matrix': matrix.tolist(), 'value_gain': value, 'saturation_gain': saturation,
                            'gaussian_sigma_px': blur, 'border': 'replicate', 'compression': 'PNG lossless'}
                records.append({**parent, 'image': rel(image_target), 'label': rel(label_target),
                    'previous_image': parent['image'], 'generation_method': 'augmentation',
                    'tone_profile': 'simulated', 'tone_processing': settings,
                    'label_review': 'projected_from_two_AI_reviewed_parent; batch_visual_QA_pending'})
                counts[('train', parent['state'])] += 1
                combinations[(str(scale), str(value), str(saturation), str(blur))] += 1
        (OUT / 'data.yaml').write_text(yaml.safe_dump({'names': {0: 'plate'},
            **{split: 'images/' + split for split in ('train', 'val', 'test')}}, allow_unicode=True), encoding='utf-8')
        manifest = {'status': 'draft_not_deliverable', 'seed': SEED, 'parent_manifest': rel(SOURCE / 'manifest.json'),
            'scope': 'Train augmentation only; no evaluation. Visual QA, source-gate review and broader coverage remain.',
            'distinct_parent_images': prior['distinct_parent_images'],
            'train_parent_images_augmented': len(originals), 'requested_per_parent': len(combos),
            'skipped': skipped, 'combination_counts': {'|'.join(k): v for k, v in combinations.items()},
            'counts': {split: {state: counts[(split, state)] for state in ('positive', 'negative')}
                       for split in ('train', 'val', 'test')}, 'records': records}
        (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'output': rel(OUT), 'counts': manifest['counts'], 'skipped': len(skipped)}, ensure_ascii=False))
    except BaseException:
        (OUT / 'FAILED.txt').write_text('Draft generation failed; inspect before reuse.\n', encoding='utf-8')
        raise


def _guard_inputs():
    rows = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))['records']
    return [ROOT / r['image'] for r in rows], [SOURCE / 'manifest.json'] + [ROOT / r['label'] for r in rows], {'seed': SEED, 'scales': SCALES, 'value_gains': VALUE_GAINS, 'saturation_gains': SATURATION_GAINS, 'blur_sigmas': BLUR_SIGMAS}


def main():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    media, procedural, parameters = _guard_inputs()
    media = sorted(set(media))
    procedural = sorted(set(procedural))
    with generation_guard("plate.expand_plate_v4_real_core", media, [OUT], parameters,
                          code_paths=[Path(__file__), ROOT / 'lab/training_common/camera_tone.py'], procedural_inputs=procedural, root=ROOT) as run:
        if run.reused:
            return
        _generate()
        manifest_path = OUT / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['source_refs'] = run.source_refs
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
