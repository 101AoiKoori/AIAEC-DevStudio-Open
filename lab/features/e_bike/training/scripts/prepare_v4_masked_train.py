"""Prepare 640x480 e-bike training candidates from recorded masked scenes.

The source scenes share a background video and a finite set of vehicle sprites.
This batch is training-only; it is not an independent dataset or final package.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common')]
from camera_tone import apply_tone, read_image, source_role, write_image  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402
from lab.training_common.production_guard import generation_guard  # noqa: E402
from e_bike_queue_inputs import queue_inputs  # noqa: E402

SOURCE = ROOT / 'datasets/exports/e_bike/real_scene_v4_candidates_4000_20261110'
OUT = ROOT / 'datasets/exports/e_bike/v4_640_training_candidates_20261112'
SEED = 20261114
QUOTAS = {'upright_only': 400, 'fallen_present': 400, 'empty': 200}
CROP = (160, 0, 800, 480)
PARENT_REVIEW = ROOT / 'datasets/exports/e_bike/v4_production_candidate_20261113/positive_parent_review_20261113.json'


def rel(path):
    return path.relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--upright-count', type=int, default=QUOTAS['upright_only'])
    parser.add_argument('--fallen-count', type=int, default=QUOTAS['fallen_present'])
    parser.add_argument('--empty-count', type=int, default=QUOTAS['empty'])
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    source, output = (ROOT / args.source).resolve(), (ROOT / args.output).resolve()
    owner = (ROOT / 'datasets/exports/e_bike').resolve()
    if not source.is_relative_to(owner) or not output.is_relative_to(owner) or source == owner or output == owner:
        parser.error('source and output must be e-bike export subdirectories')
    quotas = {'upright_only': args.upright_count, 'fallen_present': args.fallen_count, 'empty': args.empty_count}
    if any(value < 0 for value in quotas.values()) or sum(quotas.values()) < 1:
        parser.error('counts must be nonnegative and at least one must be positive')
    manifest = source / 'manifest.jsonl'
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    checked = [ROOT / row['background']['image'] for row in rows]
    media = checked + [source / row['image'] for row in rows]
    media += [source / obj['visible_mask'] for row in rows for obj in row['objects']]
    queue = build_queue(ROOT)
    extra_media, extra_procedural = queue_inputs(ROOT, queue, checked)
    media = list(dict.fromkeys(media + extra_media))
    procedural = list(dict.fromkeys([manifest, PARENT_REVIEW] + extra_procedural))
    with generation_guard('e_bike.masked_train:' + str(output), media, [output],
                          {'seed': args.seed, 'quotas': quotas, 'crop': CROP},
                          procedural_inputs=procedural,
                          code_paths=[__file__, ROOT/'lab/training_common/camera_tone.py',
                                      ROOT/'lab/scripts/data/v4_source_gate.py', ROOT/'lab/training_common/train.py',
                                      ROOT/'lab/training_common/engine/data.py', ROOT/'lab/training_common/engine/config.py',
                                      Path(__file__).with_name('e_bike_queue_inputs.py')]) as run:
        if run.reused:
            return
        return _main(source, output, quotas, args.seed)


def _main(source, output, quotas, seed):
    if output.exists():
        raise ValueError('Output exists; select a new version instead of overwriting')
    rows = [json.loads(line) for line in (source / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    grouped = {key: [r for r in rows if r['scenario'] == key] for key in quotas}
    if any(len(grouped[key]) < count for key, count in quotas.items()):
        raise ValueError('Insufficient source scenarios')
    rng = random.Random(seed)
    excluded = set(json.loads(PARENT_REVIEW.read_text(encoding='utf-8'))['excluded_foreground_stems'])
    for values in grouped.values():
        rng.shuffle(values)
    known = build_queue(ROOT)
    background_roles = {}
    for row in rows:
        image = row['background']['image']
        if image not in background_roles:
            background_roles[image] = source_role(ROOT, ROOT / image, known)[0]
    records, skipped, counts = [], Counter(), Counter()
    try:
        (output / 'images/train').mkdir(parents=True)
        (output / 'labels/train').mkdir(parents=True)
        (output / 'masks').mkdir()
        for scenario, quota in quotas.items():
            accepted = 0
            for row in grouped[scenario]:
                if accepted >= quota:
                    break
                if any(Path(obj['mask_source']['path']).stem in excluded for obj in row['objects']):
                    skipped['known_detached_matte'] += 1
                    continue
                if background_roles[row['background']['image']] == 'regression':
                    skipped['diagnostic_background'] += 1
                    continue
                image = read_image(source / row['image'])
                if image.shape[:2] != (480, 960):
                    raise ValueError('Unexpected source image size: ' + row['image'])
                x1, y1, x2, y2 = CROP
                crop = image[y1:y2, x1:x2].copy()
                objects, reject = [], False
                for obj in row['objects']:
                    mask_path = source / obj['visible_mask']
                    mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if mask is None or mask.shape != (480, 960):
                        raise ValueError('Missing or malformed source mask: ' + str(mask_path))
                    full = mask > 127
                    visible = full[y1:y2, x1:x2]
                    pixels, total = int(visible.sum()), int(full.sum())
                    if not pixels:
                        continue
                    yy, xx = np.nonzero(visible)
                    box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
                    if not total or pixels / total < .65 or pixels < 120 or box[2]-box[0] < 20 or box[3]-box[1] < 20:
                        reject = True
                        break
                    objects.append((obj, visible.astype(np.uint8) * 255, box, pixels / total))
                if reject:
                    skipped['ambiguous_crop'] += 1
                    continue
                # A nominal positive whose targets all leave the crop does not
                # count toward positive-state coverage.
                if scenario != 'empty' and not objects:
                    skipped['positive_outside_crop'] += 1
                    continue
                if scenario == 'fallen_present' and not any(obj['class'] == 'fallen' for obj, _, _, _ in objects):
                    skipped['fallen_outside_crop'] += 1
                    continue
                profile = ('original', 'mild', 'moderate')[accepted % 3]
                if profile != 'original':
                    value, saturation = (.9, .85) if profile == 'mild' else (.7, .6)
                    crop = apply_tone(crop, value_gain=value, saturation_gain=saturation)
                else:
                    value, saturation = 1., 1.
                stem = f'bike_{len(records)+1:06d}'
                image_out, label_out = output / 'images/train' / (stem+'.png'), output / 'labels/train' / (stem+'.txt')
                write_image(image_out, crop)
                lines, info = [], []
                for index, (obj, mask, box, fraction) in enumerate(objects):
                    a, b, c, d = box
                    cls = {'upright': 0, 'fallen': 1}[obj['class']]
                    lines.append(f'{cls} {(a+c)/1280:.8f} {(b+d)/960:.8f} {(c-a)/640:.8f} {(d-b)/480:.8f}')
                    mask_out = output / 'masks' / f'{stem}_{index:02d}.png'
                    write_image(mask_out, mask)
                    info.append({'class': obj['class'], 'bbox_xyxy': box, 'visible_mask': rel(mask_out),
                                 'crop_visible_fraction': fraction, 'foreground_source': obj['mask_source']})
                    counts['instances_'+obj['class']] += 1
                label_out.write_text(('\n'.join(lines)+'\n') if lines else '', encoding='utf-8')
                records.append({'image': rel(image_out), 'label': rel(label_out), 'split': 'train_candidate',
                    'source_image': rel(source / row['image']), 'source_background': row['background'],
                    'scenario': scenario, 'objects': info, 'parent_image_size': [960, 480],
                    'image_size': [640, 480], 'crop_xyxy': list(CROP), 'tone_profile': profile,
                    'tone_parameters': {'value_gain': value, 'saturation_gain': saturation},
                    'generation_method': 'real_scene_composite_then_crop_and_tone',
                    'label_review': 'visible_mask_projection_candidate_batch_QA_pending'})
                counts['images_'+scenario] += 1
                counts['tone_'+profile] += 1
                accepted += 1
            if accepted < quota:
                raise ValueError(f'Could not satisfy {scenario}: {accepted}/{quota}')
        manifest = {'status': 'train_candidates_not_deliverable', 'seed': seed, 'quotas': quotas,
            'scope': 'Shared 49-second background video and finite foreground masks; not independent captures. '
                     'No validation/test or model result.', 'source_manifest': rel(source/'manifest.jsonl'),
            'foreground_exclusion_review': rel(PARENT_REVIEW),
            'counts': dict(counts), 'skipped': dict(skipped), 'records': records}
        (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'output': rel(output), 'counts': dict(counts), 'skipped': dict(skipped)}, ensure_ascii=False))
    except BaseException:
        (output/'FAILED.txt').write_text('Generation failed; inspect before reuse.\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
