"""Crop existing source-tracked frontal standee scenes to 640x480 candidates."""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/person/. Production remains paused.")

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common')]
from camera_tone import apply_tone, read_image, source_role, write_image  # noqa: E402
from lab.training_common.production_guard import generation_guard  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue, queue_inputs  # noqa: E402

SOURCE = ROOT / 'datasets/exports/person/real_scene_candidates_3500_20261110'
OUT = ROOT / 'datasets/exports/person/v4_640_training_candidates_20261112'
QUOTAS = {'single': 400, 'multi': 400, 'occlusion': 250, 'no_person': 200}
CROP = (160, 0, 800, 480)
SEED = 20261116


def rel(path):
    return path.relative_to(ROOT).as_posix()


def source_role_inputs(root):
    """Declare the source queue and the files source_role compares."""
    known = build_queue(root)
    media, procedural, code = queue_inputs(root)
    media = set(media)
    for name in known['exposed_images']:
        path = root / name
        if path.is_file():
            media.add(path)
    for row in known['exposed_video_frames']:
        path = root / row['video']
        if path.is_file():
            media.add(path)
    return media, set(procedural), set(code)


def _produce(output=OUT, quotas=QUOTAS, seed=SEED, exclude_manifest=None, allow_shortfall=False):
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'datasets/exports/person') or output == ROOT / 'datasets/exports/person':
        raise ValueError('Output must be a person export subdirectory')
    excluded_sources = set()
    if exclude_manifest is not None:
        exclude_manifest = exclude_manifest.resolve()
        if not exclude_manifest.is_relative_to(ROOT / 'datasets/exports/person'):
            raise ValueError('Exclusion manifest must be inside person exports')
        excluded = json.loads(exclude_manifest.read_text(encoding='utf-8'))
        excluded_sources = {r['source_image'] for r in excluded['records']}
    if output.exists():
        raise ValueError('Output exists; select a new version instead of overwriting')
    rows = [json.loads(line) for line in (SOURCE / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    grouped = {key: [r for r in rows if r['scenario']['kind'] == key and
                     r['background']['kind'] in ('ground_only', 'track_context')]
               for key in quotas}
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)
    known = build_queue(ROOT)
    roles = {}
    for row in rows:
        image = row['background']['image']
        if image not in roles:
            roles[image] = source_role(ROOT, ROOT / image, known)[0]
    records, counts, skipped, shortfalls = [], Counter(), Counter(), {}
    try:
        (output / 'images/train').mkdir(parents=True)
        (output / 'labels/train').mkdir(parents=True)
        (output / 'masks').mkdir()
        for scenario, quota in quotas.items():
            accepted = 0
            for row in grouped[scenario]:
                if accepted >= quota:
                    break
                if rel(SOURCE / row['person_image']) in excluded_sources:
                    skipped['used_source_scene'] += 1
                    continue
                person_sources = [o['source'] for o in row['objects'] if o['kind'] == 'person']
                if len(person_sources) != len(set(person_sources)):
                    skipped['repeated_standee_artwork'] += 1
                    continue
                if roles[row['background']['image']] == 'regression':
                    skipped['diagnostic_background'] += 1
                    continue
                image = read_image(SOURCE / row['person_image'])
                if image.shape[:2] != (480, 960):
                    raise ValueError('Unexpected image size: ' + row['person_image'])
                x1, y1, x2, y2 = CROP
                crop = image[y1:y2, x1:x2].copy()
                objects, reject = [], False
                for obj in row['objects']:
                    if obj['kind'] != 'person':
                        continue
                    mask = cv2.imdecode(np.fromfile(str(SOURCE / obj['visible_mask']), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if mask is None or mask.shape != (480, 960):
                        raise ValueError('Missing or malformed mask: ' + obj['visible_mask'])
                    full = mask > 127
                    visible = full[y1:y2, x1:x2]
                    pixels, total = int(visible.sum()), int(full.sum())
                    if not pixels:
                        continue
                    yy, xx = np.nonzero(visible)
                    box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
                    if (pixels / total < .85 or obj['visible_fraction'] < .8 or box[3] - box[1] < 100
                            or abs(obj['angles_degrees']['yaw']) > 15):
                        reject = True
                        break
                    objects.append((obj, visible.astype(np.uint8) * 255, box, pixels / total))
                if reject or (scenario != 'no_person' and not objects):
                    skipped['ambiguous_or_missing_person'] += 1
                    continue
                if scenario == 'no_person' and objects:
                    raise ValueError('Negative scene contains person')
                profile = ('original', 'mild', 'moderate')[accepted % 3]
                value, saturation = {'original': (1., 1.), 'mild': (.9, .85), 'moderate': (.7, .6)}[profile]
                if profile != 'original':
                    crop = apply_tone(crop, value_gain=value, saturation_gain=saturation)
                stem = f'person_{len(records)+1:06d}'
                image_out = output / 'images/train' / (stem + '.png')
                label_out = output / 'labels/train' / (stem + '.txt')
                write_image(image_out, crop)
                lines, info = [], []
                for index, (obj, mask, box, fraction) in enumerate(objects):
                    a, b, c, d = box
                    cls = {'community': 0, 'noncommunity': 1}[obj['category']]
                    lines.append(f'{cls} {(a+c)/1280:.8f} {(b+d)/960:.8f} {(c-a)/640:.8f} {(d-b)/480:.8f}')
                    mask_out = output / 'masks' / f'{stem}_{index:02d}.png'
                    write_image(mask_out, mask)
                    info.append({'class': obj['category'], 'bbox_xyxy': box, 'visible_mask': rel(mask_out),
                                 'crop_visible_fraction': fraction, 'foreground_source': obj['source'],
                                 'source_visible_fraction': obj['visible_fraction'],
                                 'planar_yaw_degrees': obj['angles_degrees']['yaw']})
                    counts['instances_' + obj['category']] += 1
                label_out.write_text(('\n'.join(lines) + '\n') if lines else '', encoding='utf-8')
                records.append({'image': rel(image_out), 'label': rel(label_out), 'split': 'train_candidate',
                    'source_image': rel(SOURCE / row['person_image']), 'source_background': row['background'],
                    'scenario': scenario, 'objects': info, 'parent_image_size': [960, 480],
                    'image_size': [640, 480], 'crop_xyxy': list(CROP), 'tone_profile': profile,
                    'tone_parameters': {'value_gain': value, 'saturation_gain': saturation},
                    'generation_method': 'existing_real_scene_composite_crop_and_tone',
                    'label_review': 'visible_mask_projection_candidate_batch_QA_pending'})
                counts['images_' + scenario] += 1
                counts['tone_' + profile] += 1
                accepted += 1
            if accepted < quota:
                shortfalls[scenario] = {'requested': quota, 'accepted': accepted}
                if not allow_shortfall:
                    raise ValueError(f'Could not satisfy {scenario}: {accepted}/{quota}')
        manifest = {'status': 'partial_train_candidates_not_deliverable' if shortfalls else 'train_candidates_not_deliverable',
            'seed': seed, 'quotas': quotas, 'shortfalls': shortfalls,
            'excluded_source_manifest': rel(exclude_manifest) if exclude_manifest else None,
            'scope': 'Same background video and finite printed standee templates; not independent captures. '
                     'No validation/test or model result.', 'source_manifest': rel(SOURCE / 'manifest.jsonl'),
            'counts': dict(counts), 'skipped': dict(skipped), 'records': records}
        (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'output': rel(output), 'counts': dict(counts), 'skipped': dict(skipped)}, ensure_ascii=False))
    except BaseException:
        (output / 'FAILED.txt').write_text('Generation failed; inspect before reuse.\n', encoding='utf-8')
        raise


def main(output=OUT, quotas=QUOTAS, seed=SEED, exclude_manifest=None, allow_shortfall=False):
    output = output.resolve()
    source_manifest = SOURCE / 'manifest.jsonl'
    rows = [json.loads(line) for line in source_manifest.read_text(encoding='utf-8').splitlines()]
    media = set()
    for row in rows:
        media.add(SOURCE / row['person_image'])
        media.add(ROOT / row['background']['image'])
        media.update(SOURCE / obj['visible_mask'] for obj in row['objects'] if obj['kind'] == 'person')
    queue_media, queue_procedural, queue_code = source_role_inputs(ROOT)
    media.update(queue_media)
    procedural = {source_manifest, *queue_procedural}
    if exclude_manifest is not None:
        procedural.add(exclude_manifest.resolve())
    media = {path.resolve() for path in media}
    procedural = {path.resolve() for path in procedural}
    with generation_guard('person.prepare_v4_masked_train:' + output.relative_to(ROOT).as_posix(),
                          [*media, *procedural], [output],
                          {'quotas': quotas, 'seed': seed, 'allow_shortfall': allow_shortfall},
                          code_paths=[Path(__file__), ROOT/'lab/training_common/camera_tone.py', *queue_code],
                          procedural_inputs=procedural, root=ROOT) as run:
        if run.reused:
            return
        return _produce(output, quotas, seed, exclude_manifest, allow_shortfall)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--exclude-manifest', type=Path)
    parser.add_argument('--allow-shortfall', action='store_true')
    parser.add_argument('--quota', nargs=4, type=int, metavar=('SINGLE', 'MULTI', 'OCCLUSION', 'NO_PERSON'))
    args = parser.parse_args()
    quotas = dict(zip(QUOTAS, args.quota)) if args.quota else QUOTAS
    if any(value < 0 for value in quotas.values()) or not any(quotas.values()):
        parser.error('quotas must be nonnegative and not all zero')
    main(args.output, quotas, args.seed, args.exclude_manifest, args.allow_shortfall)
