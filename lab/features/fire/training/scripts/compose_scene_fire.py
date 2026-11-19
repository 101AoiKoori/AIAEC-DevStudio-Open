"""Compose existing labelled building renders over reviewed scene frames.

Outputs separate fire and name-paper detection candidates.  This uses the
existing RGB/instance passes, not a new Blender render.  A source-family group
is retained; generated samples are train candidates, not independent tests.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/fire/.")

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from fire_blender_labels import decode_ids, srgb_to_linear, yolo_row

ROOT = Path.cwd().resolve()
EXPORTS = ROOT / 'datasets/exports/fire'
DEFAULT_SOURCE = EXPORTS / 'blender_candidates_160_20261110'
BACKGROUND = ROOT / 'datasets/annotated/fire/real_background_candidates_20261110'
ICON = ROOT / 'Archive/Assets/working_sources/digital/fire/team_assets/image1.png'
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
from datasets.scripts.common.images import read_image


def building_mask(image):
    """Recover bright building silhouettes from this renderer's gray stage."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    floor = int(np.median(gray[:50]))
    mask = (gray > floor + 9).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = np.zeros_like(mask)
    for idx in sorted(range(1, count), key=lambda j: stats[j, 4], reverse=True)[:3]:
        _, _, width, height, area = stats[idx]
        if area > 200 and height > 1.4 * width:
            result[components == idx] = 255
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, 255, -1)
    return result


def load_sources(source):
    run = json.loads((source / 'run.json').read_text(encoding='utf-8'))
    if (run.get('status') != 'complete_candidates' or run.get('resolution') != 640 or
            run.get('image_size', [640, 640]) != [640, 640]):
        raise ValueError('expected complete 640x640 source render candidates')
    groups = {0: [], 1: [], 2: [], 3: []}
    skipped = Counter()
    for line in (source / 'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        if not item.get('fire_label') or not item.get('name_label') or item.get('occluder'):
            skipped['labels_or_occlusion'] += 1
            continue
        objects = item['instances']
        if any(obj['state'] not in ('visible', 'outside_frame') for obj in objects):
            skipped['visibility'] += 1
            continue
        image = read_image(source / item['image'])
        mask = building_mask(image)
        raw_ids = read_image(source / item['instance_pass'])
        encoded = cv2.cvtColor(raw_ids, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ids = decode_ids(srgb_to_linear(encoded), [obj['id'] for obj in objects])
        # Verify the silhouette includes each labelled target in the ID pass.
        if any(np.count_nonzero((ids == o['id']) & (mask == 0)) for o in objects if o['state'] == 'visible'):
            skipped['target_outside_silhouette'] += 1
            continue
        count = sum(o['kind'] == 'fire_sticker' and o['state'] == 'visible' for o in objects)
        groups[min(count, 3)].append((item, image, mask))
    if any(not group for group in groups.values()):
        raise ValueError(f'insufficient eligible source types: {[len(x) for x in groups.values()]}')
    return groups, skipped


def crop_background(image, rng):
    h, w = image.shape[:2]
    edge = min(h, w)
    center = (w - edge) // 2
    x = min(w - edge, max(0, center + rng.randrange(-100, 101))) if w > edge else 0
    y = rng.randrange(h - edge + 1) if h > edge else 0
    return cv2.resize(image[y:y + edge, x:x + edge], (640, 640)), [x, y, edge]


def ground_line(background):
    gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    bright = (gray[:, 140:500] > 160).mean(axis=1)
    candidates = np.where((np.arange(640) > 300) & (bright > .20))[0]
    if not len(candidates):
        raise ValueError('no visible white ground boundary in selected crop')
    return int(candidates.max())


def transform(image, mask, instances, rng, ground):
    # A source touching the boundary was originally truncated; preserve its
    # position so the hard edge stays on the output border.
    touching = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    ys, xs = np.nonzero(mask)
    bottom = int(ys.max())
    center = float((xs.min() + xs.max()) / 2)
    for _ in range(30):
        scale = 1.0 if touching else rng.uniform(.70, .98)
        tx = 0.0 if touching else rng.uniform(-22, 22) + (1 - scale) * center
        target_bottom = min(620, ground + rng.uniform(20, 55))
        ty = 0.0 if touching else target_bottom - scale * bottom
        matrix = np.float32([[scale, 0, tx], [0, scale, ty]])
        records = []
        for obj in instances:
            if obj['state'] != 'visible':
                continue
            x1, y1, x2, y2 = obj['pixel_bbox_xyxy']
            box = [round(x1*scale+tx), round(y1*scale+ty), round(x2*scale+tx), round(y2*scale+ty)]
            if not (0 <= box[0] < box[2] <= 640 and 0 <= box[1] < box[3] <= 640):
                break
            records.append((obj, box))
        else:
            return (cv2.warpAffine(image, matrix, (640, 640)),
                    cv2.warpAffine(mask, matrix, (640, 640), flags=cv2.INTER_NEAREST), records,
                    {'scale': scale, 'tx': tx, 'ty': ty})
    raise ValueError('could not keep all labelled objects in frame')


def distribute(total, weights):
    cumulative = [round(total * sum(weights[:i+1]) / sum(weights)) for i in range(len(weights))]
    return [cumulative[0]] + [cumulative[i] - cumulative[i-1] for i in range(1, len(cumulative))]


def add_two_papers(image, mask, instances, icon, rng):
    """Put two labelled papers on different floors of a single empty facade."""
    result = image.copy()
    ys, xs = np.nonzero(mask)
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    width, height = x2 - x1, y2 - y1
    paper_w = max(10, round(width * rng.uniform(.105, .13)))
    paper_h = round(paper_w * 3.5 / 2.5)
    paper = cv2.resize(icon, (paper_w, paper_h), interpolation=cv2.INTER_AREA)
    shade = rng.uniform(.77, .94)
    paper = np.clip(paper.astype(np.float32) * shade, 0, 255).astype(np.uint8)
    records = list(instances)
    # Different rows prevent overlapping paper and preserve the name area.
    for fraction_y in rng.sample((.30, .42, .54, .66, .78), 2):
        for _ in range(20):
            x = int(x1 + width * rng.uniform(.15, .75))
            y = int(y1 + height * fraction_y)
            x = min(x, x2 - paper_w - 1)
            y = min(y, y2 - paper_h - 1)
            if x >= 0 and y >= 0 and np.all(mask[y:y+paper_h, x:x+paper_w] > 0):
                break
        else:
            raise ValueError('could not fit paper on the source facade')
        result[y:y+paper_h, x:x+paper_w] = paper
        records.append({'state': 'visible', 'kind': 'fire_sticker', 'building': 'A',
                        'pixel_bbox_xyxy': [x, y, x+paper_w, y+paper_h]})
    return result, records


def _produce(source, output, count, seed):
    if output.exists() or not output.resolve().is_relative_to(EXPORTS.resolve()):
        raise ValueError('output exists or is outside fire exports')
    if not 10 <= count <= 10000:
        raise ValueError('count must be between 10 and 10000')
    groups, skipped = load_sources(source)
    backgrounds = [read_image(BACKGROUND / 'images' / f'{i:04}.png') for i in range(1, 7)]
    icon = read_image(ICON)
    rng = random.Random(seed)
    schedule = [k for k, n in enumerate(distribute(count, [20, 30, 25, 25])) for _ in range(n)]
    rng.shuffle(schedule)
    name_positive = round(count * .8)
    selection = set(rng.sample(range(count), name_positive))
    for kind in ('fire', 'name'):
        for folder in ('images/train', 'labels/train'):
            (output / kind / folder).mkdir(parents=True, exist_ok=True)
        (output / kind / 'classes.txt').write_text(
            ('fire_sticker' if kind == 'fire' else 'name_paper') + '\n', encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(
        {'status': 'running', 'requested_per_task': count}, ensure_ascii=False, indent=2), encoding='utf-8')
    counts = Counter()
    with (output / 'manifest.jsonl').open('w', encoding='utf-8') as manifest:
        for i, bucket in enumerate(schedule):
            generated_papers = bucket == 2 and rng.random() < .8
            item, foreground, mask = rng.choice(groups[0 if generated_papers else bucket])
            instances = item['instances']
            if generated_papers:
                foreground, instances = add_two_papers(foreground, mask, instances, icon, rng)
            for _ in range(50):
                bg_index = rng.randrange(len(backgrounds))
                bg, crop = crop_background(backgrounds[bg_index], rng)
                try:
                    ground = ground_line(bg)
                    fg, cutout, objects, affine = transform(foreground, mask, instances, rng, ground)
                    break
                except ValueError:
                    pass
            else:
                raise ValueError('no valid scene/background placement')
            alpha = cv2.GaussianBlur(cutout, (3, 3), .6).astype(np.float32)[:, :, None] / 255
            result = np.clip(fg * alpha + bg * (1-alpha), 0, 255).astype(np.uint8)
            gain = min(1.25, max(.72, rng.gauss(1, .1)))
            result = np.clip(result.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            name_result = result if i in selection else np.clip(bg.astype(np.float32)*gain, 0, 255).astype(np.uint8)
            stem = f'scene_{i+1:06d}'
            for kind, image in (('fire', result), ('name', name_result)):
                target = 'fire_sticker' if kind == 'fire' else 'name_sign'
                rows = [yolo_row(box, 640, 640) for obj, box in objects
                        if obj['kind'] == target and (kind != 'name' or i in selection)]
                ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise ValueError('JPEG encoding failed')
                data.tofile(str(output / kind / 'images/train' / (stem + '.jpg')))
                (output / kind / 'labels/train' / (stem + '.txt')).write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf-8')
            counts[str(bucket)] += 1
            manifest.write(json.dumps({'id': stem, 'background_index': bg_index + 1,
                'background_crop_xy_size': crop, 'background_source_group': 'scene_video_49s_shared',
                'foreground_source': item['image'], 'foreground_source_group': item['source_group'],
                'generation': 'real_scene_background_plus_segmented_blender_render',
                'fire_bucket': bucket, 'generated_two_papers': generated_papers,
                'paper_icon_source': ICON.relative_to(ROOT).as_posix() if generated_papers else None,
                'name_positive': i in selection, 'affine': affine,
                'gain': gain, 'ground_boundary_y': ground, 'instances': [{'kind': obj['kind'], 'building': obj['building'],
                    'box': box, 'text': obj.get('text')} for obj, box in objects],
                'split': 'train', 'status': 'candidate_requires_visual_review'}, ensure_ascii=False) + '\n')
    summary = {'status': 'complete_candidates', 'images_per_task': count, 'fire_buckets': dict(counts),
               'name_positive': name_positive, 'name_negative': count - name_positive, 'seed': seed,
               'foreground_source': source.relative_to(ROOT).as_posix(),
               'eligible_source_count_by_bucket': {str(k): len(v) for k, v in groups.items()},
               'source_exclusions': dict(skipped), 'background_manifest': (BACKGROUND / 'manifest.json').relative_to(ROOT).as_posix(),
               'background_review': 'six frames visually inspected; background targets and reflections remain review candidates',
               'source_group': ['scene_video_49s_shared', 'blender_v13_shared_model'],
               'usage': 'training candidates only, no independent validation/test; generated text is not OCR truth'}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def produce(source, output, count, seed):
    manifest = source / 'manifest.jsonl'
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    media = [BACKGROUND / 'images' / f'{i:04}.png' for i in range(1, 7)] + [ICON]
    for row in rows:
        if not row.get('fire_label') or not row.get('name_label') or row.get('occluder'):
            continue
        if any(obj['state'] not in ('visible', 'outside_frame') for obj in row['instances']):
            continue
        media.extend([source / row['image'], source / row['instance_pass']])
    metadata = [source / 'run.json', manifest]
    inputs = list(dict.fromkeys([*metadata, *media]))
    with generation_guard('fire.compose_scene:' + str(output.resolve().relative_to(ROOT)),
                          inputs, [output], {'count': count, 'seed': seed},
                          code_paths=[Path(__file__), ROOT / 'lab/features/fire/training/scripts/fire_blender_labels.py',
                                      ROOT / 'datasets/scripts/common/images.py', ROOT / 'datasets/scripts/common/annotations.py'],
                          procedural_inputs=metadata, root=ROOT) as run:
        if run.reused:
            return json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        result = _produce(source, output, count, seed)
        result['batch_source_refs'] = run.source_refs
        (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1919)
    args = parser.parse_args()
    print(json.dumps(produce(DEFAULT_SOURCE, (ROOT / args.output).resolve(), args.count, args.seed), ensure_ascii=False))
