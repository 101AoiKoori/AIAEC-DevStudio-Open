"""Compose source-recorded bike cutouts onto rendered low-camera arena views.

The selected support points are inferred from real source pictures; the scene
camera and physical dimensions are fitted, not measured vehicle calibration.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/e_bike/.")

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common'), str(HERE)]
import preview_person_occlusions as assets  # noqa: E402
from datasets.scripts.common.annotations import class_yolo_row  # noqa: E402
from compose_v4_multibike_parking import clean_rgba  # noqa: E402
from camera_tone import apply_tone, read_image, write_image  # noqa: E402
from lab.training_common.production_guard import generation_guard  # noqa: E402
from datasets.scripts.common.composite import visible_masks  # noqa: E402

BACKGROUND = ROOT / 'datasets/exports/e_bike/v4_vehicle_view_backgrounds_20261113/manifest.json'
SELECTION = ROOT / 'datasets/annotated/e_bike/sam2_orbit_candidates_20261110/selection_review_candidate.json'
REVIEW = ROOT / 'datasets/exports/e_bike/v4_production_candidate_20261113/positive_parent_review_20261113.json'
MASTER = ROOT / 'datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend'
# Regions identify wheels in the side-view photos, or the fallen handlebar
# and basket in its actual source photo. These are estimated, not measured.
PROFILES = {
    'source_002_frame_000006': {'class': 'upright', 'bands': ((0.0, .32), (.68, 1.0)), 'span_m': .20,
                               'contact_parts': ['front_wheel', 'rear_wheel']},
    'source_002_frame_000007': {'class': 'upright', 'bands': ((0.0, .32), (.68, 1.0)), 'span_m': .20,
                               'contact_parts': ['front_wheel', 'rear_wheel']},
    'sam2_01_fallen': {'class': 'fallen', 'bands': ((0.0, .20), (.25, .64)), 'span_m': .09,
                       'contact_parts': ['handlebar', 'basket']},
}
TONES = {'normal': (1.0, 1.0), 'dim': (.78, .86), 'low_saturation': (.89, .62)}
SPLITS = {'train': (0, 160, (0, 1, 2, 3)), 'val': (160, 200, (4, 5)),
          'test': (200, 240, (6, 7))}


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def project(camera, point):
    """Project a world point with the master camera's verified 36 mm sensor."""
    loc = np.asarray(camera['camera_location_m'], dtype=float)
    target = np.asarray(camera['camera_target_m'], dtype=float)
    forward = target - loc
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0., 0., 1.]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    vector = np.asarray(point, dtype=float) - loc
    depth = float(np.dot(vector, forward))
    if depth <= .05:
        raise ValueError('point behind or too near camera')
    focal = 640 * float(camera['camera_lens_mm']) / 36.0
    return np.array([320 + focal * np.dot(vector, right) / depth,
                     240 - focal * np.dot(vector, up) / depth])


def support_points(rgba, bands):
    """Use the two explicitly selected visible contact regions, not whole-image minima."""
    h, w = rgba.shape[:2]
    points = []
    for left, right in bands:
        start, end = int(w * left), max(int(w * right), int(w * left) + 1)
        ys, xs = np.nonzero(rgba[:, start:end, 3] > 127)
        if len(xs) < 20:
            raise ValueError('selected contact region has insufficient alpha')
        cutoff = np.percentile(ys, 99.5)
        points.append([float(start + np.median(xs[ys >= cutoff])), float(cutoff)])
    if points[1][0] - points[0][0] < w * .15:
        raise ValueError('contact regions do not span subject')
    return np.asarray(points, dtype=np.float32)


def similarity(source, target):
    """Map the pair of inferred contacts onto projected ground contacts."""
    delta, desired = source[1] - source[0], target[1] - target[0]
    denom = float(np.dot(delta, delta))
    if denom <= 1e-6:
        raise ValueError('invalid support separation')
    a = float(np.dot(delta, desired) / denom)
    b = float((delta[0] * desired[1] - delta[1] * desired[0]) / denom)
    linear = np.array([[a, -b], [b, a]], dtype=np.float32)
    translation = target[0] - linear @ source[0]
    return np.column_stack((linear, translation)).astype(np.float32)


def transformed_vehicle(rgba, profile, camera, x, depth, mirrored):
    sprite = np.ascontiguousarray(rgba[:, ::-1]) if mirrored else rgba
    bands = profile['bands'] if not mirrored else tuple((1 - b, 1 - a) for a, b in profile['bands'][::-1])
    source = support_points(sprite, bands)
    floor = float(camera['floor_top_m'])
    y = float(camera['camera_location_m'][1]) + depth
    half = profile['span_m'] / 2
    world = [[x - half, y, floor], [x + half, y, floor]]
    target = np.asarray([project(camera, p) for p in world], dtype=np.float32)
    matrix = similarity(source, target)
    warped = cv2.warpAffine(sprite, matrix, (640, 480), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    mask = warped[:, :, 3] > 127
    ys, xs = np.nonzero(mask)
    if (len(xs) < 120 or xs.min() <= 1 or xs.max() >= 638 or ys.min() <= 1 or ys.max() >= 478 or
            not np.all((target[:, 0] >= 2) & (target[:, 0] < 638) &
                       (target[:, 1] >= 242) & (target[:, 1] < 470))):
        raise ValueError('candidate is too small, clipped or outside ground')
    return {'rgba': warped, 'mask': mask, 'contacts': target, 'world_contacts': world,
            'source_contacts': source, 'matrix': matrix, 'depth': depth, 'mirrored': mirrored}


def shadow(canvas, contacts):
    layer = np.zeros(canvas.shape[:2], np.uint8)
    for x, y in contacts:
        cv2.ellipse(layer, (round(float(x)), round(float(y))), (9, 3), 0, 0, 360, 190, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), 2.4).astype(np.float32) / 255
    canvas[:] = np.clip(canvas.astype(np.float32) * (1 - .32 * layer[:, :, None]), 0, 255).astype(np.uint8)


def paste(canvas, rgba):
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255
    canvas[:] = np.clip(rgba[:, :, :3] * alpha + canvas * (1 - alpha), 0, 255).astype(np.uint8)
    return rgba[:, :, 3] > 127


def person_rgba(sprite, camera, x, depth):
    floor = float(camera['floor_top_m'])
    y = float(camera['camera_location_m'][1]) + depth
    foot = project(camera, [x, y, floor])
    head = project(camera, [x, y, floor + .24])
    height = max(25, int(round(foot[1] - head[1])))
    resized = assets.resized(sprite, height=height)
    matrix = np.array([[1, 0, foot[0] - resized.shape[1] / 2],
                       [0, 1, foot[1] - resized.shape[0]]], dtype=np.float32)
    image = cv2.warpAffine(resized, matrix, (640, 480), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return image, foot


def source_inputs():
    backgrounds = json.loads(BACKGROUND.read_text(encoding='utf8'))
    source = json.loads(SELECTION.read_text(encoding='utf8'))
    review = json.loads(REVIEW.read_text(encoding='utf8'))
    if len(backgrounds['records']) != 8 or backgrounds.get('source_scene') != relative(MASTER):
        raise ValueError('unexpected background batch or mother scene')
    if set(PROFILES) & set(review['excluded_foreground_stems']):
        raise ValueError('a selected foreground has known detached matte')
    entries = {Path(item['path']).stem: item for item in source['vehicles'] if Path(item['path']).stem in PROFILES}
    if set(entries) != set(PROFILES) or any(entries[key]['class'] != profile['class'] for key, profile in PROFILES.items()):
        raise ValueError('selected source views are unavailable or have wrong pose')
    people = source['people']
    if not people:
        raise ValueError('person occluder sources missing')
    media = [ROOT / row['image'] for row in backgrounds['records']]
    media += [ROOT / entry[key] for entry in entries.values() for key in ('path', 'source_frame', 'source_video')]
    media += [ROOT / item['path'] for item in people]
    return backgrounds, entries, people, list(dict.fromkeys(media))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=2026111304)
    parser.add_argument('--compact-existing', action='store_true')
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    owner = (ROOT / 'datasets/exports/e_bike').resolve()
    if not out.is_relative_to(owner) or out == owner:
        parser.error('output must be an e-bike export subdirectory')
    if args.compact_existing:
        compact_existing(out)
        return
    raise RuntimeError('e-bike arena generation is disabled: explicit planned-state execution is not implemented')
    backgrounds, entries, people, media = source_inputs()
    procedural = [BACKGROUND, SELECTION, REVIEW, MASTER]
    code = [Path(__file__), HERE / 'preview_person_occlusions.py',
            HERE / 'compose_v4_multibike_parking.py',
            HERE / 'synthesize_vehicle_scene.py', ROOT / 'lab/scripts/data/v4_source_gate.py',
            ROOT / 'datasets/scripts/common/annotations.py',
            ROOT / 'lab/training_common/camera_tone.py', ROOT / 'lab/training_common/production_guard.py',
            ROOT / 'datasets/scripts/common/images.py', ROOT / 'datasets/scripts/common/perspective.py', ROOT / 'datasets/scripts/common/composite.py', ROOT / 'datasets/scripts/common/annotations.py']
    params = {'seed': args.seed, 'splits': {k: [v[0], v[1]] for k, v in SPLITS.items()},
              'profiles': PROFILES, 'tones': TONES, 'camera_sensor_width_mm': 36.0}
    with generation_guard('e_bike.compose_vehicle_view_arena:' + relative(out), media, [out], params,
                          procedural_inputs=procedural, code_paths=code, root=ROOT) as run:
        if run.reused:
            return
        if out.exists():
            raise ValueError('output exists; choose a new directory')
        produce(out, args.seed, backgrounds, entries, people, run.source_refs)


def compact_refs(refs):
    relevant = [ref for ref in refs if ref.get('relation') == 'describes']
    if not relevant and refs:
        relevant = refs[:1]
    return [{key: ref[key] for key in ('manifest', 'record_index', 'relation') if key in ref}
            for ref in relevant[:1]]


def compact_existing(out):
    expected = ROOT / 'datasets/exports/e_bike/v4_vehicle_view_arena_20261113'
    if out != expected.resolve():
        raise ValueError('compaction restricted to the existing candidate manifest')
    path = out / 'manifest.json'
    with generation_guard('e_bike.compact_vehicle_view_manifest', [], [path],
                          {'scope': 'remove redundant reverse source references', 'version': 1},
                          procedural_inputs=[path], inplace_paths=[path], code_paths=[Path(__file__)],
                          root=ROOT) as run:
        if run.reused:
            return
        data = json.loads(path.read_text(encoding='utf8'))
        if len(data['records']) != 240:
            raise ValueError('unexpected candidate count')
        assets_before = {}
        for row in data['records']:
            assets = [row['image'], row['label']] + [obj['visible_mask'] for obj in row['objects']]
            for name in assets:
                stat = (ROOT / name).stat()
                assets_before[name] = (stat.st_size, stat.st_mtime_ns)
            for key in ('background_source_refs', 'person_source_refs'):
                row[key] = compact_refs(row.get(key, []))
            for obj in row['objects']:
                obj['foreground_source_refs'] = compact_refs(obj.get('foreground_source_refs', []))
        encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n'
        if len(encoded.encode('utf8')) >= 64 * 1024 * 1024:
            raise ValueError('manifest still exceeds index limit')
        for name, state in assets_before.items():
            stat = (ROOT / name).stat()
            if (stat.st_size, stat.st_mtime_ns) != state:
                raise ValueError('image, label or mask changed during compaction: ' + name)
        path.write_text(encoded, encoding='utf8')
        run.set_result({'records': len(data['records']), 'checked_assets': len(assets_before),
                        'manifest_bytes': path.stat().st_size})
    print(json.dumps({'records': len(data['records']), 'manifest_bytes': path.stat().st_size}))


def produce(out, seed, backgrounds, entries, people, refs):
    raise RuntimeError('e-bike arena generation is disabled: explicit planned-state execution is not implemented')
    sprites = {key: clean_rgba(assets.load_rgba(ROOT / item['path'])) for key, item in entries.items()}
    persons = [(item, assets.load_rgba(ROOT / item['path'])) for item in people]
    records, counts = [], Counter()
    for split in SPLITS:
        for part in ('images', 'labels', 'masks'):
            (out / part / split).mkdir(parents=True, exist_ok=True)
    for split, (start, stop, background_indices) in SPLITS.items():
        for index in range(start, stop):
            rng = random.Random(seed + index)
            bg = backgrounds['records'][background_indices[(index - start) % len(background_indices)]]
            base = read_image(ROOT / bg['image'])
            if base.shape != (480, 640, 3):
                raise ValueError('expected 640x480 Blender background')
            plan = ('negative', 'negative', 'upright', 'upright', 'upright',
                    'fallen', 'fallen', 'fallen', 'mixed', 'mixed')[index % 10]
            classes = {'negative': [], 'upright': ['upright'], 'fallen': ['fallen'],
                       'mixed': ['upright', 'fallen']}[plan]
            if classes and index % 7 == 0:
                classes = classes + [classes[-1]]
            background_base = base
            for layout_attempt in range(40):
                base = background_base.copy()
                actors = []
                for number, category in enumerate(classes):
                    names = [key for key, profile in PROFILES.items() if profile['class'] == category]
                    name = rng.choice(names)
                    for attempt in range(80):
                        x = rng.uniform(-.34, .34) + (number - (len(classes) - 1) / 2) * .13
                        depth = rng.uniform(.68, 1.28) + number * .045
                        try:
                            actor = transformed_vehicle(sprites[name], PROFILES[name], bg, x, depth, rng.choice((False, True)))
                            actor.update({'name': name, 'class': category, 'source': entries[name], 'x': x})
                            actors.append(actor)
                            break
                        except ValueError:
                            continue
                    else:
                        raise ValueError('could not place valid projected vehicle: ' + str(index))
                actors.sort(key=lambda item: item['depth'], reverse=True)
                for actor in actors:
                    shadow(base, actor['contacts'])
                masks = [paste(base, actor['rgba']) for actor in actors]
                use_person = plan == 'negative' or (classes and index % 4 == 0)
                person = None
                if use_person:
                    item, sprite = rng.choice(persons)
                    near = actors[-1] if actors else None
                    depth = max(.55, near['depth'] - .07) if near else rng.uniform(.65, 1.25)
                    x = near['x'] + rng.uniform(-.04, .04) if near else rng.uniform(-.35, .35)
                    person, person_foot = person_rgba(sprite, bg, x, depth)
                    shadow(base, [person_foot])
                    masks.append(paste(base, person))
                shown = visible_masks(masks)
                if any(mask.sum() < 120 or shown[i].sum() / mask.sum() < .58
                       for i, mask in enumerate(masks[:len(actors)])):
                    continue
                break
            else:
                raise ValueError('could not place unambiguous layout: ' + str(index))
            tone = tuple(TONES)[(index // 10) % 3]
            gain, saturation = TONES[tone]
            if tone != 'normal':
                base = apply_tone(base, value_gain=gain, saturation_gain=saturation)
            stem = f'arena_bike_{index:04d}'
            image = out / 'images' / split / (stem + '.png')
            label = out / 'labels' / split / (stem + '.txt')
            write_image(image, base)
            labels, objects = [], []
            for number, actor in enumerate(actors):
                visible = shown[number]
                yy, xx = np.nonzero(visible)
                box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
                labels.append(class_yolo_row({'upright': 0, 'fallen': 1}[actor['class']], box, 640, 480))
                mask_path = out / 'masks' / split / f'{stem}_{number:02d}.png'
                write_image(mask_path, visible.astype(np.uint8) * 255)
                objects.append({'class': actor['class'], 'bbox_xyxy': box, 'visible_mask': relative(mask_path),
                    'visible_fraction': float(visible.sum() / masks[number].sum()),
                    'foreground_source': actor['source'],
                    'foreground_source_refs': compact_refs(refs.get(actor['source']['path'], [])),
                    'estimated_source_contact_xy': actor['source_contacts'].tolist(),
                    'projected_contact_xy': actor['contacts'].tolist(),
                    'fitted_world_contact_m': actor['world_contacts'],
                    'source_to_image_affine': actor['matrix'].tolist(),
                    'contact_parts': (PROFILES[actor['name']]['contact_parts'][::-1] if actor['mirrored']
                                      else PROFILES[actor['name']]['contact_parts']),
                    'mirror_horizontal': actor['mirrored'],
                    'contact_limit': 'visible source regions visually inferred; not physical measurement'})
            label.write_text('\n'.join(labels) + ('\n' if labels else ''), encoding='utf8')
            records.append({'image': relative(image), 'label': relative(label), 'split': split,
                'image_size': [640, 480], 'plan': plan, 'tone': tone,
                'tone_parameters': {'value_gain': gain, 'saturation_gain': saturation},
                'generation_method': 'blender_render_then_real_cutout_composite',
                'background_origin': 'blender', 'mother_scene': relative(MASTER),
                'background': {'kind': 'blender', 'source_group': relative(MASTER),
                               'view_group': Path(bg['image']).stem},
                'background_image': bg['image'], 'background_camera': bg,
                'background_source_refs': compact_refs(refs.get(bg['image'], [])), 'objects': objects,
                'person_source': item if person is not None else None,
                'person_source_refs': compact_refs(refs.get(item['path'], [])) if person is not None else [],
                'person_vehicle_overlap': bool(person is not None and any(np.any(mask & masks[-1]) for mask in masks[:len(actors)])),
                'label_review': 'generated visible mask; candidate pending representative visual inspection'})
            counts[split] += 1
            counts[split + '_' + plan] += 1
            counts[split + '_' + tone] += 1
    summary = {split: {'blender_count': counts[split], 'total_count': counts[split],
                       'blender_fraction': 1.0} for split in SPLITS}
    output = {'status': 'candidate_QA_pending', 'seed': seed, 'counts': dict(counts),
              'split_background_distribution': summary, 'mother_scene': relative(MASTER),
              'generation_method': 'blender_render_then_real_cutout_composite',
              'source_reuse': {'background_scene_shared': True, 'foreground_physical_instance': 'unknown',
                               'foreground_source_videos': sorted({item['source_video'] for item in entries.values()}),
                               'rendered_backgrounds_by_split': {k: list(v[2]) for k, v in SPLITS.items()},
                               'independent_capture': False},
              'background_distribution': {'blender_count': sum(counts[s] for s in SPLITS),
                                          'total_count': sum(counts[s] for s in SPLITS)},
              'scope': 'Fitted camera and selected real cutouts; support landmarks inferred from source photos. '
                       'No physical contact measurement, independent acquisition, or evaluation. '
                       'Foreground vehicles are kept within frame; near-edge truncation is not covered.', 'records': records}
    (out / 'manifest.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'output': relative(out), 'counts': dict(counts)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
