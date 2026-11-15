"""Compose labelled e-bike training candidates on reviewed 640x480 backgrounds."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/e_bike/. Production remains paused.")

import argparse
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT/'lab/training_common'), str(Path(__file__).resolve().parent)]
from camera_tone import apply_tone, read_image, source_role, write_image  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402
import synthesize_vehicle_scene as bike  # noqa: E402
from datasets.scripts.common.composite import alpha_place, visible_masks  # noqa: E402

BACKGROUND = ROOT/'datasets/annotated/e_bike/v4_complex_backgrounds_20261112/backgrounds.json'
OUT = ROOT/'datasets/exports/e_bike/v4_complex_training_candidates_20261112'
SEED = 20261115
SIZE = {'small': (95, 125), 'medium': (155, 190), 'large': (220, 255)}
TONES = {'original': (1., 1.), 'mild': (.9, .85), 'moderate': (.7, .6)}


def rel(path):
    return path.relative_to(ROOT).as_posix()


def scenario(slot):
    if slot >= 16:
        return 'negative', []
    count = (1, 2, 3, 4, 1, 2, 3, 4)[slot % 8]
    if slot < 8:
        return 'upright_only', ['upright'] * count
    return 'fallen_present', ['fallen' if i % 2 == 0 else 'upright' for i in range(count)]


def _produce():
    if OUT.exists():
        raise ValueError('Output exists; choose a new version')
    backgrounds = json.loads(BACKGROUND.read_text(encoding='utf-8'))['backgrounds']
    if len(backgrounds) != 4 or any(b['task_state'] != 'candidate_complete_negative' for b in backgrounds):
        raise ValueError('Expected four task-reviewed negative backgrounds')
    known = build_queue(ROOT)
    for bg in backgrounds:
        if source_role(ROOT, ROOT/bg['parent_image'], known)[0] == 'regression':
            raise ValueError('Background restricted to regression: '+bg['parent_image'])
    _, vehicles = bike.sources(include_people=False)
    scene = bike.scene
    counts, skipped, records = Counter(), Counter(), []
    try:
        (OUT/'images/train').mkdir(parents=True)
        (OUT/'labels/train').mkdir(parents=True)
        (OUT/'masks').mkdir()
        for bg_index, slot, tone, size in itertools.product(range(4), range(20), TONES, SIZE):
            name, classes = scenario(slot)
            # Empty images are identical across the irrelevant size/slot axes.
            if not classes and (slot != 16 or size != 'small'):
                continue
            # Crowded large sprites cannot plausibly fit this 640px scene.
            if len(classes) >= 3 and size == 'large':
                skipped['infeasible_large_crowd'] += 1
                continue
            bg = backgrounds[bg_index]
            seed = SEED + bg_index*10000 + slot*100 + list(TONES).index(tone)*10 + list(SIZE).index(size)
            rng = random.Random(seed)
            success = None
            for attempt in range(20):
                frame = read_image(ROOT/bg['image'])
                if frame.shape != (480, 640, 3):
                    raise ValueError('Expected 640x480 background')
                try:
                    actors = []
                    for category in classes:
                        source, sprite = rng.choice(vehicles[category])
                        height = rng.uniform(*SIZE[size])
                        x = rng.uniform(*bg['anchor_x'])
                        foot = rng.uniform(385, 462)
                        actors.append(scene.random_actor(rng, 'vehicle', category, sprite, source, x, foot, height))
                    actors.sort(key=lambda a: a['foot_y'])
                    for actor in actors:
                        scene.draw_shadow(frame, actor)
                    masks = [alpha_place(frame, a['sprite'], a['foot_x'], a['foot_y']) for a in actors]
                    visible = visible_masks(masks)
                    if any(v.sum() < 120 or v.sum()/m.sum() < .65 for m, v in zip(masks, visible)):
                        raise ValueError('target too occluded')
                    success = frame, actors, visible, attempt+1
                    break
                except ValueError:
                    continue
            if success is None:
                skipped['placement_or_visibility'] += 1
                continue
            frame, actors, masks, attempts = success
            value, saturation = TONES[tone]
            if tone != 'original':
                frame = apply_tone(frame, value_gain=value, saturation_gain=saturation)
            stem = f'bike_{len(records)+1:06d}'
            image_out, label_out = OUT/'images/train'/(stem+'.png'), OUT/'labels/train'/(stem+'.txt')
            write_image(image_out, frame)
            labels, objects = [], []
            for index, (actor, mask) in enumerate(zip(actors, masks)):
                y, x = np.nonzero(mask)
                box = [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1]
                cls = {'upright': 0, 'fallen': 1}[actor['category']]
                labels.append(scene.yolo_row(cls, box, 640, 480))
                mask_out = OUT/'masks'/f'{stem}_{index:02d}.png'
                write_image(mask_out, mask.astype(np.uint8)*255)
                objects.append({'class': actor['category'], 'bbox_xyxy': box, 'visible_mask': rel(mask_out),
                    'foreground_source': actor['source'], 'angles_degrees': actor['angles_degrees'],
                    'foot_xy': [actor['foot_x'], actor['foot_y']]})
                counts['instances_'+actor['category']] += 1
            label_out.write_text(('\n'.join(labels)+'\n') if labels else '', encoding='utf-8')
            records.append({'image': rel(image_out), 'label': rel(label_out), 'split': 'train_candidate',
                'background': bg, 'scenario': name, 'objects': objects, 'size_band': size, 'tone': tone,
                'tone_parameters': {'value_gain': value, 'saturation_gain': saturation}, 'attempts': attempts,
                'image_size': [640, 480], 'generation_method': 'real_scene_composite',
                'label_review': 'visible_mask_projection_candidate_batch_QA_pending'})
            counts['images_'+name] += 1
            counts['tone_'+tone] += 1
        manifest = {'status': 'held_scene_geometry', 'seed': SEED,
            'scope': 'Four background frames from one site video and shared vehicle sprites. '
                     'No independent capture, validation/test, or model result.',
            'background_manifest': rel(BACKGROUND), 'background_parent_count': len(backgrounds),
            'counts': dict(counts), 'skipped': dict(skipped), 'records': records}
        (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'output': rel(OUT), 'counts': dict(counts), 'skipped': dict(skipped)}, ensure_ascii=False))
    except BaseException:
        (OUT/'FAILED.txt').write_text('Generation failed; inspect before reuse.\n', encoding='utf-8')
        raise


def main():
    from lab.training_common.production_guard import generation_guard
    from e_bike_queue_inputs import queue_inputs
    from lab.scripts.data import v4_source_gate as gate
    parser = argparse.ArgumentParser(description=__doc__)

    opts = parser.parse_args()
    out = OUT.resolve()
    manifests = [BACKGROUND] + [bike.SELECTION]
    people, vehicles = bike.sources(include_people=False)
    media = set()
    for group in people.values():
        media.update(Path(source).resolve() for source, _ in group)
    for group in vehicles.values():
        for source, _ in group:
            for key in ('path', 'source_frame', 'source_video'):
                media.add((ROOT / source[key]).resolve())
    suffixes = {'.png','.jpg','.jpeg','.bmp','.webp','.ppm','.mp4','.mov','.avi','.mkv'}
    def discover(value):
        if isinstance(value, dict):
            for child in value.values(): discover(child)
        elif isinstance(value, list):
            for child in value: discover(child)
        elif isinstance(value, str) and Path(value).suffix.lower() in suffixes:
            path = (ROOT / value).resolve()
            if path.is_relative_to(ROOT) and path.is_file(): media.add(path)
    for path in manifests:
        discover(json.loads(path.read_text(encoding='utf-8')))
    for row in json.loads(BACKGROUND.read_text(encoding='utf-8'))['backgrounds']:
        media.update((ROOT / row[key]).resolve() for key in ('image', 'parent_image'))
    queue = build_queue(ROOT)
    qm, qp = queue_inputs(ROOT, queue, media)
    media.update(qm)
    procedural = set(qp) | set(manifests)
    code = {Path(__file__).resolve(), Path(bike.__file__).resolve(), Path(bike.scene.__file__).resolve(), Path(bike.scene.assets.__file__).resolve(),
            Path(gate.__file__).resolve(), ROOT / 'lab/scripts/media/media_index.py', Path(__file__).resolve().parent / 'e_bike_queue_inputs.py',
            ROOT / 'lab/training_common/camera_tone.py', ROOT / 'datasets/scripts/common/perspective.py', ROOT / 'datasets/scripts/common/composite.py', ROOT / 'datasets/scripts/common/annotations.py',
            ROOT / 'lab/training_common/engine/data.py', ROOT / 'lab/training_common/engine/config.py',
            ROOT / 'lab/training_common/train.py', ROOT / 'lab/training_common/production_guard.py',
            Path(__file__).resolve().parent / 'preview_person_occlusions.py',
            Path(__file__).resolve().parent / 'compose_v4_multibike_parking.py'}
    params = {key: str(value) if isinstance(value, Path) else value for key, value in vars(opts).items()}
    with generation_guard('e_bike.compose_v4_complex_train:' + out.relative_to(ROOT).as_posix(), sorted(media), [out], params,
                          procedural_inputs=sorted(procedural), code_paths=sorted(code), root=ROOT) as run:
        if run.reused:
            return
        return _produce()


if __name__ == '__main__':
    main()
