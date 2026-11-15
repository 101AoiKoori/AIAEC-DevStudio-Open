"""Run a bounded prior-only inspection using three locally reviewed facade crops."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/fire/. Production remains paused.")

import copy
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard, plan_production
from generate_fire_dataset import read_image, write_image
from inspect_real_facade_crops import ARTWORK, CROP, PATCHES
from produce_integrated_fire import generate_for_state, source_state_plan

EVIDENCE = ROOT / 'runs/fire_sticker/real_facade_bounded_inspection_v3_20261114/crop_evidence.json'
OUTPUT = ROOT / 'runs/fire_sticker/facade_crop_bayesian_inspection_20261114'
SEED = 431
BUDGET = 18


def load_entries():
    records = json.loads(EVIDENCE.read_text(encoding='utf8'))
    if len(records) != 3 or {r['source_id'] for r in records} != set(PATCHES):
        raise ValueError('three reviewed facade crops are required')
    entries, parents = [], []
    for r in records:
        source_id = r['source_id']
        if (r['crop_xyxy'] != list(CROP) or
                r['fire_sticker'] != 'no_recognizable_target_in_crop_candidate' or
                r['building_name_paper'] != 'no_recognizable_target_in_crop_candidate'):
            raise ValueError('crop review status or bounds differ from inspected evidence')
        expected_quad = [[round(x * (CROP[2]-CROP[0]) / 600, 2),
                          round(y * (CROP[3]-CROP[1]) / 450, 2)] for x, y in PATCHES[source_id]]
        if r['wall_patch_xy'] != expected_quad:
            raise ValueError('wall patch differs from the inspected region')
        parent, crop = (ROOT / r['parent']).resolve(), (ROOT / r['crop']).resolve()
        if (not parent.is_relative_to(ROOT / 'datasets/raw/fire') or
                not crop.is_relative_to(ROOT / 'runs/fire_sticker') or
                not parent.is_file() or not crop.is_file()):
            raise ValueError('parent or crop missing or outside its source scope')
        raw, actual_crop = read_image(parent), read_image(crop)
        if raw.shape[:2] != (4640, 3472) or not np.array_equal(actual_crop, raw[CROP[1]:CROP[3], CROP[0]:CROP[2]]):
            raise ValueError('crop does not match the declared parent pixels')
        entries.append({'image': r['crop'], 'quads': [expected_quad], 'supported_counts': [0, 1],
            'review_state': 'ai_reviewed_empty_candidate', 'view': 'bounded_facade',
            'material_type': 'derived_crop_of_camera_photo', 'crop_xyxy': None,
            'parent': r['parent'], 'source_id': source_id})
        parents.append(parent)
    return entries, parents


def make_plan(entries):
    plan = source_state_plan(BUDGET, SEED, entries)
    for state in plan['strata']:
        state['min_count'], state['max_count'] = 1, 7
        state['augmentation']['tone'] = {'contrast_uniform': [0.9, 1.1],
                                          'brightness_uniform': [-8.0, 8.0]}
    plan['purpose'] = 'prior-only bounded crop candidate inspection; not a training release'
    plan['independence_basis'] = 'three parent photo paths; capture session independence unknown'
    return plan


def main():
    entries, parents = load_entries()
    if OUTPUT.exists():
        raise ValueError('output already exists')
    declaration = make_plan(entries)
    plan = plan_production(declaration, ROOT)
    equal = copy.deepcopy(declaration)
    for state in equal['strata']:
        state['prior'] = {'alpha': 1, 'beta': 1}
        state['exposure_weight'] = state['failure_cost'] = 1
    baseline = plan_production(equal, ROOT)
    if [s['planned_count'] for s in plan['strata']] == [s['planned_count'] for s in baseline['strata']]:
        raise ValueError('allocation did not respond to changed prior and costs')
    inputs = [EVIDENCE, ARTWORK, *parents, *(ROOT / e['image'] for e in entries)]
    with generation_guard('fire.facade_crop_bayesian:' + OUTPUT.relative_to(ROOT).as_posix(),
            inputs, [OUTPUT], {'seed': SEED, 'budget': BUDGET},
            code_paths=[Path(__file__), ROOT / 'lab/features/fire/training/scripts/produce_integrated_fire.py',
                        ROOT / 'lab/features/fire/training/scripts/generate_fire_dataset.py',
                        ROOT / 'lab/features/fire/training/scripts/inspect_real_facade_crops.py'],
            procedural_inputs=[EVIDENCE], production_plan=declaration,
            sample_output_paths=[OUTPUT / 'samples'], root=ROOT) as run:
        if run.reused:
            return
        OUTPUT.mkdir(parents=True)
        (OUTPUT / 'samples').mkdir()
        (OUTPUT / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf8')
        (OUTPUT / 'equal_weight_allocation.json').write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding='utf8')
        schedule = [s for s in run.plan['strata'] for _ in range(s['planned_count'])]
        random.Random(SEED).shuffle(schedule)
        by_path = {e['image']: e for e in entries}
        with (OUTPUT / 'manifest.jsonl').open('w', encoding='utf8') as log:
            for i, state in enumerate(schedule):
                sample_seed = SEED + i
                source_file = OUTPUT / ('source_%03d.json' % i)
                sample_dir = OUTPUT / 'samples' / ('sample_%03d' % i)
                entry = generate_for_state(state, by_path, sample_seed, source_file, ARTWORK, sample_dir, None)
                generated = json.loads((sample_dir / 'manifest.json').read_text(encoding='utf8'))[0]
                image = sample_dir / 'images/train/fire_000001.png'
                tone_rng = random.Random(sample_seed)
                contrast, brightness = tone_rng.uniform(.9, 1.1), tone_rng.uniform(-8, 8)
                enhanced = np.clip(read_image(image).astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
                write_image(image, enhanced)
                name_dir = sample_dir / 'labels/name_candidate'
                name_dir.mkdir(parents=True)
                (name_dir / 'fire_000001.txt').write_text('', encoding='utf8')
                actual = {'source_id': entry['source_id'], 'parent_photo': entry['parent'],
                    'crop': entry['image'], 'review_state': entry['review_state'],
                    'background_name_review': 'no_recognizable_target_in_crop_candidate',
                    'target_count': state['target']['count'], 'wall_quad': entry['quads'][0],
                    'projected': generated['projected'], 'sample_seed': sample_seed,
                    'tone': {'contrast': contrast, 'brightness': brightness},
                    'generation_method': 'real_photo_crop_plus_perspective_digital_artwork',
                    'status': 'candidate_requires_review'}
                run.record_sample(state['id'], output=image, sources=state['sources'],
                                  parameters=state['parameters'], actual=actual)
                log.write(json.dumps({'state_id': state['id'], 'image': image.relative_to(ROOT).as_posix(),
                    'fire_label': (sample_dir / 'labels/train/fire_000001.txt').relative_to(ROOT).as_posix(),
                    'name_label': (name_dir / 'fire_000001.txt').relative_to(ROOT).as_posix(), **actual},
                    ensure_ascii=False) + '\n')
                log.flush()
        run.set_result({'samples': len(schedule), 'status': 'candidate_requires_review'})
    print('FACADE_CROP_INSPECTION', BUDGET, OUTPUT)


if __name__ == '__main__':
    main()
