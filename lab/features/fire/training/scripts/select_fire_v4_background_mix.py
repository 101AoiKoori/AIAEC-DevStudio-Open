"""Select the fire V4 background mix from already reviewed source records."""
import copy
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/scripts/data')]
from lab.training_common.production_guard import generation_guard
import v4_source_gate as gate

BASE = 'datasets/exports/fire/v4_integrated_candidates_20261112/fire_sticker'
SOURCE = ROOT / BASE / 'source_manifest.json'
DESCRIPTOR = ROOT / BASE / 'distribution_manifest.candidate.json'
CONFIG = ROOT / 'lab/features/fire/training/config/fire_sticker_v4_delivery_20261113.yaml'
ARENA = ROOT / 'datasets/exports/fire/fire_sticker_v4_arena_v4_train_20261113/manifest.json'
OUTPUT = ROOT / 'datasets/exports/fire/fire_sticker_v4_background_mix_20261113'


def select(data, arena):
    rows = [r for r in data['samples'] if r.get('split') in ('train', 'val', 'test')
            and 'fire_sticker' in r.get('selected_tasks', [])]
    by_image = {r['image']: r for r in rows}
    views = {}
    pairs = defaultdict(list)
    for record in arena['records']:
        image = record['image']
        if image not in by_image:
            raise ValueError('arena record missing from source manifest: ' + image)
        rotation = tuple(round(float(x), 3) for x in record['camera']['rotation_euler_rad'])
        views[image] = 'camera_rotation_' + '_'.join(str(x) for x in rotation)
        index = int(Path(record['source_image']).stem.rsplit('_', 1)[1])
        pairs[(record['source_batch'], index // 2)].append(by_image[image])
    if len(views) != 210 or any(len(v) != 2 for v in pairs.values()):
        raise ValueError('expected 210 arena records in 105 normal/dark pairs')
    rng = random.Random(20261113)
    keys = sorted(pairs)
    rng.shuffle(keys)
    assignments = {}
    for i, key in enumerate(keys):
        split = 'val' if i < 5 else 'test' if i < 10 else 'train'
        for row in pairs[key]:
            assignments[row['id']] = split
    real = [r for r in rows if r['image'].startswith(BASE + '/images/')]
    for row in real:
        if row['split'] in ('val', 'test'):
            assignments[row['id']] = row['split']
    buckets = defaultdict(list)
    for row in real:
        if row['split'] != 'train':
            continue
        task = row['tasks']['fire_sticker']
        conditions = task.get('strata', {}).get('conditions', [])
        buckets[(row.get('parent_video'), task['state'], 'low_light' in conditions)].append(row)
    # Sample proportionately within each real-video/state/light stratum.
    candidates = []
    for key, items in sorted(buckets.items()):
        rng.shuffle(items)
        candidates.extend((i / len(items), key, row) for i, row in enumerate(items))
    candidates.sort(key=lambda x: (x[0], x[1]))
    for _, _, row in candidates[:215]:
        assignments[row['id']] = 'train'
    if Counter(assignments.values()) != {'train': 405, 'val': 60, 'test': 60}:
        raise ValueError('unexpected selection counts: ' + str(Counter(assignments.values())))
    return assignments, views


def main():
    if OUTPUT.exists():
        raise ValueError('selection output already exists; inspect its receipt before reuse')
    data = json.loads(SOURCE.read_text(encoding='utf8'))
    arena = json.loads(ARENA.read_text(encoding='utf8'))
    assignments, views = select(data, arena)
    selected = [r for r in data['samples'] if r['id'] in assignments]
    inputs = [ROOT / r['image'] for r in selected]
    labels = [ROOT / r['tasks']['fire_sticker']['label'] for r in selected]
    procedural = [SOURCE, DESCRIPTOR, CONFIG, ARENA, *labels]
    with generation_guard('fire.select_v4_background_mix_20261113', inputs,
                          [OUTPUT, SOURCE, DESCRIPTOR, CONFIG],
                          {'seed': 20261113, 'train': 405, 'val': 60, 'test': 60},
                          code_paths=[Path(__file__)], procedural_inputs=procedural,
                          inplace_paths=[SOURCE, DESCRIPTOR, CONFIG], adopt_outputs=[OUTPUT], root=ROOT) as run:
        if run.reused:
            print('Reused background selection')
            return
        additions = []
        for row in data['samples']:
            if row.get('split') in ('train', 'val', 'test'):
                row['split'] = 'unassigned'
                row['selected_tasks'] = []
        for index, original in enumerate(selected):
            row = copy.deepcopy(original)
            split = assignments[row['id']]
            source_image = row['image']
            source_label = row['tasks']['fire_sticker']['label']
            stem = f'{split}_{index:05d}'
            image = OUTPUT / 'images' / split / (stem + Path(source_image).suffix.lower())
            label = OUTPUT / 'labels' / split / (stem + '.txt')
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source_image, image)
            shutil.copyfile(ROOT / source_label, label)
            row['id'] = 'fire_background_mix_' + str(index)
            row['parent_ids'] = list(dict.fromkeys([original['id'], *row.get('parent_ids', [])]))
            row['image'] = image.relative_to(ROOT).as_posix()
            row['split'] = split
            row['selected_tasks'] = ['fire_sticker']
            row['tasks']['fire_sticker']['label'] = label.relative_to(ROOT).as_posix()
            if source_image in views:
                row['background'] = {'kind': 'blender', 'source_group': original['mother_scene'],
                                     'view_group': views[source_image]}
            else:
                video = original.get('parent_video')
                if not video or not (ROOT / video).is_file():
                    raise ValueError('real source video not recorded: ' + source_image)
                row['background'] = {'kind': 'real', 'source_group': video}
            additions.append(row)
        data['samples'].extend(additions)
        data.pop('delivery_machine_source_check', None)
        config = yaml.safe_load(CONFIG.read_text(encoding='utf8'))
        config['data_overrides'] = {split: [{'images': (OUTPUT / 'images' / split).relative_to(ROOT).as_posix(),
                                            'labels': (OUTPUT / 'labels' / split).relative_to(ROOT).as_posix(),
                                            'source_group': SOURCE.relative_to(ROOT).as_posix()}]
                                    for split in ('train', 'val', 'test')}
        config['source_notes'] = ['525 candidate images: 210 fitted Blender scene and 315 real backgrounds; '
                                  'train/val/test 405/60/60; paired tone variants share a split. '
                                  'Scene and physical assets are shared; source audit pending.']
        descriptor = json.loads(DESCRIPTOR.read_text(encoding='utf8'))
        descriptor['scope'] = '525 candidate images; 210 fitted Blender scene, 315 real backgrounds; source audit pending.'
        SOURCE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        CONFIG.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf8')
        DESCRIPTOR.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(json.dumps({'selected': len(additions), 'splits': dict(Counter(assignments.values()))}))


if __name__ == '__main__':
    main()
