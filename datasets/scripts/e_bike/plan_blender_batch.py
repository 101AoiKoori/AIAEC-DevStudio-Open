"""Create explicit scene/foreground parameter states for the e-bike Blender entry."""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import load_task_config, source_path

_SOURCE = load_task_config('e_bike')[1]
DESIGN = source_path('e_bike', 'blender_design_manifest')
USE = source_path('e_bike', 'blender_task_use_manifest')
FOREGROUNDS = source_path('e_bike', 'blender_foreground_manifest')
SCENES = _SOURCE['blender_scenes']
CLASSES = ((), ('upright',), ('fallen',), ('upright', 'upright'), ('fallen', 'fallen'), ('upright', 'fallen'))


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def make_plan(scene_key, seed=20261114, pilot=False):
    scene = SCENES[scene_key]
    if not (ROOT / scene).is_file():
        raise ValueError('scene missing')
    ready = {}
    for row in read(USE)['records']:
        for relation in row['task_use_status']:
            if (relation.get('task'), relation.get('split'), relation.get('role'), relation.get('status')) == (
                    'e_bike', 'train', 'target_foreground', 'ready'):
                ready[relation['posture']] = row['image']
    candidates = read(FOREGROUNDS)['records']
    selected = {}
    for cls in ('upright', 'fallen'):
        found = [row for row in candidates if row['path'] == ready.get(cls) and row['class'] == cls]
        if len(found) != 1:
            raise ValueError('ready foreground not uniquely identified')
        selected[cls] = found[0]
        if any(not (ROOT / found[0][key]).is_file() for key in ('path', 'cropped_image', 'source_frame', 'source_video')):
            raise ValueError('foreground input missing')
    rng = random.Random(seed)
    states, signatures = [], set()
    for branch, classes in zip(read(DESIGN)['v4_combination_source_plan_20261114']['branches'], CLASSES):
        quota = 1 if pilot else branch['blender_parent_scene_quota'][scene_key]
        for _ in range(quota):
            while True:
                # Independent integer positions prevent identical declared states.
                params = {'camera_x_mm': rng.randint(-30, 30), 'exposure_hundredth_ev': rng.randint(-30, 20),
                          'actors': [{'class': cls, 'x_mm': (0 if len(classes)==1 else (-55 if i==0 else 55)) + rng.randint(-8, 8),
                                      'y_mm': rng.randint(10, 35), 'height_mm': rng.randint(135, 155) if cls=='upright' else rng.randint(80, 95),
                                      'yaw_degrees': rng.choice((-8, -4, 0, 4, 8))}
                                     for i, cls in enumerate(classes)]}
                signature = json.dumps(params, sort_keys=True)
                if signature not in signatures:
                    signatures.add(signature)
                    break
            sources = {scene}
            for cls in classes:
                sources.update(selected[cls][key] for key in ('path', 'cropped_image', 'source_frame', 'source_video'))
            states.append({'id': f'state_{len(states)+1:06d}', 'failure_event': branch['combination_id'],
                           'min_count': 1, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
                           'exposure_weight': 1, 'failure_cost': 1, 'conditions': [],
                           'background': {'scene': scene}, 'target': {'classes': list(classes)},
                           'labels': {'format': 'yolo_visible_box', 'classes': {'upright': 0, 'fallen': 1}},
                           'augmentation': {'method': 'camera_and_exposure'},
                           'parameters': params, 'sources': sorted(sources)})
    return {'schema_version': 1, 'model_key': 'e_bike', 'target_split': 'train',
            'purpose': 'Explicit fitted Blender e-bike candidate states.', 'budget': len(states),
            'uncertainty_weight': 0, 'independence_basis': 'Scene and real cutout parents recur; fitted renders are not independent camera captures.',
            'scene': scene, 'constraints': {'minimum_visible_fraction': .7, 'minimum_visible_pixels': 120,
                                           'allow_truncation': False}, 'strata': states}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=SCENES, required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    plan = make_plan(args.scene, pilot=args.pilot)
    print(json.dumps(plan, ensure_ascii=False, indent=2))



if __name__ == '__main__':
    main()
