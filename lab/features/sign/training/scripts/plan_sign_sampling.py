"""Draw bounded sign candidate states for the existing Bayesian allocator."""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

import compose_sign_scene as scene
import compose_sign_v4_measured as measured

ROOT = scene.ROOT
SCENE_VIDEO = 'Archive/Assets/working_sources/legacy/site/场景/综合视频/空场与场景物料_001.mp4'
VIEWS = list(measured.VIEWS)
MODES = ('far_full', 'mid_full', 'near_full', 'top_clipped', 'bottom_clipped')


def geometry(rng, mode, anchor, mask_size):
    if mode == 'far_full':
        height, bottom = rng.randint(170, 239), rng.randint(390, 465)
        top = bottom - height
    elif mode == 'mid_full':
        height, bottom = rng.randint(240, 320), rng.randint(390, 465)
        top = bottom - height
    elif mode == 'near_full':
        height = rng.randint(390, 465)
        bottom = rng.randint(height, 480)
        top = bottom - height
    elif mode == 'top_clipped':
        height, bottom = rng.randint(500, 550), rng.randint(420, 475)
        top = bottom - height
    elif mode == 'bottom_clipped':
        height, top = rng.randint(500, 550), rng.randint(5, 25)
    else:
        raise ValueError('unknown sign visibility state')
    width = max(2, round(mask_size[0] * height / mask_size[1]))
    low = max(15, anchor[0] - width // 2) + 2
    high = min(625 - width, anchor[1] - width // 2) - 2
    if low > high:
        raise ValueError('sign geometry cannot fit selected anchor')
    return {'height': height, 'left': rng.randint(low, high), 'top': top}


def mask_sizes():
    sizes = {}
    for mask, _ in VIEWS:
        image = cv2.imdecode(np.fromfile(str(measured.base.ANNOTATED / mask), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            raise ValueError('Missing RGBA sign mask: ' + mask)
        y, x = np.where(image[:, :, 3] > 127)
        if not len(x):
            raise ValueError('Empty sign mask: ' + mask)
        sizes[mask] = (int(x.max() - x.min() + 1), int(y.max() - y.min() + 1))
    return sizes


def build(kind, budget, seed):
    if kind not in ('measured', 'scene') or type(budget) is not int or not 10 <= budget <= 1000:
        raise ValueError('Expected measured/scene and budget between 10 and 1000')
    rng = random.Random(seed)
    sizes = mask_sizes()
    mask_manifest = json.loads((scene.ANNOTATED / 'manifest.json').read_text(encoding='utf8'))
    origins = {f"train_{row['index']}.png": row['source'] for row in mask_manifest['records'] if 'index' in row}
    frames_manifest = json.loads((measured.BACKGROUNDS / 'manifest.json').read_text(encoding='utf8'))
    frames = [row for row in frames_manifest['frames'] if row.get('usage') != 'regression']
    old_backgrounds = scene.train_backgrounds() if kind == 'scene' else []
    if not frames or (kind == 'scene' and not old_backgrounds):
        raise ValueError('No available declared background frames')
    candidates, seen = [], set()
    attempts = 0
    while len(candidates) < budget * 2:
        attempts += 1
        if attempts > budget * 100:
            raise ValueError('Could not draw enough distinct feasible states')
        index = len(candidates)
        if kind == 'measured':
            mask, view = VIEWS[index] if index < len(VIEWS) else rng.choices(VIEWS, weights=(2, 1.5, 1.5, 1, 1.5))[0]
            mode = MODES[index] if index < len(MODES) else rng.choices(MODES, weights=(3, 3, 1, 1, 1))[0]
            frame = rng.choices(frames, weights=[3 if row['image'] in ('images/background_01.png', 'images/background_03.png') else 1 for row in frames])[0]
            tone = rng.choices(('normal', 'dark_low_saturation'), weights=(3, 2))[0]
            placement = geometry(rng, mode, frame['road_anchor_center_x'], sizes[mask])
            background = {'image': frame['image']}
            target = {'state': mode, 'mask': mask}
            augmentation = {'tone': tone}
            conditions = [mode, view, tone]
            sources = list(dict.fromkeys([(measured.BACKGROUNDS / frame['image']).relative_to(ROOT).as_posix(),
                frame['parent_image'], frames_manifest['video'], (scene.ANNOTATED / mask).relative_to(ROOT).as_posix(),
                origins[mask], scene.SIGN_VIDEO]))
            cost = 2 if view == 'back' or mode.endswith('clipped') else 1
        else:
            mask, view = rng.choice(VIEWS[1:])
            path, _ = rng.choice(old_backgrounds)
            height = rng.randint(170, 320)
            bottom = rng.randint(390, 465)
            width = max(2, round(sizes[mask][0] * height / sizes[mask][1]))
            placement = {'height': height, 'left': rng.randint(25, 935-width), 'top': bottom-height}
            gain = round(rng.uniform(.86, 1.12), 4)
            background = {'image': path.relative_to(ROOT).as_posix()}
            target = {'kind': 'positive', 'mask': mask}
            augmentation = {'gain': gain}
            conditions = ['positive']
            sources = list(dict.fromkeys([background['image'], SCENE_VIDEO,
                (scene.ANNOTATED / mask).relative_to(ROOT).as_posix(), origins[mask], scene.SIGN_VIDEO]))
            cost = 1.5 if view in ('side', 'oblique') else 1
        identity = json.dumps([background, target, augmentation, placement], sort_keys=True)
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append({'id': f'state_{rng.getrandbits(64):016x}',
            'failure_event': '牌具漏检或定位失败', 'min_count': int(index < 5), 'max_count': 1,
            'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': cost,
            'conditions': conditions, 'background': background, 'target': target,
            'labels': {'class_id': 0, 'kind': 'visible_fixture_box'}, 'augmentation': augmentation,
            'parameters': placement, 'sources': sources})
    constraints = ({'image_size': [640, 480], 'label_kind': 'visible_fixture_box'} if kind == 'measured' else
                   {'image_size': [960, 480], 'background_crop_xyxy': [480, 600, 1440, 1080],
                    'label_kind': 'visible_fixture_box'})
    return {'schema_version': 1, 'model_key': 'sign', 'target_split': 'train', 'budget': budget,
            'purpose': 'seeded state candidates for real-background sign composites; labels remain candidates',
            'uncertainty_weight': 0, 'independence_basis': 'one background video per batch; one shared physical sign video',
            'constraints': constraints, 'sampling_seed': seed,
            'sampling_description': {'method': 'Python random.Random seeded draws; no model feedback',
                'candidate_count': budget * 2, 'allocation': 'first 5 anchors minimum 1; others 0..1; Bayesian allocator chooses budget',
                'view_weights': 'measured back 2, side 1.5, oblique masks 1.5 each, front 1; scene uniformly selects four masks',
                'size_weights': 'measured far:mid:near:top:bottom = 3:3:1:1:1; scene uniform 170..320 px',
                'tone_weights': 'measured normal:dark = 3:2; scene gain uniform 0.86..1.12',
                'background_weights': 'measured two original frames weight 3 each, remaining frames 1; scene 50 frames uniform',
                'failure_cost': 'design heuristic: back/clipped measured 2 vs other 1; scene side/oblique 1.5 vs front 1',
                'prior': 'Beta(1,1) is a non-data prior, not an observed error rate',
                'limitations': 'background frames share one video per batch; all signs share one real physical object'},
            'strata': candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('measured', 'scene'), required=True)
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to((ROOT / 'runs/sign').resolve()) or output.exists():
        raise ValueError('Plan must be a new file under runs/sign')
    result = build(args.kind, args.budget, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'plan': output.relative_to(ROOT).as_posix(), 'budget': args.budget,
                      'candidate_states': len(result['strata']), 'seed': args.seed}))


if __name__ == '__main__':
    main()
