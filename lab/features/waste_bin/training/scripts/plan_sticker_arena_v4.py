"""Write a bounded waste-sticker Blender render declaration; renders nothing."""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ASSET = 'datasets/assets/blender/waste_bin/waste_sticker_animated_v4.blend'
CONSTRAINTS = {'resolution': [640, 480], 'labels': 'visible_paper_yolo_candidate',
               'min_visible_fraction': 0.70, 'min_visible_pixels': 24}


def make_plan(scene, count, offset, seed, branch='front_view'):
    if not 1 <= count <= 20 or offset < 0:
        raise ValueError('count must be 1..20 and offset nonnegative')
    if branch not in ('front_view', 'oblique_view'):
        raise ValueError('unsupported camera branch')
    rng = random.Random(seed + offset)
    states = []
    for n in range(count):
        frame = ((offset + n) % 240) + 1
        x = rng.uniform(-.025, .025) if branch == 'front_view' else rng.choice((-1, 1)) * rng.uniform(.12, .18)
        p = {'frame': frame, 'camera_location': [round(x, 5),
             round(rng.uniform(.62,.66), 5), round(rng.uniform(.16,.20), 5)],
             'camera_aim': [0.0, 1.0, .075]}
        states.append({'id': f'sticker_{offset+n:05d}', 'min_count': 1, 'max_count': 1,
            'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
            'failure_event': 'render_or_visible_label_incomplete',
            'background': {'scene': scene}, 'target': {'asset': ASSET, 'frame': frame},
            'labels': {'format': 'yolo_detection', 'visibility': 'visible_paper_candidate'},
            'augmentation': {'seeded_camera': True}, 'parameters': p, 'sources': [scene, ASSET]})
    return {'schema_version': 1, 'model_key': 'waste_sticker', 'target_split': 'train',
            'purpose': 'Bounded animated sticker render candidate batch', 'budget': count,
            'uncertainty_weight': 0, 'independence_basis': 'one source arena and one animated bin asset',
            'constraints': CONSTRAINTS, 'strata': states,
            'planning': {'seed': seed, 'offset': offset, 'branch': branch, 'asset_status': 'candidate_review_pending'}}


def make_full_plan(scenes, seed):
    prior_manifest = ROOT / 'runs/waste_sticker/waste_sticker_arena_pilot_20261114/manifest.json'
    prior_data = json.loads(prior_manifest.read_text(encoding='utf8'))
    if (prior_data.get('status') != 'candidate' or len(prior_data.get('records', [])) != 2
            or any(row['actual']['scene'] != scenes[1] for row in prior_data['records'])
            or [row['actual']['frame'] for row in prior_data['records']] != [1, 2]):
        raise ValueError('two prior front-view candidates are not available as expected')
    batches = []
    for scene in scenes:
        prior = 2 if scene.endswith('arena_master_user_20261113.blend') else 0
        for branch, count, offset in (('front_view', 767-prior, prior), ('oblique_view', 384, 767)):
            for start in range(0, count, 20):
                plan = make_plan(scene, min(20, count-start), offset+start, seed, branch)
                batches.append({'index': len(batches), 'scene': scene, 'branch': branch, 'declaration': plan})
    return {'task': 'waste_sticker', 'status': 'planned_not_produced', 'seed': seed,
            'asset_status': 'candidate_review_pending', 'source_plan': 'lab/features/waste_bin/training/config/waste_sticker_source_plan_20261114.json',
            'branches': [{'name': 'front_view', 'target': 1534, 'existing_candidates': 2, 'scheduled': 1532, 'gap': 0},
                         {'name': 'oblique_view', 'target': 768, 'existing_candidates': 0, 'scheduled': 768, 'gap': 0,
                          'condition': 'camera is offset horizontally; image framing and visibility require actual rendering'},
                         {'name': 'partially_occluded_but_identifiable', 'target': 766, 'existing_candidates': 0,
                          'scheduled': 0, 'gap': 766, 'condition': 'entry has no explicit occluder control'}],
            'prior_candidate': 'runs/waste_sticker/waste_sticker_arena_pilot_20261114/manifest.json',
            'scope': 'Candidate planning; two scene files do not establish independent capture. Visibility gate may stop a batch.',
            'batches': batches}


def make_flexible_plan(scene, count, offset, seed):
    if not 2 <= count <= 20:
        raise ValueError('budget must be 2..20 to cover both view groups')
    rng = random.Random(seed + offset + 100000)
    easy = make_plan(scene, count, offset, seed, 'front_view')
    hard = make_plan(scene, count, offset + count, seed, 'oblique_view')
    for group, states in (('front', easy['strata']), ('oblique', hard['strata'])):
        for n, state in enumerate(states):
            p = state['parameters']
            p.update(exposure_ev=round(rng.uniform(-.5, .5), 3),
                     saturation=round(rng.uniform(.75, 1.15), 3), blur_pixels=rng.choice((0, 1, 2)),
                     occluder_color=(None if n % 2 == 0 else ('red' if n % 4 == 1 else 'gray')),
                     paper_width_ratio=round(rng.uniform(.39,.46), 4), paper_aspect=round(rng.uniform(.9,1.04), 4))
            state['augmentation'] = {k: p[k] for k in ('exposure_ev', 'saturation', 'blur_pixels', 'occluder_color', 'paper_width_ratio', 'paper_aspect')}
            state['min_count'] = 1 if n == 0 else 0
            state['prior'] = {'alpha': 2, 'beta': 8} if group == 'front' else {'alpha': 8, 'beta': 2}
            state['conditions'] = ['design_group_' + group]
    easy['strata'] += hard['strata']
    easy['purpose'] = 'Flexible prior-weighted front and oblique sticker candidates'
    easy['planning'] = {'seed': seed, 'offset': offset, 'asset_status': 'candidate_review_pending',
                        'prior_scope': 'design weights for allocation, not observed failure rates',
                        'paper_scale_reference':'datasets/annotated/waste_bin/waste_sticker_p22_stationary_annotations_20261114.json',
                        'candidate_states': 2 * count}
    return easy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene')
    p.add_argument('--out', required=True)
    p.add_argument('--count', type=int, default=4)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--seed', type=int, default=20261114)
    p.add_argument('--full-plan', action='store_true')
    p.add_argument('--flexible', action='store_true')
    p.add_argument('--branch', choices=('front_view','oblique_view'), default='front_view')
    args = p.parse_args()
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT) or out.suffix != '.json' or out.exists():
        raise ValueError('scene must exist and plan must be a new workspace JSON')
    if args.full_plan:
        scenes = ['datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_indoor_detailed.blend',
                  'datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend']
        if not all((ROOT / scene).is_file() for scene in scenes):
            raise ValueError('planned arena scene is missing')
        data = make_full_plan(scenes, args.seed)
    else:
        if not args.scene:
            raise ValueError('provide --scene for a single batch')
        scene = (ROOT / args.scene).resolve()
        if not scene.is_relative_to(ROOT) or not scene.is_file():
            raise ValueError('scene must be an existing workspace file')
        data = (make_flexible_plan(scene.relative_to(ROOT).as_posix(), args.count, args.offset, args.seed)
                if args.flexible else make_plan(scene.relative_to(ROOT).as_posix(), args.count, args.offset, args.seed, args.branch))
    with out.open('x', encoding='utf8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('planned batches:', len(data['batches']) if args.full_plan else 1)


if __name__ == '__main__':
    main()
