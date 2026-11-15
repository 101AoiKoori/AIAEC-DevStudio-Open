"""Build bounded meter scene plans from the existing task-use and source records."""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.bayesian_allocation import plan_generation
from meter_station_plan import (allocated_arena_states, allocated_station_states,
                                validate_arena_declaration, validate_station_declaration)

SOURCE_PLAN = ROOT / 'datasets/annotated/meter/real_review_candidates_20261110/manifest.json'
IMAGE_USES = ROOT / 'datasets/annotated/meter/real_meter_visual_manifest.jsonl'
BASE = ROOT / 'lab/features/meter/training/config/meter_v4_unoccluded_scene_20261114.json'
ENDPOINTS = ROOT / 'lab/features/meter/training/config/meter_parametric_endpoints_v4.json'


def image_uses(path, *, task, split, role, status):
    """Select only when all requested fields belong to the same use relationship."""
    selected = []
    with path.open(encoding='utf8') as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            record = json.loads(line)
            for use in record.get('task_use_status', []):
                if all(use.get(key) == value for key, value in (
                        ('task', task), ('split', split), ('role', role), ('status', status))):
                    image = record.get('path') or record.get('image') or record.get('source')
                    if not isinstance(image, str) or not (ROOT / image).is_file():
                        raise ValueError('selected image is missing: ' + str(image))
                    selected.append({'image': image, 'record_index': index, 'use': use})
    return selected


def scene_plan(kind, count, seed, arena, station):
    """Sample unique needle/tone states; sources can repeat across states."""
    fractions = [round(.05 + .1 * i, 2) for i in range(10)]
    grid = [(t, h, e, s) for t in fractions for h in fractions
            for e in (0.0, -.4) for s in (1.0, .7)]
    if not 0 < count <= len(grid):
        raise ValueError('requested scene states exceed the explicit unique-state pool')
    choices = random.Random(seed).sample(grid, count)
    sources = [arena, station] if kind == 'arena' else [station]
    states = []
    for i, (temperature, humidity, exposure, saturation) in enumerate(choices):
        common = {'temperature_fraction': temperature, 'humidity_fraction': humidity,
                  'ground_angle_deg': 65, 'exposure_ev': exposure, 'saturation': saturation}
        if kind == 'arena':
            params = dict(common, station_x=-.6, station_y=.15,
                          camera_x=-.6, camera_y=-.75, camera_z=.42)
            background, augmentation = 'arena_scene', 'explicit_arena_parameters'
        elif kind == 'station':
            params = dict(common, camera_z=10., camera_x_offset=0., camera_yaw_deg=0.,
                          light_side=1., occlusion='none')
            background, augmentation = 'station_scene', 'explicit_station_parameters'
        else:
            raise ValueError('unsupported scene kind')
        states.append({'id': f'{kind}_{i:04d}', 'failure_event': 'dual_needle_pose_unreadable',
                       'min_count': 1, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
                       'exposure_weight': 1, 'failure_cost': 1, 'conditions': [],
                       'background': {'kind': background}, 'target': {'kind': 'dual_needle_meter'},
                       'labels': {'kind': 'candidate_pose'}, 'augmentation': {'kind': augmentation},
                       'sources': sources, 'parameters': params})
    constraints = ({'resolution': [640, 480], 'label_policy': 'candidate_pose', 'foreground': False}
                   if kind == 'arena' else
                   {'resolution': [640, 480], 'camera_fallback': 'reject', 'label_policy': 'candidate_pose'})
    return {'schema_version': 1, 'model_key': 'meter', 'target_split': 'train',
            'purpose': 'Seeded candidate states; geometry and labels are not physical truth or measured performance.',
            'budget': count, 'uncertainty_weight': 0,
            'independence_basis': 'Repeated scene and model inputs; no independent capture or scene claim.',
            'constraints': constraints, 'strata': states}


def make_plans(source, base, endpoints, uses_path, seed):
    plan = source['v4_combination_source_plan']
    branches = {row['id']: row for row in plan['branches']}
    normal = branches['B-normal']
    shares = normal['parents_planned_shares']
    arena_count, station_count = shares['arena'], shares['station_room']
    if arena_count + station_count != normal['target_images']:
        raise ValueError('scene parent shares do not match B-normal')
    arena, station = base['strata'][0]['sources']
    old_station = plan['train_source_groups']['blender_background_candidates'][1]['path']
    if plan['train_source_groups']['blender_background_candidates'][0]['path'] != arena:
        raise ValueError('arena is not the planned parent')
    if endpoints['source_blend'] != station or endpoints['source_face'] != str(Path(station).parent / 'face.png').replace('\\', '/'):
        raise ValueError('parametric endpoint configuration does not match the selected station')
    for item in (arena, station, endpoints['source_face'], old_station):
        if not (ROOT / item).is_file():
            raise ValueError('scene or texture missing: ' + item)
    refs = image_uses(uses_path, task='meter', split='train', role='visual_reference', status='ready')
    direct = image_uses(uses_path, task='meter', split='train', role='direct_sample', status='ready')
    background = image_uses(uses_path, task='meter', split='train', role='composite_background', status='ready')
    references = [{'image': row['image'], 'record_index': row['record_index'],
                   'source_group': row['use'].get('source_group', 'unknown'),
                   'parent_source': row['use'].get('parent_source', 'unknown'),
                   'parent_evidence': row['use'].get('parent_evidence')}
                  for row in refs]
    random.Random(seed).shuffle(references)
    arena_plan = scene_plan('arena', arena_count, seed, arena, station)
    station_plan = scene_plan('station', station_count, seed + 1, arena, station)
    validate_arena_declaration(arena_plan, count=arena_count, sources=[arena, station], foreground=False)
    validate_station_declaration(station_plan, count=station_count, source=station, allow_occlusion=False)
    allocated_arena_states(plan_generation(arena_plan, ROOT), count=arena_count)
    allocated_station_states(plan_generation(station_plan, ROOT), count=station_count,
                             source=station, allow_occlusion=False)
    reasons = {'D-normal': 'No ready direct-sample relation with a complete meter pose label.',
               'D-glare': 'No ready direct-sample relation with confirmed reflection and pose labels.',
               'D-negative': 'No complete confirmed meter-absent labels for the planned images.',
               'R-normal': 'No ready composite-background relation and no existing full-station RGBA batch.',
               'R-glare': 'The current compositor has no controlled glare or point-readability contract.',
               'R-negative': 'The meter compositor requires a positive rendered station.',
               'B-glare': 'The renderers do not declare controlled glare positions or point readability.',
               'B-negative': 'The current planned renderers require a dual-needle meter.'}
    summary = []
    for branch in plan['branches']:
        name, target = branch['id'], branch['target_images']
        arranged = arena_count + station_count if name == 'B-normal' else 0
        summary.append({'branch': name, 'target': target, 'arranged': arranged,
                        'shortfall': target - arranged, 'reason': reasons.get(name)})
    report = {'status': 'planned_not_produced', 'seed': seed,
              'selection': {'manifest': IMAGE_USES.relative_to(ROOT).as_posix(),
                            'filter': 'task=meter, split=train, role and status matched within one relationship',
                            'visual_reference_ready': len(refs), 'direct_sample_ready': len(direct),
                            'composite_background_ready': len(background),
                            'visual_reference_order_seed': seed,
                            'visual_reference_order': references,
                            'visual_reference_scope': 'Reference order only; these images are not render pixels or assigned state parents.',
                            'references_are_not_render_pixels': True},
              'B-normal': {'arena_count': arena_count, 'station_count': station_count,
                           'arena': arena, 'station': station, 'original_station_room': old_station,
                           'station_parent_replacement': 'The old station uses known incorrect v1 endpoints; '
                               'the parametric station replaces it for the station-scene share. '
                               'The same parametric meter is used in both scene branches; not an independent model.',
                           'sampling': 'scene shares fixed by source plan; within each scene sample without replacement '
                               'from 10x10 needle pairs and 2x2 exposure/saturation states; repeated scene/model inputs',
                           'arena_seed': seed, 'station_seed': seed + 1},
              'branches': summary,
              'limits': ['Glare, pixel-level keypoint visibility and physical tick correspondence are not established.',
                         'Entry parsing and camera geometry do not prove raster output or training-label acceptance.',
                         'Production guards, rendering, image generation and training have not run.']}
    return arena_plan, station_plan, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arena-output', required=True)
    parser.add_argument('--station-output', required=True)
    parser.add_argument('--report-output', required=True)
    parser.add_argument('--seed', type=int, default=20261114)
    args = parser.parse_args()
    outputs = [Path(value).resolve() for value in (args.arena_output, args.station_output, args.report_output)]
    allowed = (ROOT / 'runs/meter').resolve()
    if (len(set(outputs)) != 3 or any(not p.is_relative_to(allowed) or p.suffix.lower() != '.json'
                                      or p.exists() or not p.parent.is_dir() for p in outputs)):
        parser.error('outputs must be three different new JSON files under an existing runs/meter directory')
    source, base, endpoints = (json.loads(p.read_text(encoding='utf8')) for p in (SOURCE_PLAN, BASE, ENDPOINTS))
    values = make_plans(source, base, endpoints, IMAGE_USES, args.seed)
    values[2]['artifacts'] = {key: path.relative_to(ROOT).as_posix()
                              for key, path in zip(('arena_plan', 'station_plan', 'report'), outputs)}
    values[2]['B-normal']['arena_entry'] = 'lab/features/meter/training/scripts/render_meter_arena_v4.py'
    values[2]['B-normal']['station_entry'] = 'lab/features/meter/training/scripts/render_meter_candidates.py'
    values[2]['B-normal']['station_endpoints'] = ENDPOINTS.relative_to(ROOT).as_posix()
    for path, value in zip(outputs, values):
        with path.open('x', encoding='utf8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'status': 'planned_not_produced', 'outputs': [str(p.relative_to(ROOT)) for p in outputs]}))


if __name__ == '__main__':
    main()
