"""Make a bounded prior-only station pilot with competing explicit states."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab.training_common.bayesian_allocation import allocate, plan_generation
from meter_station_plan import (validate_station_declaration, allocated_station_states,
                                validate_arena_declaration, allocated_arena_states)

STATION = 'datasets/assets/blender/meter/v4_parametric_station_v2_20261113/station.blend'
ARENA = 'datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend'

def make_plan(prior_left=(4, 2), prior_right=(2, 4), budget=6):
    states = []
    for side, x, yaw, light, prior in [('left', -.35, -.6, -1, prior_left),
                                       ('right', .35, .6, 1, prior_right)]:
        for i, (temperature, humidity, exposure, saturation, distance) in enumerate([
            (.15, .25, -.4, .7, 7.5), (.35, .85, 0., 1., 7.5),
            (.55, .45, -.4, 1., 7.7), (.75, .65, 0., .7, 7.7),
            (.95, .05, -.4, .7, 7.9), (.25, .95, 0., 1., 7.9)]):
            states.append({'id': f'{side}_{i}', 'failure_event': 'dual_needle_pose_unreadable',
                'min_count': int(i == 0), 'max_count': 1, 'prior': {'alpha': prior[0], 'beta': prior[1]},
                'exposure_weight': 1, 'failure_cost': 1, 'conditions': [],
                'background': {'kind': 'station_scene'}, 'target': {'kind': 'dual_needle_meter'},
                'labels': {'kind': 'candidate_pose'}, 'augmentation': {'kind': 'explicit_station_parameters'},
                'sources': [STATION], 'parameters': {'temperature_fraction': temperature,
                'humidity_fraction': humidity, 'ground_angle_deg': 65, 'camera_z': distance,
                'camera_x_offset': x, 'camera_yaw_deg': yaw, 'exposure_ev': exposure,
                'saturation': saturation, 'light_side': light, 'occlusion': 'none'}})
    return {'schema_version': 1, 'model_key': 'meter', 'target_split': 'train', 'budget': budget,
        'purpose': 'Limited pilot of camera, lighting and tone; priors are design assumptions, not measured failure rates.',
        'uncertainty_weight': .5, 'independence_basis': 'One station scene and model; no independent capture assumed.',
        'constraints': {'resolution': [640, 480], 'camera_fallback': 'reject', 'label_policy': 'candidate_pose'},
        'strata': states}

def make_arena_plan(budget=6):
    plan = make_plan(budget=budget)
    plan['purpose'] = 'Limited arena pilot; priors are design assumptions, not measured failure rates.'
    plan['independence_basis'] = 'One arena scene shares the same meter model as the station scene; independent origin unknown.'
    plan['constraints'] = {'resolution': [640, 480], 'label_policy': 'candidate_pose', 'foreground': False}
    for state in plan['strata']:
        old = state['parameters']
        side = -1 if state['id'].startswith('left') else 1
        state['background'] = {'kind': 'arena_scene'}
        state['augmentation'] = {'kind': 'explicit_arena_parameters'}
        state['sources'] = [ARENA, STATION]
        state['parameters'] = {k: old[k] for k in ('temperature_fraction', 'humidity_fraction',
                                                    'ground_angle_deg', 'exposure_ev', 'saturation')}
        state['parameters'].update(station_x=-.6, station_y=.15, camera_x=-.6 + .06 * side,
                                   camera_y=-.75, camera_z=.42, camera_lens_mm=35.)
    return plan

def parent_request(station_prior=(3, 2), arena_prior=(2, 3)):
    return {'schema_version': 1, 'model_key': 'meter', 'target_split': 'train', 'budget': 12,
        'purpose': 'Pilot parent allocation with prior-only design assumptions; no measured failure rates.',
        'uncertainty_weight': .5, 'independence_basis': 'Two scene files use one shared meter model; independent origin unknown.',
        'strata': [{'id': name, 'failure_event': 'dual_needle_pose_unreadable', 'conditions': [],
                    'min_count': 2, 'max_count': 10, 'prior': {'alpha': prior[0], 'beta': prior[1]},
                    'exposure_weight': 1, 'failure_cost': 1}
                   for name, prior in [('station', station_prior), ('arena', arena_prior)]]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scene', choices=('station', 'arena', 'parent'), default='station')
    parser.add_argument('--budget', type=int, default=6)
    args = parser.parse_args()
    path = Path(args.output).resolve()
    if not path.is_relative_to((ROOT / 'runs/meter').resolve()) or path.suffix != '.json' or path.exists():
        parser.error('output must be a new JSON under runs/meter')
    if args.scene == 'parent':
        plan = parent_request()
        result = allocate(plan, ROOT)
        with path.open('x', encoding='utf8') as out:
            json.dump({'declaration': plan, 'allocation': result}, out, ensure_ascii=False, indent=2)
            out.write('\n')
        print(json.dumps({'output': path.relative_to(ROOT).as_posix(), 'allocation':
            {row['id']: row['planned_count'] for row in result['strata']}}))
        return
    plan = make_plan(budget=args.budget) if args.scene == 'station' else make_arena_plan(budget=args.budget)
    if args.scene == 'station':
        validate_station_declaration(plan, count=args.budget, source=STATION, allow_occlusion=False)
    else:
        validate_arena_declaration(plan, count=args.budget, sources=[ARENA, STATION], foreground=False)
    allocation = plan_generation(plan, ROOT)
    if args.scene == 'station':
        allocated_station_states(allocation, count=args.budget, source=STATION, allow_occlusion=False)
    else:
        allocated_arena_states(allocation, count=args.budget)
    with path.open('x', encoding='utf8') as out:
        json.dump(plan, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'output': path.relative_to(ROOT).as_posix(), 'allocation':
        {row['id']: row['planned_count'] for row in allocation['strata']}}))

if __name__ == '__main__':
    main()
