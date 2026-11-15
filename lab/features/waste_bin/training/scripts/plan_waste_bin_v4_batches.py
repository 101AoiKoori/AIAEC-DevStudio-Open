"""Create bounded waste-bin candidate plans from existing task-use relations.

This is a planning-only entry: it never opens Blender or creates images.
"""
import argparse
import itertools
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab.training_common.bayesian_allocation import allocate, plan_generation  # noqa: E402
from bin_render_contract import BIN_NAMES, standalone_planned_jobs  # noqa: E402

RELATIONS = ROOT / 'lab/features/waste_bin/training/config/waste_bin_state_single_closed_plan_20261114.json'
MAIN = ROOT / 'datasets/annotated/waste_bin/v4_grounded_backgrounds_20261112/backgrounds.json'
OUTPUT = ROOT / 'datasets/annotated/waste_bin_state'
V2 = 'datasets/assets/blender/waste_bin/blender_pbr_v2_20261110/waste_bin_pbr.blend'
V3 = 'datasets/assets/blender/waste_bin/blender_pbr_v3_20261110/waste_bin_pbr.blend'


def _matching(row, role):
    """Match all task-use dimensions within one relation, never across uses."""
    return [(i, use) for i, use in enumerate(row.get('task_use_status', []))
            if (use.get('task'), use.get('split'), use.get('role')) == ('waste_bin_state', 'train', role)
            and use.get('status') in ('candidate', 'ready')]


def allocate_scene_states(branches):
    """Allocate candidate attempts before expanding them to exact render states."""
    strata = []
    for branch in ('B4', 'Bfew', 'Bocc'):
        for scene, share in (('arena_master', branches[branch]['shares']['arena_master']),
                             ('arena_detailed', branches[branch]['shares']['arena_detailed'])):
            if not share:
                continue
            strata.append({'id': branch + ':' + scene, 'failure_event': 'visible bin or state evidence unavailable',
                           'conditions': [], 'min_count': math.floor(share * .8), 'max_count': math.ceil(share * 1.2),
                           'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1,
                           'failure_cost': 2 if branch == 'Bocc' else 1, 'observations': []})
    request = {'schema_version': 1, 'purpose': 'Prior-only elastic scene/state candidate attempt allocation',
               'model_key': 'waste_bin_state', 'target_split': 'train',
               'budget': sum(branches[b]['target'] for b in ('B4', 'Bfew', 'Bocc')),
               'independence_basis': 'candidate scene source; source independence not established',
               'uncertainty_weight': 1, 'strata': strata}
    return allocate(request, root=ROOT)


def _state(job_id, scene, bins, rng, occlusion=False):
    # Positions are sampled as explicit renderer inputs; scene-fit is not asserted.
    cam_x = (rng.choice((-1, 1)) * rng.uniform(5.5, 6.5) if occlusion else rng.uniform(-.35, .35))
    return {'id': job_id, 'failure_event': 'visible bin or state evidence unavailable', 'conditions': [],
            'min_count': 1, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
            'exposure_weight': 1, 'failure_cost': 1, 'observations': [],
            'background': {'scene': scene, 'floor_gray': round(rng.uniform(.08, .16), 5),
                           'backdrop_gray': round(rng.uniform(.25, .4), 5)},
            'target': {'bins': bins, 'open_pose': rng.choice(('upright', 'hanging_back'))},
            'labels': {'mode': 'candidate', 'open': 0, 'closed': 1}, 'augmentation': {'mode': 'none'},
            'sources': [scene],
            'parameters': {'camera_location': [round(cam_x, 5), round(rng.uniform(-5.6, -5.2), 5),
                                               round(rng.uniform(1.4, 1.7), 5)],
                           'camera_aim': [0, 0, .78], 'yaw_degrees': 0,
                           'lens_mm': 28 if occlusion else round(rng.uniform(27, 33), 5),
                           'key_energy': round(rng.uniform(800, 1000), 3),
                           'fill_energy': round(rng.uniform(400, 600), 3),
                           'color_factor': round(rng.uniform(.85, 1.15), 5),
                           'roughness_delta': round(rng.uniform(-.04, .04), 5)}}


def build(seed, relations, full_plan):
    rng = random.Random(seed)
    branches = {b['id']: b for b in full_plan['combinations']}
    pools = relations['handoff']['image_task_relations']
    scenes = relations['handoff']['asset_task_relations']
    selected = {}
    for row in scenes:
        uses = _matching(row, 'blender_scene')
        if uses and row.get('asset') in (V2, V3) and (ROOT / row['asset']).is_file():
            selected[row['asset']] = uses[0][1]['status']
    if set(selected) != {V2, V3}:
        raise ValueError('both recorded scene relations and scene files are required')
    background_rows = [r for r in pools if _matching(r, 'background_geometry_input') and (ROOT / r['image']).is_file()]
    direct = []
    for index, row in enumerate(pools):
        for use_index, use in _matching(row, 'direct_sample'):
            if not (ROOT / row['image']).is_file() or not use.get('label') or not (ROOT / use['label']).is_file():
                continue
            status = use.get('label_status')
            count = len(use.get('candidate_boxes', []))
            branch = ('D4' if status == 'candidate_complete_positive' and count == 4 else
                      'Dneg' if status == 'candidate_complete_negative' and count == 0 else None)
            if branch:
                direct.append({'id': 'selected_' + str(index), 'relation_index': index, 'use_index': use_index,
                               'image': row['image'], 'label': use['label'], 'combination_id': branch})
    rng.shuffle(direct)
    direct_counts = Counter()
    direct_selected = []
    for row in direct:
        branch = row['combination_id']
        if direct_counts[branch] < branches[branch]['target']:
            direct_counts[branch] += 1
            direct_selected.append(row)
    direct = direct_selected
    direct_parent_counts = Counter((row['combination_id'], pools[row['relation_index']].get('source_group') or 'unknown')
                                   for row in direct)
    direct_plan = {'model_key': 'waste_bin_state', 'target_split': 'train',
                   'relation_source': RELATIONS.relative_to(ROOT).as_posix(), 'seed': seed,
                   'scope': 'Source labels and role stay candidate; intake exports unassigned candidate records.',
                   'parent_shares': {branch: {group: count for (name, group), count in direct_parent_counts.items()
                                              if name == branch} for branch in ('D4', 'Dfew', 'Dneg')},
                   'records': direct}

    allocation = allocate_scene_states(branches)
    quotas = {row['id']: row['planned_count'] for row in allocation['strata']}
    totals = {b: sum(quotas.get(b + ':' + s, 0) for s in ('arena_master', 'arena_detailed'))
              for b in ('B4', 'Bfew', 'Bocc')}
    # Allocate semantic strata before materializing exact unique render states.
    quotient, remainder = divmod(totals['B4'], 16)
    patterns = [i for i in range(16) for _ in range(quotient + (i < remainder))]
    quotient, remainder = divmod(totals['Bfew'], 3)
    few_counts = [n for n in (1, 2, 3) for _ in range(quotient + (n <= remainder))]
    rng.shuffle(patterns)
    rng.shuffle(few_counts)
    scene_jobs = {V2: [], V3: []}
    specs = [('B4', patterns), ('Bfew', few_counts), ('Bocc', list(range(totals['Bocc'])))]
    for branch, values in specs:
        first_scene_count = quotas.get(branch + ':arena_master', 0)
        if first_scene_count + quotas.get(branch + ':arena_detailed', 0) != len(values):
            raise ValueError('scene parent shares do not match the branch target')
        for index, value in enumerate(values):
            scene = V2 if index < first_scene_count else V3
            if branch == 'B4':
                bins = {name: ('open' if value & (1 << i) else 'closed') for i, name in enumerate(BIN_NAMES)}
            else:
                count = value if branch == 'Bfew' else rng.choice((2, 3, 4))
                subsets = list(itertools.combinations(BIN_NAMES, count))
                if branch == 'Bocc':
                    subsets = [s for s in subsets if any(BIN_NAMES.index(s[i+1]) == BIN_NAMES.index(s[i])+1
                                                         for i in range(len(s)-1))]
                chosen = rng.choice(subsets)
                bins = {name: rng.choice(('open', 'closed')) for name in chosen}
            scene_jobs[scene].append(_state(branch + '_' + str(index).zfill(4), scene, bins, rng, branch == 'Bocc'))
    outputs = {}
    for scene, jobs in scene_jobs.items():
        declaration = {'schema_version': 1, 'purpose': 'Fixed-seed bin-state candidate render attempts',
                       'model_key': 'waste_bin_state', 'target_split': 'train', 'budget': len(jobs),
                       'independence_basis': 'one scene file; physical parent independence unknown',
                       'uncertainty_weight': 0,
                       'constraints': {'resolution': [640, 480], 'label_mode': 'candidate',
                                       'min_visibility': .7, 'min_visible_pixels': 24},
                       'strata': jobs,
                       'handoff': {'seed': seed, 'source_relation': selected[scene], 'scene': scene,
                                   'semantic_allocation': allocation,
                                   'scope': 'Candidate attempts, not approved training labels or observed yields. '
                                            'Open-state mouth proxies do not prove visual contrast. '
                                            'Side-view bounding-box overlap is not proof of pixel occlusion.',
                                   'occlusion': 'Bocc attempts use side camera; actual occlusion and state visibility '
                                                'remain unknown until masks exist.',
                                   'source_reallocation': 'Original arena scene shares are reallocated to these PBR '
                                                          'scenes; v3 registry state is review_pending.',
                                   'command': 'render_blender_waste_bin.py --plan <this file> --seed ' + str(seed)}}
        allocated = plan_generation(declaration, root=ROOT)
        if len(standalone_planned_jobs(allocated, scene, 640, 480, .7, 24)) != len(jobs):
            raise ValueError('standalone entry rejected generated plan')
        outputs[scene] = declaration
    direct_counts = Counter(row['combination_id'] for row in direct)
    report = {'seed': seed, 'relation_source': RELATIONS.relative_to(ROOT).as_posix(),
              'full_plan': MAIN.relative_to(ROOT).as_posix() + '#/v4_20261114_combination_selection',
              'sampling': 'Fixed seed. Scene states and explicit camera/light/material parameters are sampled '
                          'with replacement; duplicate exact states are rejected by the entry contract. '
                          'Direct images are shuffled and taken at most once.',
              'scene_shares': {V2: len(scene_jobs[V2]), V3: len(scene_jobs[V3])},
              'scene_relation_status': selected,
              'semantic_allocation': allocation,
              'background_relation_count': len(background_rows),
              'background_parent_groups': sorted({r['source_group'] for r in background_rows}),
              'branches': [],
              'reallocation': 'Original arena_master and arena_detailed scene shares were mapped to PBR v2 '
                              'and PBR v3 respectively; neither alternate is a rendered image and v3 is review_pending.',
              'limits': ['No actual output images or masks were generated.',
                         'Open-state mask proxies and candidate labels do not establish visible state truth.',
                         'Bocc camera arrangements are attempts; semantic occlusion count is unknown.',
                         'Real-background compositor cannot run without actual eligible foreground render and masks.']}
    for branch in ('D4', 'Dfew', 'Dneg', 'R4', 'Rfew', 'Rocc', 'B4', 'Bfew', 'Bocc'):
        target = branches[branch]['target']
        attempts = (direct_counts[branch] if branch.startswith('D') else
                    totals[branch] if branch in ('B4', 'Bfew', 'Bocc') else 0)
        confirmed = attempts if branch in ('D4', 'Dneg', 'B4', 'Bfew') else 0
        reason = ({'D4': 'Only one selected, unreserved four-bin image has an existing candidate label.',
                   'Dfew': 'No selected 1-3-bin direct relation has a complete candidate label; the known partial image is regression.',
                   'Dneg': 'Only one selected, unreserved negative candidate is available; its label remains candidate.',
                   'R4': 'Eligible foreground image and state masks do not exist.',
                   'Rfew': 'Eligible foreground image and state masks do not exist.',
                   'Rocc': 'Eligible foreground image and state masks do not exist.',
                   'Bocc': 'Side-view candidate attempts are planned; actual occlusion and state visibility remain unknown.',
                   'B4': 'Rendering and candidate-label eligibility have not been assessed.',
                   'Bfew': 'Rendering and candidate-label eligibility have not been assessed.'})[branch]
        if branch.startswith('D'):
            selected_shares = {group: count for (name, group), count in direct_parent_counts.items() if name == branch}
        elif branch.startswith('B'):
            selected_shares = {scene: sum(job['id'].startswith(branch + '_') for job in jobs)
                               for scene, jobs in scene_jobs.items()}
        else:
            selected_shares = {}
        report['branches'].append({'id': branch, 'target': target,
                                   'requested_parent_shares': branches[branch].get('shares', {}),
                                   'planned_input_parent_shares': selected_shares,
                                   'entry_arrangeable_attempts': attempts,
                                   'arranged_non_occlusion_specific_attempts': confirmed,
                                   'remaining_attempts': max(0, target-attempts),
                                   'difference_from_original_target': attempts-target,
                                   'confirmed_occlusion_count': 0 if branch == 'Bocc' else None,
                                   'unverified_occlusion_gap': target if branch == 'Bocc' else None,
                                   'occlusion_satisfaction_known': False if branch == 'Bocc' else None,
                                   'reason': reason})
    return direct_plan, outputs, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=20261114)
    args = ap.parse_args()
    relations = json.loads(RELATIONS.read_text(encoding='utf8'))
    full = json.loads(MAIN.read_text(encoding='utf8'))['v4_20261114_combination_selection']
    direct, scenes, report = build(args.seed, relations, full)
    names = {'direct': OUTPUT / 'v4_source_batches_20261114_direct.json',
             V2: OUTPUT / 'v4_source_batches_20261114_pbr_v2.json',
             V3: OUTPUT / 'v4_source_batches_20261114_pbr_v3.json',
             'report': OUTPUT / 'v4_source_batches_20261114_report.json'}
    if any(path.exists() for path in names.values()):
        raise FileExistsError('planned output already exists; preserve it')
    for key, value in [('direct', direct), (V2, scenes[V2]), (V3, scenes[V3]), ('report', report)]:
        with names[key].open('x', encoding='utf8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'files': [p.relative_to(ROOT).as_posix() for p in names.values()],
                      'scene_shares': report['scene_shares'], 'branches': report['branches']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
