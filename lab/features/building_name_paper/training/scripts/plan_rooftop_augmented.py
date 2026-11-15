"""Plan source-grouped augmented rooftop candidates without producing images."""
import copy
import json
from pathlib import Path

import compose_rooftop_papers as rooftop
from lab.training_common.production_guard import plan_production
from plan_rooftop_batch import ROOT, _candidate, _reserved, _source_record, matching_relations

BASE = ROOT / 'runs/building_name_paper'
MANIFEST = 'datasets/annotated/fire/real_review_candidates_20261110/manifest.json'


def make_plan():
    source = json.loads((BASE / 'single_rooftop_input_20261114.json').read_text(encoding='utf8'))
    master = json.loads((BASE / 'real_scene_random_plan_40.json').read_text(encoding='utf8'))['v4_task_combination_plan']
    records = json.loads((ROOT / MANIFEST).read_text(encoding='utf8'))['samples']
    # Manually inspected visible facade bands; geometry remains candidate-level.
    additions = [(0, [[1010,145],[1250,168],[1248,197],[1008,173]]),
                 (6, [[912,158],[1070,176],[1070,197],[912,179]]),
                 (18, [[710,375],[1010,287],[1010,320],[710,408]])]
    for index, quad in additions:
        row = records[index]
        source['task_use_status'].append({'image': row['image'], 'task': 'building_name_paper',
            'split': 'train', 'role': 'background', 'status': 'candidate',
            'parent_source': row['source_video'], 'image_size': [1920, 1080],
            'evidence': {'manifest': MANIFEST, 'record_index': index, 'index_basis': 'zero_based'},
            'regions': [{'slot_id': 'facade', 'building_id': 'visible_building', 'quad_xy': quad,
                         'evidence': 'Manually inspected visible upper facade; geometry is approximate.'}],
            'constraints': 'Historical real_test_review_candidate; split history unknown. Other papers and coverage unknown.'})
    template = source['strata'][0]['background']['candidates'][0]
    reserved = _reserved(master)
    candidates = []
    for relation in matching_relations(source, 'background'):
        _source_record(relation, reserved)
        candidates.append(_candidate(relation, template, reserved))
    if len(candidates) != 5 or len({c['parent_source'] for c in candidates}) != 5:
        raise ValueError('expected five usable source parents')
    states = []
    for candidate in candidates:
        layouts = [candidate['regions'][:1]]
        if len(candidate['regions']) >= 2:
            layouts.append(candidate['regions'][:2])
        for layout in layouts:
            state = {'id': f'facade_group_{len(states)}', 'failure_event': 'paper_or_background_label_uncertainty',
                     'conditions': ['rooftop_paper', 'no_added_occlusion'], 'min_count': 100, 'max_count': 300,
                     'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
                     'observations': [], 'background': {'sampling': 'uniform_group_then_image', 'candidates': [candidate]},
                     'target': {'kind': 'building_name_paper', 'papers': [{'slot_id': r['slot_id'], 'text': '清和大厦'} for r in layout]},
                     'labels': copy.deepcopy(rooftop.LABELS), 'augmentation': {'mode': 'scene_bounded_v1'},
                     'sources': ['real_scene_background', 'name_font_asset'], 'parameters': {'seed': 20261114 + len(states)}}
            states.append(state)
    plan = {'schema_version': 1, 'model_key': 'building_name_paper', 'target_split': 'train',
            'purpose': 'Source grouped augmented candidates, partial background labels; priors are design assumptions.',
            'budget': 1200, 'uncertainty_weight': 0, 'independence_basis': 'video parent groups; session independence unknown',
            'constraints': copy.deepcopy(rooftop.CONSTRAINTS), 'strata': states}
    rooftop.validate_request(plan)
    rooftop._check_font_text(plan)
    allocation = plan_production(plan, ROOT)
    changed = copy.deepcopy(plan)
    changed['strata'][0]['prior'] = {'alpha': 9, 'beta': 1}
    alternate = plan_production(changed, ROOT)
    first = [x['planned_count'] for x in allocation['strata']]
    second = [x['planned_count'] for x in alternate['strata']]
    if first == second:
        raise ValueError('prior change did not alter state allocations')
    return source, plan, {'baseline': first, 'alternative_alpha_9_beta_1_state_0': second,
                          'status': 'prior_sensitivity_contract_check_not_model_validation',
                          'risk_prior': 'Beta(1,1), equal costs are design assumptions, not measured error'}


if __name__ == '__main__':
    paths = [BASE / 'augmented_sources_20261114.json', BASE / 'augmented_plan_20261114.json',
             BASE / 'augmented_allocation_check_20261114.json']
    if any(path.exists() for path in paths):
        raise ValueError('output plan already exists')
    for path, value in zip(paths, make_plan()):
        with path.open('x', encoding='utf8') as out:
            json.dump(value, out, ensure_ascii=False, indent=2)
            out.write('\n')
    print('Wrote source relations, augmented production plan and prior sensitivity check')
