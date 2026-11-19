"""Turn existing per-image use relations into building-name candidate plans.

Only JSON plans are written; this module does not generate images or touch the
production receipt. Candidate labels are not promoted to training truth.
"""
import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
HELPERS = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HELPERS)]
from datasets.scripts.building_name_paper import compose_rooftop_papers as rooftop  # noqa: E402
from datasets.scripts.common.names import CHARS, SUFFIX, make_name  # noqa: E402
from datasets.scripts.common.bayesian import plan_production  # noqa: E402


def workspace(value):
    return rooftop._path(value)


def matching_relations(source, role):
    """Find declared task, split and role; the durable review decides eligibility."""
    return [r for r in source['task_use_status']
            if (r.get('task'), r.get('split'), r.get('role')) ==
            ('building_name_paper', 'train', role)]


def _reserved(main):
    result = set()
    for row in main['reserved_parent_groups']:
        result.update(x for x in (row.get('parent'), row.get('path'), row.get('source_group')) if x)
        result.update(row.get('paths', []))
    return result


def _source_record(relation, reserved):
    image = workspace(relation['image'])
    parent = workspace(relation['parent_source'])
    evidence = relation['evidence']
    source = workspace(evidence['manifest'])
    if not image.is_file() or not parent.is_file() or not source.is_file():
        raise ValueError('declared image, parent or evidence is missing')
    with Image.open(image) as picture:
        picture.verify()
    if relation['image'] in reserved or relation['parent_source'] in reserved:
        raise ValueError('source is reserved')
    data = json.loads(source.read_text(encoding='utf8'))
    rows = data.get('images', data.get('samples', data.get('records', [])))
    index = evidence['record_index']
    if type(index) is not int or index < 0 or index >= len(rows):
        raise ValueError('source record index is invalid')
    row = rows[index]
    if row.get('image') != relation['image']:
        raise ValueError('relation image does not match source record')
    parent_record = row.get('parent_video', row.get('source_video', data.get('source_video')))
    if parent_record != relation['parent_source']:
        raise ValueError('relation parent does not match source record')
    if row.get('split') in ('regression', 'val', 'test'):
        raise ValueError('historical regression/val/test is not a train input')
    return data, row


def _candidate(relation, template, reserved):
    _source_record(relation, reserved)
    if relation['image'] == template['image']:
        result = copy.deepcopy(template)
    else:
        result = {'image': relation['image'], 'source_group': relation['parent_source'],
                  'parent_source': relation['parent_source'], 'generation_method': 'provided_video_frame',
                  'image_size': relation['image_size'], 'coverage': 'unknown',
                  'coverage_evidence': relation['constraints'], 'existing_papers': [],
                  'regions': copy.deepcopy(relation['regions'])}
    if result['parent_source'] != relation['parent_source']:
        raise ValueError('candidate parent differs from selected relation')
    result['source_review'] = rooftop.bind_source_review(
        result, [{'slot_id': result['regions'][0]['slot_id'], 'text': '清和大厦'}], 'train')
    return result


def blender_plan(main):
    """Build the independent name renderer declaration without writing files."""
    from datasets.scripts.building_name_paper.render_blender_names import build_plan
    quotas = {row['id']: row['quota'] for row in main['combinations']}
    return build_plan({'single': quotas.get('B_SINGLE_NAME', 0),
                       'multiple': quotas.get('B_MULTI_NAME', 0),
                       'negative': quotas.get('B_NO_PAPER', 0)})


def _parent_quotas(candidates, by_parent, field):
    """Keep each source parent's declared budget when other parents are unavailable."""
    quotas = [0] * len(candidates)
    by_candidate_parent = {}
    for index, candidate in enumerate(candidates):
        by_candidate_parent.setdefault(candidate['parent_source'], []).append(index)
    for parent, indices in by_candidate_parent.items():
        count, remainder = divmod(by_parent[parent][field], len(indices))
        for offset, index in enumerate(indices):
            quotas[index] = count + (offset < remainder)
    return quotas


def build(source, main, seed=20261114):
    reserved = _reserved(main)
    template = source['strata'][0]['background']['candidates'][0]
    background_relations = matching_relations(source, 'background')
    by_parent = {g.get('parent'): g for g in main['train_parent_groups']}
    candidates, rejected = [], []
    for relation in background_relations:
        try:
            candidate = _candidate(relation, template, reserved)
            if candidate['parent_source'] not in by_parent:
                raise ValueError('background is outside the existing train parent plan')
            if any(c['image'] == candidate['image'] for c in candidates):
                raise ValueError('duplicate image in this plan')
            probe = copy.deepcopy(source)
            probe['strata'][0]['background']['candidates'] = [candidate]
            probe['strata'][0]['target']['papers'][0]['slot_id'] = candidate['regions'][0]['slot_id']
            rooftop.validate_request(probe)
            candidates.append(candidate)
        except (ValueError, OSError, KeyError, IndexError) as error:
            rejected.append({'image': relation.get('image'), 'role': 'background', 'reason': str(error)})
    candidates.sort(key=lambda c: c['parent_source'])
    multi = []
    for candidate in candidates:
        if len(candidate['regions']) < 2:
            continue
        try:
            probe = copy.deepcopy(source)
            probe['strata'][0]['background']['candidates'] = [candidate]
            probe['strata'][0]['target']['papers'] = [
                {'slot_id': region['slot_id'], 'text': '清和大厦'} for region in candidate['regions'][:2]]
            rooftop.validate_request(probe)
            multi.append(candidate)
        except (ValueError, OSError, KeyError, IndexError) as error:
            rejected.append({'image': candidate['image'], 'role': 'multi_background', 'reason': str(error)})
    rng = random.Random(seed)
    single_quotas = _parent_quotas(candidates, by_parent, 'single_paper_quota')
    multi_quotas = _parent_quotas(multi, by_parent, 'multi_paper_quota')
    single_target = sum(single_quotas)
    multi_target = sum(multi_quotas)
    names_needed = single_target + 2 * multi_target
    pool_size = len(CHARS) * len(CHARS) * len(SUFFIX)
    if names_needed > pool_size:
        raise ValueError('unique-name pool is too small for the requested plan')
    name_indices = iter(rng.sample(range(pool_size), names_needed))
    states, shares = [], []
    for branch, pool, quotas in (('R_SINGLE_NAME', candidates, single_quotas),
                                  ('R_MULTI_NAME', multi, multi_quotas)):
        for candidate, quota in zip(pool, quotas):
            for _ in range(quota):
                count = 1 if branch == 'R_SINGLE_NAME' else 2
                papers = [{'slot_id': candidate['regions'][i]['slot_id'], 'text': make_name(next(name_indices))}
                          for i in range(count)]
                index = len(states)
                states.append({'id': f'{branch.lower()}_{index:05d}',
                               'failure_event': 'paper_or_background_label_uncertainty',
                               'conditions': ['rooftop_paper', 'no_added_occlusion'],
                               'min_count': 1, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
                               'exposure_weight': 1, 'failure_cost': 1, 'observations': [],
                               'background': {'sampling': 'uniform_group_then_image', 'candidates': [candidate]},
                               'target': {'kind': 'building_name_paper', 'papers': papers},
                               'labels': copy.deepcopy(rooftop.LABELS), 'augmentation': {'mode': 'none'},
                               'sources': ['real_scene_background', 'name_font_asset'],
                               'parameters': {'seed': seed}})
        field = 'single_paper_quota' if branch == 'R_SINGLE_NAME' else 'multi_paper_quota'
        for group in main['train_parent_groups']:
            parent = group.get('parent')
            selected = [(candidate['image'], quota) for candidate, quota in zip(pool, quotas)
                        if candidate['parent_source'] == parent and quota]
            actual_quota = sum(quota for _, quota in selected)
            shares.append({'branch': branch, 'parent_source': parent, 'original_quota': group[field],
                           'reallocated_quota': actual_quota, 'difference': actual_quota - group[field],
                           'source_images': [image for image, _ in selected],
                           'reason': None if selected else 'no eligible task-role source for this branch'})
    rng.shuffle(states)
    plan = {'schema_version': 1, 'model_key': 'building_name_paper', 'target_split': 'train',
            'purpose': 'Seeded name-paper plan from stored detection-eligible sources; original OCR stays unknown.',
            'budget': len(states), 'uncertainty_weight': 0,
            'independence_basis': 'Sources grouped by parent; repeated variants are not independent captures.',
            'constraints': copy.deepcopy(rooftop.CONSTRAINTS), 'strata': states}
    if states:
        rooftop.validate_request(plan)
        rooftop._check_font_text(plan)
        result = plan_production(plan, ROOT)
        if result['planned_total'] != len(states):
            raise ValueError('allocation did not preserve selected source quotas')
    else:
        plan = {'status': 'not_executable', 'reason': 'no eligible name-paper composite source'}

    direct = []
    for relation in matching_relations(source, 'direct_sample'):
        if 'source_label' not in relation:
            continue
        try:
            _, row = _source_record(relation, reserved)
            task = row.get('tasks', {}).get('building_name_paper', {})
            if (task.get('state') != relation['source_state'] or task.get('label') != relation['source_label'] or
                    not workspace(relation['source_label']).is_file()):
                raise ValueError('direct label differs from source declaration')
            if task['state'] not in ('complete_positive', 'complete_negative'):
                raise ValueError('direct source lacks a complete candidate label')
            lines = workspace(relation['source_label']).read_text(encoding='utf8').splitlines()
            if bool(lines) != (task['state'] == 'complete_positive'):
                raise ValueError('direct candidate polarity and existing label disagree')
            selected = copy.deepcopy(relation)
            selected['role'] = ('direct_target' if task['state'] == 'complete_positive'
                                else 'empty_negative')
            from datasets.scripts.building_name_paper.package_real_candidates import bound_direct_review
            selected['source_review'] = bound_direct_review(selected)
            direct.append(selected)
        except (ValueError, OSError, KeyError, IndexError) as error:
            rejected.append({'image': relation.get('image'), 'role': 'direct_sample', 'reason': str(error)})
    rng.shuffle(direct)
    selection = {'schema_version': 1, 'model_key': 'building_name_paper', 'target_split': 'train',
                 'purpose': 'Source-labelled direct candidates; historical review and candidate status remain.',
                 'seed': seed, 'reservation_plan': 'datasets/scripts/building_name_paper/config/combination_plan.json',
                 'records': direct}
    if direct:
        from datasets.scripts.building_name_paper.package_real_candidates import collect_selection_request
        if len(collect_selection_request(selection)) != len(direct):
            raise ValueError('direct entry does not accept the selected relations')
    counts = Counter(r['source_state'] for r in direct)
    try:
        blender = blender_plan(main)
        blender_single = next(x['quota'] for x in main['combinations'] if x['id'] == 'B_SINGLE_NAME')
        blender_multi = next(x['quota'] for x in main['combinations'] if x['id'] == 'B_MULTI_NAME')
        blender_negative = next((x['quota'] for x in main['combinations'] if x['id'] == 'B_NO_PAPER'), 0)
        blender_error = None
    except (ValueError, OSError, KeyError, IndexError) as error:
        blender = {'status': 'not_executable', 'reason': str(error)}
        blender_single = blender_multi = blender_negative = 0
        blender_error = str(error)
    actual = {'D_READABLE': 0, 'D_TEXT_UNKNOWN': counts['complete_positive'],
              'D_NO_PAPER': counts['complete_negative'], 'R_SINGLE_NAME': single_target,
              'R_MULTI_NAME': sum(multi_quotas), 'B_SINGLE_NAME': blender_single,
              'B_MULTI_NAME': blender_multi, 'B_NO_PAPER': blender_negative}
    reason = {'D_READABLE': 'No selected source has confirmed full transcription.',
              'D_TEXT_UNKNOWN': 'Stored detection source is eligible; original OCR remains unknown.',
              'D_NO_PAPER': 'Stored full-frame task absence makes these negative sources eligible.',
              'R_SINGLE_NAME': 'Only eligible sources use their own declared parent quotas; missing parents remain shortfall.',
              'R_MULTI_NAME': 'Only eligible two-slot sources use their own parent quotas; missing parents remain shortfall.',
              'B_SINGLE_NAME': blender_error or 'Independent name renderer; one board and its generated transcription.',
              'B_MULTI_NAME': blender_error or 'Independent name renderer; two boards with separate text and visible masks.',
              'B_NO_PAPER': blender_error or 'Independent name renderer hides all name boards and text; empty detection labels.'}
    leaves = [{'id': c['id'], 'target': c['quota'], 'arranged_candidates': actual[c['id']],
               'shortfall': c['quota'] - actual[c['id']], 'status': 'planned_not_produced',
               'reason': reason[c['id']]} for c in main['combinations']]
    report = {'seed': seed, 'status': 'plans_only_not_produced', 'leaves': leaves,
              'source_quota_reallocation': shares, 'rejected_relations': rejected,
              'selection': 'Same relation task+split+role; reserved and known regression excluded; source history preserved.',
              'randomness': f'Backgrounds reused within eligible parent quotas; {names_needed} distinct generated name indices sampled without replacement.',
              'label_scope': 'Only DB eligible task-role records enter plans. Original OCR remains unknown unless separately labelled.'}
    report['blender_scope'] = 'Independent building-name declaration; shared scene asset, task-owned labels and generated text; rendering has not been run.'
    return plan, selection, report, blender


def main():
    from datasets.scripts.common.config import source_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261114)
    args = parser.parse_args()
    source = json.loads(source_path('building_name_paper', 'rooftop_sources').read_text(encoding='utf8'))
    main_plan = json.loads(source_path('building_name_paper', 'combination_plan').read_text(encoding='utf8'))['v4_task_combination_plan']
    plans = build(source, main_plan, seed=args.seed)
    print(json.dumps(dict(zip(('composite', 'direct', 'report', 'blender'), plans)), ensure_ascii=False))


if __name__ == '__main__':
    main()
