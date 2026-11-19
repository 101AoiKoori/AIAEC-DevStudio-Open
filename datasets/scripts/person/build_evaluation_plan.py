"""Compile complete-label person development and known-source diagnostic splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.package_direct_real import prepare
from datasets.scripts.common.source_history import historical_source, read_source_history
from datasets.scripts.common.source_ledger.config import build_groups
from datasets.scripts.person import build_full_person_plan as scenes
from datasets.scripts.person.compose_planned_ground_person import _batch_layout
from datasets.scripts.person.package_person_dataset import prepare_direct
from datasets.scripts.person.planned_sources import (TASK_COMPLETE, complete_background, configured_background_dispatch,
                                                     verify_current_backgrounds)
from lab.training_common.bayesian_allocation import plan_generation


def _parents(rows, selected, split, history):
    backgrounds = [row.get('parent_group') or row.get('source_video') or row.get('mother_scene') or row.get('source_scene')
                   for row in rows if row.get('split') == split]
    captures = [row.get('parent_video') for _, row, *_ in selected]
    return {historical_source(path, history) for path in [*backgrounds, *captures] if path}


def _reservation_conflicts(row, split, groups, history):
    """Find explicit reservations covering this source, not other candidates."""
    values = [row.get(key) for key in ('parent_group', 'source_group', 'parent_video',
                                       'source_video', 'mother_scene', 'source_scene')]
    candidates = {historical_source(value.replace('\\', '/').rstrip('/'), history)
                  for value in values if isinstance(value, str) and value}
    conflicts = []
    for group in groups:
        rule = group.split
        primary = rule.get('primary')
        reserved = rule.get('status') == 'reserved' or rule.get('action') == 'reserved'
        if not reserved or primary not in ('train', 'val', 'test') or split == primary:
            continue
        if split in (rule.get('same_source_allowed_in') or []):
            continue
        group_paths = {historical_source(path.replace('\\', '/').rstrip('/'), history)
                       for path in group.paths}
        matched = group.group_id in candidates or any(
            value == path or value.startswith(path + '/') for value in candidates for path in group_paths)
        if matched:
            conflicts.append({'group': group.group_id, 'reserved_split': primary})
    return conflicts


def _contrasts(declaration, backgrounds, per_class, max_groups=None):
    if max_groups == 0:
        return [], set()
    try:
        textures = scenes.white_back_textures()
    except KeyError as error:
        if error.args != ('white_back',):
            raise
        textures = {}

    pools = defaultdict(list)
    for state in declaration['strata']:
        index = state['background']['record_index']
        if (state['axes']['person_count'] == 2 and state['axes']['vehicle_occlusion'] == 'none'
                and not backgrounds[index].get('white_back_reference_boxes')):
            pools[index].append(state)
    selected, used = [], set()
    for category in ('community', 'noncommunity'):
        remaining = per_class
        while remaining and (max_groups is None or len(selected) // 3 < max_groups):
            before = remaining
            for states in pools.values():
                for state in states:
                    if state['id'] in used or not scenes._has_white_contrast(state, category, textures):
                        continue
                    triplet = list(scenes._contrast_states(state, category, textures))[1:]
                    candidate = dict(declaration, strata=triplet, budget=3)
                    scenes.attach_sources(candidate)
                    if scenes.validate(candidate, limit=1):
                        continue
                    used.add(state['id'])
                    selected.extend(triplet)
                    remaining -= 1
                    break
                if remaining == 0 or (max_groups is not None and len(selected) // 3 >= max_groups):
                    break
            if remaining == before:
                break
    return selected, used


def _negatives(backgrounds, indices, dispatch_manifest):
    for index in indices:
        row = backgrounds[index]
        background, polygon = scenes.background_block(backgrounds, index, dispatch_manifest)
        width, height = row['source_image_size']
        polygon = [[x, round(y * 4 / 3) if width * 3 == height * 4 else y + 60] for x, y in polygon]
        event = 'white_back_hard_negative' if row.get('white_back_reference_boxes') else 'no_standee_negative'
        state = scenes.make_state(f'evaluation_background_{index}', event, background, polygon, [],
                                  {'value_gain': 1, 'saturation_gain': 1}, [])
        state['min_count'] = 1
        state['parameters']['resolution'] = [640, 480]
        yield state


def build(split, total=300, *, dispatch_manifest=None):
    if split not in ('val', 'test') or type(total) is not int or not 1 <= total <= 3000:
        raise ValueError('Expected a development or diagnostic split and a budget from 1 to 3000')
    dataset, source = load_task_config('person')
    dispatch_manifest, environment_backed = configured_background_dispatch(dispatch_manifest)
    backgrounds = scenes.read_json(dispatch_manifest)['records']
    direct = {name: prepare('person', split=name)[0] for name in ('train', 'val', 'test')}
    history = read_source_history(scenes.ROOT)
    groups = build_groups('person', source.get('pairing') or {})
    parents = {name: _parents(backgrounds, selected, name, history) for name, selected in direct.items()}
    reservation_conflicts = []

    def allowed(row, kind):
        conflicts = _reservation_conflicts(row, split, groups, history)
        if conflicts:
            reservation_conflicts.append({'kind': kind, 'source': row.get('image') or row.get('id'),
                                          'reservations': conflicts})
        return not conflicts

    # The scene production entry needs at least one synthetic state; reserve
    # that slot while selecting no more direct images than the requested total.
    selected_direct = [item for item in direct[split] if allowed(item[1], 'direct')][:max(0, total - 1)]
    direct_ids = [row['id'] for _, row, *_ in selected_direct]
    prepare_direct(split, direct_ids)
    direct_frames = {row['frame_environment']['frame_id'] for _, row, *_ in selected_direct}
    indices = [index for index, row in enumerate(backgrounds)
               if complete_background(row, row['state'].removeprefix('candidate_'), split)
               and allowed(row, 'background')]
    if not indices:
        raise ValueError(f'No eligible person composite backgrounds in {split}: {dispatch_manifest}')
    negative_indices = [index for index in indices if not backgrounds[index].get('boxes') and
                        backgrounds[index].get('frame_id') not in direct_frames]
    synth_budget = total - len(direct_ids)
    negatives = list(_negatives(backgrounds, negative_indices, dispatch_manifest))
    base_rounds = max(1, math.ceil(synth_budget / (15 * len(indices))))
    batch = None
    layout_rounds = 0
    for multiplier in (1, 3, 6):
        layout_rounds = base_rounds * multiplier
        try:
            candidate, _ = scenes.build_sample(split, budget=synth_budget, background_indices=indices,
                                               validate_selected=False, background_scope=TASK_COMPLETE,
                                               directed_train=True,
                                               min_placed_fraction=dataset['target_spec']['minimum_placed_height_fraction'],
                                               dispatch_manifest=dispatch_manifest, layout_rounds=layout_rounds)
        except ValueError as error:
            if str(error) != 'no feasible person sample scenes':
                raise
            break
        if batch is None or len(candidate['strata']) > len(batch['strata']):
            batch = candidate
        if len(candidate['strata']) >= synth_budget:
            break
    if batch is None:
        # Negatives are independent of whether any person placement is viable.
        batch = {'schema_version': 1, 'model_key': 'person', 'target_split': split,
                 'uncertainty_weight': 0, 'constraints': {'foreground': 'planned_actors',
                 'identity': 'front_visible', 'background': TASK_COMPLETE}, 'design': {}, 'strata': []}
    scenes.attach_sources(dict(batch, strata=negatives))
    errors = scenes.validate(dict(batch, strata=negatives), limit=None)
    invalid_negatives = {item['state_id'] for item in errors}
    negatives = [state for state in negatives if state['id'] not in invalid_negatives]
    negative_budget = max(1, synth_budget // 4) if batch['strata'] else synth_budget
    negatives = negatives[:negative_budget]
    remaining = max(0, synth_budget - len(negatives))
    # Each matched group is three complete images; leave a multi-state slot
    # where possible. Deficient coverage is reported, not fabricated.
    group_budget = (remaining - 1) // 3 if remaining > 3 else remaining // 3
    contrasts, used = _contrasts(batch, backgrounds, 6, max_groups=group_budget)
    pool = [state for state in batch['strata'] if state['id'] not in used]
    multi_target = remaining - len(contrasts)
    verified, multi = set(), []
    while pool and multi_target:
        allocator = dict(batch, strata=pool, budget=min(multi_target, len(pool)))
        multi = [state for state in plan_generation(allocator, scenes.ROOT)['strata'] if state['planned_count']]
        pending = [state for state in multi if state['id'] not in verified]
        failed = {item['state_id'] for item in scenes.validate(dict(batch, strata=pending), limit=None)}
        verified.update(state['id'] for state in pending if state['id'] not in failed)
        if not failed:
            break
        pool = [state for state in pool if state['id'] not in failed]
        multi = []

    states = [*negatives, *contrasts, *multi]
    if len(states) + len(direct_ids) > total:
        raise ValueError('Person evaluation plan exceeds the requested image budget')
    for state in states:
        state['min_count'] = 1
        state.pop('planned_count', None)
        state.pop('_procedural', None)
    if not states:
        raise ValueError('No feasible person evaluation scenes; direct-only export requires a separate entry')
    batch.pop('conditional_balance', None)
    batch.update(strata=states, budget=len(states), direct_capture_ids=direct_ids)
    batch['purpose'] = ('Same-venue development validation' if split == 'val' else
                        'Known-route real diagnostic and same-venue composites')
    batch['independence_basis'] = ('Source-group contact and possible overlap must be interpreted as recorded. '
                                   'Fixed standee identities, vehicle occluders and venue are shared; '
                                   'prior-development direct captures retain that history.')
    selected_parents = {historical_source(path, history) for state in states
                        for path in [backgrounds[state['background']['record_index']].get('parent_group') or
                                     backgrounds[state['background']['record_index']].get('source_video') or
                                     backgrounds[state['background']['record_index']].get('source_scene')]
                        if path}
    selected_parents.update(historical_source(row['parent_video'], history)
                            for _, row, *_ in selected_direct if row.get('parent_video'))
    other_candidates = set().union(*(group for name, group in parents.items() if name != split))
    batch['design'].update({
        'requested_images': total,
        'dispatch_manifest': dispatch_manifest,
        'eligible_backgrounds': len(indices),
        'backgrounds_with_actor_placements': len({state['background']['record_index'] for state in states
                                                   if state['target']['actors']}),
        'layout_rounds': layout_rounds,
        'planned_direct_real': len(direct_ids),
        'planned_composites': len(states),
        'planned_images': len(states) + len(direct_ids),
        'unfilled_images': max(0, total - len(states) - len(direct_ids)),
        'matched_groups': len(contrasts) // 3,
        'matched_groups_target': 12,
        'matched_group_shortfall': max(0, 12 - len(contrasts) // 3),
        'invalid_negative_states': sorted(invalid_negatives),
        'excluded_reserved_sources': reservation_conflicts,
        'selected_parent_overlap_with_other_split_candidates': sorted(selected_parents & other_candidates),
        'background_source_groups': dict(Counter(backgrounds[state['background']['record_index']]['source_group']
                                                  for state in states)),
        'scene_counts': dict(Counter(state['failure_event'] for state in states)),
        'evaluation_scope': ('same_venue_development_validation' if split == 'val'
                             else 'known_route_regression_plus_same_venue_composites'),
    })
    _batch_layout(batch)
    if environment_backed:
        verify_current_backgrounds(states, split)
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--dispatch-manifest', help='use a particular registered background dispatch')
    parser.add_argument('--output', type=Path, help='write a new JSON evaluation plan in person/config')
    options = parser.parse_args()
    if options.output:
        output = (scenes.ROOT / options.output).resolve()
        if output.parent != scenes.CONFIG or output.suffix != '.json' or output.exists():
            parser.error('--output must name a new JSON file in person/config')
    plan = build(options.split, dispatch_manifest=options.dispatch_manifest)
    if options.output:
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=1)
            stream.write('\n')
    print(json.dumps(plan['design'], ensure_ascii=False))
