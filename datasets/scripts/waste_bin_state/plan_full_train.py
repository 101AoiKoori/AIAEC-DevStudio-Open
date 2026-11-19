"""Compile the full train positive composition budget from eligible backgrounds."""

import argparse
from collections import Counter, defaultdict
from itertools import cycle, product
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.scripts.waste_bin_state import plan_representative_sample as sample


BASE = ROOT / 'datasets/annotated/waste_bin_state'
PLANS = BASE / 'full_plans'
CAMERA = 'real_background_composite'
BLENDER = 'blender_route_composite'
AXES = ('color_set', 'foreground_pair', 'background_frame', 'tone', 'position', 'size', 'view', 'occlusion')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def oblique_source():
    """Expose paired, already verified light-downward renders to the compositor."""
    path = BASE / 'sample_history_20261117/lid_diagnostics/manifest.json'
    if not path.is_file():
        return None
    report = read_json(path)
    if report['scene_name'] != sample.BINS_PHOTO_SCENE:
        raise ValueError('The light-downward renders use an unexpected scene')
    groups = defaultdict(list)
    for row in report['samples']:
        tilt = row['camera']['above_rim_deg']
        if (row['variant'] == 'fixed_light' and 0 < tilt <= 6
                and row['geometry_evidence_eligible'] and row['candidate_label_applies_to_this_image']):
            groups[row['pair']].append(row)
    rows = []
    for members in groups.values():
        if {row['physical_state'] for row in members} != {'open', 'closed'} or len(members) != 2:
            continue
        for row in members:
            paths = {}
            for column, original in (('image', 'image'), ('candidate_label', 'geometry_candidate_label'),
                                     ('mask_visible', 'whole_mask'), ('mask_state_visible', 'state_mask')):
                source_path = (path.parent / row[original]).resolve()
                if not source_path.is_file() or not source_path.is_relative_to(BASE):
                    raise ValueError(f'Missing diagnostic asset: {source_path}')
                paths[column] = '../' + source_path.relative_to(BASE).as_posix()
            rows.append({**paths, 'labels': [{'bin': 'green', 'state': row['physical_state'],
                                             'label_eligible': True,
                                             'visible_fraction': row['whole_visible_fraction']}],
                         'visible_bins': ['green'], 'bin_count': 1, 'open_pose': 'hanging_back',
                         'camera': row['camera'], 'plan_state_id': row['id']})
    if not rows:
        return None
    output = PLANS / 'train_foreground_oblique.json'
    write_json(output, {'status': 'complete_candidates', 'split': 'train', 'scene_file': report['scene_file'],
                        'scene_name': report['scene_name'], 'scene_sha256': report['scene_sha256'],
                        'provenance': {'manifest': sample.compositor.rel(path),
                                       'use': 'Verified train precursor with unchanged image and mask files'},
                        'samples': rows})
    return sample.compositor.rel(output)


def pair_signature(frame):
    return tuple(sorted((color, state == 'open') for color, state in frame['bins'].items()))


def foreground_pairs(manifests):
    grouped = defaultdict(lambda: ([], []))
    for frame in sample.foregrounds(manifests):
        actual = pair_signature(frame)
        inverse = tuple((color, not opened) for color, opened in actual)
        grouped[min(actual, inverse)][int(actual > inverse)].append(frame)
    by_quantity = defaultdict(list)
    for signature, (a_rows, b_rows) in grouped.items():
        for a, b in product(a_rows, b_rows):
            if a['view'] == b['view'] and a['occlusion'] == b['occlusion']:
                by_quantity[len(signature)].append((a, b))
    if not by_quantity:
        raise ValueError('No label-complete complementary foreground pairs are available')
    return by_quantity


def load_backgrounds(source):
    rows = sample.qualified_backgrounds(source, 'train')
    database, eligible = sample.compositor.eligible_ground_backgrounds(source, rows, 'train')
    role = sample.compositor.exposure_checker(sample.compositor.build_queue(ROOT))
    backgrounds = defaultdict(list)
    for row in rows:
        if row['id'] not in eligible or role(row['parent_image'])[0] == 'regression':
            continue
        prepared = sample.prepared_background(row, *eligible[row['id']])
        if prepared['height_range'][0] > prepared['height_range'][1]:
            continue
        prepared['perspective_scale_observation'] = sample.compositor.current_scale_observation(
            database, row['frame_id'], prepared['source_image_size'])
        backgrounds[sample.source_kind(row)].append(prepared)
    return backgrounds


def pair_budgets(camera, blender):
    if any(type(n) is not int or n < 0 or n % 2 for n in (camera, blender)) or not camera + blender:
        raise ValueError('Composition requires a positive combined budget of complete pairs')
    budgets = {}
    for kind, count in ((CAMERA, camera), (BLENDER, blender)):
        pairs = count // 2
        single_pairs = (pairs + 1) // 2
        each, remainder = divmod(pairs - single_pairs, 3)
        budgets[kind] = {1: single_pairs, **{n: each + int(n - 2 < remainder) for n in (2, 3, 4)}}
    return budgets


def placement_pair(a, b, background, rng):
    # Restart with the same placement draw to keep both poses near matching positions.
    paired_rng = random.Random()
    paired_rng.setstate(rng.getstate())
    left = sample.placement_for(a, background, rng)
    right = sample.placement_for(b, background, paired_rng)
    return left, right


def shuffled_passes(values, rng):
    while True:
        yield from rng.sample(values, len(values))


def candidates(kind, quantity, budget, foregrounds, backgrounds, rng, seen, rejections):
    backgrounds = [row for row in backgrounds if row['existing_bins'] == []
                   or (quantity in (2, 3) and len(row['existing_bins']) <= 4 - quantity)]
    if not foregrounds or not backgrounds:
        return []
    bg_order = shuffled_passes(backgrounds, rng)
    by_color = defaultdict(list)
    for pair in foregrounds:
        by_color[tuple(sorted(pair[0]['bins']))].append(pair)
    variants = {colors: cycle(rng.sample(group, len(group))) for colors, group in by_color.items()}
    colors = cycle(rng.sample(list(variants), len(variants)))
    accepted, failures = [], Counter()
    target = budget + max(8, budget // 8)
    for _ in range(target * 16):
        bg = next(bg_order)
        a, b = next(variants[next(colors)])
        key = (bg['id'], a['image'], b['image'])
        if failures[key] >= 3:
            continue
        try:
            placements = placement_pair(a, b, bg, rng)
        except ValueError as error:
            rejections[str(error)] += 1
            failures[key] += 1
            continue
        identities = [(bg['id'], frame['path'], tuple(placement[1]))
                      for frame, placement in zip((a, b), placements)]
        if any(identity in seen for identity in identities):
            continue
        seen.update(identities)
        accepted.append((a, b, bg, *placements, rng.choice(tuple(sample.TONES))))
        if len(accepted) == target:
            break
    return accepted


def feasible_pair_budgets(requested, foregrounds, backgrounds, rng, rejections):
    """Fill shortages from other feasible groups, favouring the same visible-bin count."""
    goals = {(kind, quantity): pairs for kind, groups in requested.items() for quantity, pairs in groups.items()}
    proposed = {group: [] for group in goals}
    allocated = dict.fromkeys(goals, 0)
    seen = set()

    def expand(group, want):
        kind, quantity = group
        proposed[group].extend(candidates(kind, quantity, want, foregrounds.get(quantity, ()),
                                           backgrounds.get(kind, ()), rng, seen, rejections))

    for group, want in goals.items():
        if want:
            expand(group, want)
            allocated[group] = min(want, len(proposed[group]))

    shortages = {group: want - allocated[group] for group, want in goals.items()}
    for group, remaining in shortages.items():
        if not remaining:
            continue
        kind, quantity = group
        alternatives = sorted((key for key in goals if key != group),
                              key=lambda key: (key[1] != quantity, (key[1] == 1) != (quantity == 1), key[0] != kind))
        for alternate in alternatives:
            if not remaining:
                break
            spare = len(proposed[alternate]) - allocated[alternate]
            if spare < remaining:
                expand(alternate, remaining - spare)
            added = min(remaining, len(proposed[alternate]) - allocated[alternate])
            allocated[alternate] += added
            remaining -= added
    return proposed, allocated


def choose_pairs(kind, quantity, budget, proposed, seed, offset):
    definition, members = [], {}
    for index, (a, b, bg, first, second, tone) in enumerate(proposed):
        pair_id = f'full_train_pair_{kind}_{quantity}_{index:04d}'
        rows = [sample.make_state(0, a, bg, first, tone), sample.make_state(0, b, bg, second, tone)]
        axes = {axis: '+'.join(sorted({row['axes'][axis] for row in rows}))
                for axis in ('position', 'size', 'view', 'occlusion')}
        axes.update(color_set=','.join(sorted(a['bins'])), tone=tone, background_frame=bg['frame_id'],
                    foreground_pair=' + '.join(Path(frame['image']).name for frame in (a, b)))
        definition.append({'id': pair_id, 'conditions': [], 'min_count': 0, 'max_count': 1,
                           'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1,
                           'failure_cost': max(row['failure_cost'] for row in rows),
                           'failure_event': 'complementary_bin_state_label_unavailable', 'axes': axes})
        members[pair_id] = rows
    request = {'schema_version': 1, 'model_key': sample.TASK, 'target_split': 'train', 'budget': budget,
               'purpose': f'Full train complementary pairs for {kind}, {quantity} inserted bins',
               'independence_basis': 'eligible camera parent groups and shared Blender scene family',
               'uncertainty_weight': .25, 'strata': definition,
               'conditional_balance': {'axes': list(AXES), 'stratify_by': 'color_set', 'prior': 1, 'seed': seed}}
    result = sample.allocate(request, root=ROOT)
    states = []
    for row in result['strata']:
        if row['planned_count']:
            for member in members[row['id']]:
                states.append({**member, 'id': f'full_train_{offset+len(states):05d}', 'min_count': 1,
                               'pair_id': row['id']})
    if len(states) != 2 * budget:
        raise ValueError('Allocated pair count does not match the composition budget')
    return states, result


def compile_train(camera, blender, seed, additional):
    rng = random.Random(seed)
    _, source = sample.load_task_config(sample.TASK)
    default_manifests = (source['current_selection']['render_manifest'],
                         sample.compositor.rel(BASE / 'foreground_multi/manifest.json'))
    available = tuple(path for path in default_manifests if (ROOT / path).is_file())
    oblique = oblique_source()
    manifests = available + ((oblique,) if oblique else ()) + tuple(additional)
    foregrounds = foreground_pairs(manifests)
    backgrounds = load_backgrounds(source)
    print(json.dumps({'eligible_backgrounds': {kind: len(rows) for kind, rows in backgrounds.items()},
                      'foreground_pairs': {n: len(rows) for n, rows in foregrounds.items()}},
                     ensure_ascii=False), flush=True)
    requested = pair_budgets(camera, blender)
    strata, allocations, planned, rejections = [], defaultdict(dict), defaultdict(dict), Counter()
    proposed, selected = feasible_pair_budgets(requested, foregrounds, backgrounds, rng, rejections)
    for (kind, quantity), pairs in selected.items():
        if not pairs:
            continue
        proposed_pairs = proposed[kind, quantity]
        states, receipt = choose_pairs(kind, quantity, pairs, proposed_pairs, seed, len(strata))
        strata.extend(states)
        allocations[kind][quantity] = receipt
        planned[kind][quantity] = pairs
        print(json.dumps({'source_kind': kind, 'visible_inserted_bins': quantity, 'feasible_pairs': len(proposed_pairs),
                          'selected_pairs': pairs}, ensure_ascii=False), flush=True)
    if not strata:
        raise ValueError('No current foreground and background form a feasible complementary pair')
    by_group = Counter((row['axes']['source_kind'], row['background']['id']) for row in strata)
    result = {'schema_version': 1, 'model_key': sample.TASK, 'target_split': 'train', 'budget': len(strata),
              'purpose': 'Full train positive waste-bin composites on qualified camera and Blender backgrounds',
              'independence_basis': 'camera parent groups and a shared Blender scene/foreground asset family',
              'uncertainty_weight': .25, 'constraints': sample.CONSTRAINTS, 'strata': strata,
              'planning': {'seed': seed, 'pair_budgets': dict(planned), 'requested_pair_budgets': requested,
                           'requested_images': camera + blender, 'shortfall_images': camera + blender - len(strata),
                           'foreground_manifests': list(manifests),
                           'unavailable_default_foregrounds': sorted(set(default_manifests) - set(available)),
                           'background_candidate_counts': {kind: len(rows) for kind, rows in backgrounds.items()},
                           'used_background_counts': dict(Counter(kind for kind, _ in by_group)),
                           'max_images_from_one_background': {kind: max(n for (k, _), n in by_group.items() if k == kind)
                                                              for kind in planned},
                           'rejected_attempts': dict(rejections)}}
    sample.plan_production(result, root=ROOT)
    return result, {'pair_budgets': dict(planned), 'allocation_unit': 'two complementary images on the same background',
                    'allocations': dict(allocations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=800)
    parser.add_argument('--blender', type=int, default=2396)
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--additional-render-manifest', action='append', default=[])
    args = parser.parse_args()
    PLANS.mkdir(parents=True, exist_ok=True)
    plan, audit = compile_train(args.camera, args.blender, args.seed, args.additional_render_manifest)
    write_json(PLANS / 'train_composite_plan.json', plan)
    write_json(PLANS / 'train_composite_allocation.json', audit)
    print(json.dumps({'planned_images': plan['budget'], **plan['planning']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
