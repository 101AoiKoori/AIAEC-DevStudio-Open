"""Compile reserved val/test foregrounds and waste-bin composition plans."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from math import atan2, cos, degrees, radians, sin
from pathlib import Path
import random
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.scripts.common.bayesian import plan_production
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.frame_environment import query_frames
from datasets.scripts.common.paths import bin_state_foreground_dir
from datasets.scripts.common.source_history import historical_source, read_source_history
from datasets.scripts.common.source_ledger.config import build_groups, normalize_path
from datasets.scripts.common.source_ledger.reservations import candidate_reservation
from datasets.scripts.waste_bin_state import generate_waste_v4_grouped as compositor
from datasets.scripts.waste_bin_state.bin_render_contract import standalone_planned_jobs
from datasets.scripts.waste_bin_state.plan_representative_sample import (
    CONSTRAINTS, TONES, foregrounds, make_state, pair_selection, prepared_background,
)
from datasets.scripts.waste_bin_state.qualified_pool import qualified_backgrounds


TASK = 'waste_bin_state'
COLORS = ('blue', 'red', 'green', 'gray')
BIN_X = dict(zip(COLORS, (-1.68, -.56, .56, 1.68)))
EVAL_SPLITS = ('val', 'test')
SEEDS = {'val': 20261118, 'test': 20261119}
DEFAULT_IMAGES = {'val': 128, 'test': 192}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.json.tmp')
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def render_declaration(split):
    """Hold out actual camera and appearance settings for both evaluation splits."""
    _, source = load_task_config(TASK)
    train = read_json(ROOT / source['render_plan'])
    single = [(color, (color,), ('open',)) for color in COLORS]
    if split == 'val':
        groups = [('two', ('blue', 'red'), ('open', 'closed')),
                  ('three', ('red', 'green', 'gray'), ('open', 'closed', 'open'))]
        yaw_angles = (-11, 14, -18, 8, -15, 13, 12, -9)
    else:
        groups = [('two', ('green', 'gray'), ('open', 'closed')),
                  ('three', ('blue', 'red', 'green'), ('closed', 'open', 'closed'))]
        yaw_angles = (22, -21, 17, -24, 19, -17, -20, 21)
    groups += [('four_mixed', COLORS, ('open', 'closed', 'open', 'closed')),
               ('four_uniform', COLORS, ('open',) * 4)]
    strata = []
    for index, ((name, selected, lid_states), yaw) in enumerate(zip(single + groups, yaw_angles)):
        width = len(selected)
        center = (min(BIN_X[c] for c in selected) + max(BIN_X[c] for c in selected)) / 2
        distance = 5.45 + 0.28 * (width - 1) + (0.23 if split == 'test' else 0)
        eye_z = 2.12 + .08 * (width - 1) + (.11 if split == 'test' else 0)
        params = {'camera_location': [round(center + distance * sin(radians(yaw)), 5),
                                     round(-distance * cos(radians(yaw)), 5), round(eye_z, 5)],
                  'camera_aim': [center, 0, .83 if width == 1 else .9], 'yaw_degrees': 0,
                  'lens_mm': (36 if split == 'val' else 37) if width == 1 else (31 if width == 4 else 34),
                  'key_energy': (790 if split == 'val' else 710) + (index % 2) * 64,
                  'fill_energy': (445 if split == 'val' else 405) + (index % 2) * 28,
                  'color_factor': .97 if split == 'val' else .91, 'roughness_delta': .035 if split == 'val' else -.025}
        for complement in (False, True):
            states = tuple(('open' if state == 'closed' else 'closed') for state in lid_states) if complement else lid_states
            row = deepcopy(train['strata'][0])
            row.update(id=f'{split}_{name}_{int(complement)}', parameters=params,
                       target={'bins': dict(zip(selected, states)), 'open_pose': 'hanging_back'})
            row['background'].update(floor_gray=.13 if split == 'val' else .10,
                                     backdrop_gray=.31 if split == 'val' else .38)
            strata.append(row)
    return {**train, 'target_split': split, 'budget': len(strata), 'strata': strata,
            'purpose': f'{split} foregrounds rendered from reserved Waste Bins Photo camera configurations',
            'independence_basis': 'split-specific foreground renders from the shared Blender asset family'}


def evaluation_backgrounds(source, split, rng):
    database = ROOT / source['databases'][source['current_selection']['qualified_pool']['database']]
    history = read_source_history(ROOT)
    groups = build_groups(TASK, source['pairing'])
    other_splits = ('train', 'val') if split == 'test' else ('train',)
    other_frames, other_hashes = set(), set()
    parent_splits = defaultdict(set)
    for other_split in other_splits:
        for role in ('direct_target', 'empty_negative', 'composite_background'):
            for row in query_frames(task=TASK, role=role, split=other_split, database=database):
                other_frames.add(row['frame_id'])
                other_hashes.add(row['content_sha256'])
                if parent := row.get('parent_group'):
                    parent_splits[historical_source(normalize_path(parent), history)].add(other_split)
    rows = qualified_backgrounds(source, split)
    database, eligible = compositor.eligible_ground_backgrounds(source, rows, split)
    known = compositor.build_queue(ROOT)
    grouped = defaultdict(list)
    contacts, rejected = {}, Counter()
    for row in rows:
        if row['id'] not in eligible:
            rejected['no_current_usable_ground_assessment'] += 1
            continue
        qualification, assessment = eligible[row['id']]
        reservation = candidate_reservation(qualification, split, groups, history)
        if reservation['status'] == 'conflict':
            rejected['source_yaml_parent_reservation_conflict'] += 1
            continue
        if row['frame_id'] in other_frames or qualification['content_sha256'] in other_hashes:
            rejected['camera_frame_assigned_to_earlier_split'] += 1
            continue
        if compositor.source_role(ROOT, ROOT / row['parent_image'], known)[0] == 'regression':
            rejected['existing_regression_only_source'] += 1
            continue
        parent = historical_source(normalize_path(row['parent_group']), history)
        if earlier := parent_splits.get(parent):
            contacts[row['parent_group']] = sorted(earlier)
        prepared = prepared_background(row, qualification, assessment)
        placement = (qualification['geometry'].get('validated_placement') or {}).get('output_xywh')
        if placement and prepared['tested_height_range_output_px'] is None:
            height = placement[3]
            upper = min(prepared['height_range'][1], height + 12)
            prepared['height_range'] = [max(50, height - 12), upper]
        contact_mask = np.zeros(assessment.candidate_mask.shape, dtype=np.uint8)
        polygon = np.asarray(prepared['ground_polygon_xy'], dtype=np.int32)
        cv2.fillPoly(contact_mask, [polygon], 1)
        prepared['contact_mask'] = contact_mask & assessment.candidate_mask
        prepared['contact_extent'] = cv2.boundingRect(contact_mask)
        prepared['perspective_scale_observation'] = compositor.current_scale_observation(
            database, row['frame_id'], prepared['source_image_size'])
        grouped[row['parent_group']].append(prepared)
    for group in grouped.values():
        rng.shuffle(group)
    ordered = [group[index] for index in range(max(map(len, grouped.values()), default=0))
               for group in grouped.values() if index < len(group)]
    return ordered, {key: len(group) for key, group in grouped.items()}, rejected, contacts


def set_views(renders, manifest):
    """The standalone renderer supplies measured axes via camera and aim vectors."""
    actual = {row['image']: row['actual'] for row in read_json(manifest)['samples']}
    for row in renders:
        camera = actual[row['image']]['camera_location']
        aim = actual[row['image']]['camera_aim']
        yaw = degrees(atan2(camera[0] - aim[0], aim[1] - camera[1]))
        row['view'] = 'oblique' if abs(yaw) >= 10 else 'frontal'


def count_complementary_pairs(states):
    """Count the capacity consumed by the existing paired allocator."""
    signatures = Counter()
    for row in states:
        bins = row['target']['bins']
        signature = tuple(sorted((color, state == 'open') for color, state in bins.items()))
        signatures[row['axes']['source_kind'], signature] += 1
    capacity = 0
    for (kind, key), size in signatures.items():
        opposite = tuple((color, not state) for color, state in key)
        if key < opposite:
            capacity += min(size, signatures[kind, opposite])
    return capacity


def contact_positions(background, offsets, width, height, rng):
    """Bound feet by the reviewed polygon and the visible crop by image content."""
    polygon_x, polygon_y, polygon_w, polygon_h = background['contact_extent']
    x0, y0, x1, y1 = background['output_content_xyxy']
    left = max(x0, polygon_x - min(x for x, _ in offsets))
    right = min(x1 - width, polygon_x + polygon_w - 1 - max(x for x, _ in offsets))
    top = max(y0, polygon_y - min(y for _, y in offsets))
    bottom = min(y1 - height, polygon_y + polygon_h - 1 - max(y for _, y in offsets))
    if left > right or top > bottom:
        return ()
    mask = background['contact_mask']
    valid = np.logical_and.reduce([mask[top+y:bottom+y+1, left+x:right+x+1] for x, y in offsets])
    ys, xs = np.nonzero(valid)
    sampled = rng.sample(range(len(xs)), min(24, len(xs)))
    return [(int(xs[index]) + left, int(ys[index]) + top) for index in sampled]


def placement_for_eval(foreground, background, rng, raster_cache):
    """Search projected contact points for small audited regions and retained bins."""
    start, end = background['height_range']
    if start > end:
        raise ValueError('stored height interval leaves no candidate')
    assessment = background['assessment']
    height_options = sorted({start, end, *np.linspace(start, end, 5, dtype=int).tolist()})
    rng.shuffle(height_options)
    crop_h, crop_w = foreground['crop'].shape[:2]
    cont_x0, cont_y0, cont_x1, cont_y1 = background['output_content_xyxy']
    for height in height_options:
        width = round(height * crop_w / crop_h)
        if not 0 < width <= cont_x1 - cont_x0:
            continue
        key = foreground['path'], height
        if key not in raster_cache:
            evidence = [cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                        for mask, _ in foreground['evidence'].values()]
            readable = all(mask.sum() >= 24 and (not opening or compositor.mouth_thickness(mask) >= compositor.MIN_MOUTH_THICKNESS_PX)
                           for mask, (_, opening) in zip(evidence, foreground['evidence'].values()))
            _, offsets = compositor._group_supports(foreground['masks'], [0, 0, width, height])
            raster_cache[key] = readable, [point for points in offsets.values() for point in points]
        readable, offsets = raster_cache[key]
        if not readable:
            continue
        for x, y in contact_positions(background, offsets, width, height, rng):
            planned = [x, y, width, height]
            placement, scaled, feet, _ = compositor._scaled_group_layout(
                foreground['masks'], planned, background, assessment, foreground['evidence'])
            px, py, pw, ph = placement
            if not (cont_x0 <= px < px + pw <= cont_x1 and cont_y0 <= py < py + ph <= cont_y1):
                continue
            if any(assessment.contact_status(points[1], contact_width=points[2][0] - points[0][0]) != 'candidate'
                   for points in feet.values()):
                continue
            boxes = []
            for raster in scaled.values():
                bx, by, bw, bh = cv2.boundingRect(raster)
                boxes.append([px+bx, py+by, px+bx+bw, py+by+bh])
            try:
                compositor.reject_existing_bin_overlap(boxes, background['existing_bins'])
            except ValueError:
                continue
            return planned, placement, boxes
    raise ValueError('no full visible lid and valid floor contact in the reviewed polygon')


def prepare_composites(split, requested):
    if requested < 8 or requested % 2:
        raise ValueError('requested evaluation budget must be even and at least eight')
    _, source = load_task_config(TASK)
    rng = random.Random(SEEDS[split])
    backgrounds, parents, excluded, contacts = evaluation_backgrounds(source, split, rng)
    path = bin_state_foreground_dir(split) / 'manifest.json'
    if read_json(path)['split'] != split:
        raise ValueError('evaluation foreground manifest belongs to a different split')
    renders = list(foregrounds((compositor.rel(path),), split=split))
    set_views(renders, path)
    expanded, rejections, raster_cache = [], Counter(), {}
    tones = tuple(TONES)
    for bg in backgrounds:
        for foreground in renders:
            try:
                placement = placement_for_eval(foreground, bg, rng, raster_cache)
            except ValueError as error:
                rejections[str(error)] += 1
                continue
            state = make_state(len(expanded), foreground, bg, placement, rng.choice(tones))
            state['id'] = f'{split}_{state["id"]}'
            expanded.append(state)
    budget = min(requested, 2 * count_complementary_pairs(expanded))
    request = {
        'schema_version': 1, 'model_key': TASK, 'target_split': split, 'budget': budget,
        'purpose': f'Full {split} coverage from current eligible backgrounds',
        'independence_basis': ('split-specific renders from one shared Blender asset family; '
                               'explicit camera parent reservations and identical-frame separation; '
                               'candidate parent contacts are reported'),
        'uncertainty_weight': .25, 'constraints': CONSTRAINTS, 'strata': expanded,
        'planning': {'seed': SEEDS[split], 'requested_images': requested,
                     'background_candidates_by_parent': parents, 'background_rejections': dict(excluded),
                     'background_parent_contacts': contacts,
                     'placement_rejection_counts': dict(rejections), 'foreground_candidate_count': len(renders)},
    }
    declaration, planned, audit = pair_selection(request)
    compositor._planned_inputs(planned)
    stage = path.parent.parent
    write_json(stage / 'composite_plan.json', declaration)
    write_json(stage / 'composite_allocation.json', audit)
    source_parents = {bg['id']: bg['parent_group'] for bg in backgrounds}
    actual_parents = Counter(source_parents[row['background']['id']] for row in declaration['strata'])
    print(json.dumps({'split': split, 'requested': requested, 'planned': budget,
                      'feasible_candidates': len(expanded), 'background_exclusions': dict(excluded),
                      'assigned_images_by_parent': dict(actual_parents)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('foreground', 'composites'))
    parser.add_argument('--split', choices=EVAL_SPLITS, required=True)
    parser.add_argument('--budget', type=int)
    args = parser.parse_args()
    stage = bin_state_foreground_dir(args.split).parent
    if args.mode == 'foreground':
        declaration = render_declaration(args.split)
        planned = plan_production(declaration, root=ROOT)
        _, source = load_task_config(TASK)
        standalone_planned_jobs(planned, source['standalone_scene'], 640, 480, .70, 24)
        write_json(stage / 'render_plan.json', declaration)
        print(json.dumps({'split': args.split, 'foreground_budget': declaration['budget'],
                          'render_plan': compositor.rel(stage / 'render_plan.json')}, ensure_ascii=False))
    else:
        prepare_composites(args.split, args.budget if args.budget is not None else DEFAULT_IMAGES[args.split])


if __name__ == '__main__':
    main()
