"""Allocate a representative subset of feasible train bin composites."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.scripts.common.bayesian import allocate, plan_production
from datasets.scripts.common.blender_workspace import BINS_PHOTO_SCENE
from datasets.scripts.common.config import load_task_config
from datasets.scripts.waste_bin_state import generate_waste_v4_grouped as compositor
from datasets.scripts.waste_bin_state.bin_render_contract import decode_mask
from datasets.scripts.waste_bin_state.qualified_pool import qualified_backgrounds


TASK = 'waste_bin_state'
PLAN_PATH = ROOT / 'datasets/annotated/waste_bin_state/sample_composite_plan.json'
ALLOCATION_PATH = PLAN_PATH.with_name('sample_composite_allocation.json')
TONES = {
    'normal': (1.0, 1.0), 'dim': (.83, 1.0), 'desaturated': (1.0, .78),
    'dim_desaturated': (.86, .82),
}
CONSTRAINTS = {'resolution': [640, 480], 'label_mode': 'candidate', 'ground_placement': 'stored_contact_mask'}


def source_kind(background):
    return 'blender_route_composite' if '/backgrounds/rendered/' in background['image'] else 'real_background_composite'


def background_choices(source, rng, routes_per_view=2):
    """Limit near-identical trajectory repeats while keeping all real candidates."""
    choices = defaultdict(list)
    for row in qualified_backgrounds(source, 'train'):
        path = Path(row['image'])
        kind = source_kind(row)
        view = (path.parent.as_posix(), path.stem.rsplit('_f', 1)[0]) if kind == 'blender_route_composite' else None
        choices[kind, view].append(row)
    result = []
    for (kind, view), rows in choices.items():
        rng.shuffle(rows)
        result.extend(rows if view is None else rows[:routes_per_view])
    return result


def foregrounds(manifests, *, split='train'):
    for manifest in manifests:
        path = compositor._workspace_file(manifest)
        report = json.loads(path.read_text(encoding='utf8'))
        if (report.get('status') != 'complete_candidates' or report.get('split') != split
                or report.get('scene_name') != BINS_PHOTO_SCENE):
            raise ValueError(f'foreground must have a complete {split} render receipt from Waste Bins Photo')
        for sample in report['samples']:
            labels = sample.get('labels', [])
            bins = {item['bin']: item['state'] for item in labels}
            if not labels or not sample.get('candidate_label') or not all(item.get('label_eligible') for item in labels):
                continue
            if set(bins) != set(sample.get('visible_bins', [])) or len(bins) != len(labels):
                raise ValueError('foreground label set differs from its recorded visible bins')
            if 'open' in bins.values() and sample.get('open_pose') != 'hanging_back':
                raise ValueError('sample foreground must use completely folded-back open lids')
            image_path = (path.parent / sample['image']).resolve()
            mask_path = (path.parent / sample['mask_visible']).resolve()
            image = compositor.base.read_image(image_path)
            whole = compositor.base.read_image(mask_path, cv2.IMREAD_UNCHANGED)
            states = compositor.base.read_image(path.parent / sample['mask_state_visible'], cv2.IMREAD_UNCHANGED)
            if whole.shape != states.shape or whole.shape != (*image.shape[:2], 4):
                raise ValueError('image and whole/state masks disagree')
            masks = {color: decode_mask(whole[:, :, [2, 1, 0, 3]] / 255, color) for color in bins}
            if not all(mask.any() for mask in masks.values()):
                raise ValueError('foreground contains an empty whole-bin mask')
            union = np.logical_or.reduce(list(masks.values())).astype(np.uint8)
            x, y, width, height = cv2.boundingRect(union)
            crop = np.s_[y:y+height, x:x+width]
            evidence = {color: (decode_mask(states[crop][:, :, [2, 1, 0, 3]] / 255, color).astype(np.uint8),
                                state == 'open') for color, state in bins.items()}
            actual = sample.get('actual', {})
            view = sample.get('camera', actual.get('camera', {})).get('horizontal_deg', 0)
            yield {
                'manifest': manifest, 'image': sample['image'], 'path': compositor.rel(image_path),
                'mask_path': compositor.rel(mask_path), 'crop': image[crop], 'masks': {c:m[crop] for c,m in masks.items()},
                'evidence': evidence, 'bins': bins, 'view': 'oblique' if abs(view) >= 10 else 'frontal',
                'occlusion': 'neighbor_overlap' if any(item['visible_fraction'] < .97 for item in labels) else 'separated',
            }


def prepared_background(background, qualification, assessment):
    geometry = qualification['geometry']
    bounds = (geometry.get('validated_placement') or {}).get('tested_height_range_output_px')
    maximum = geometry.get('max_apparent_height_fraction')
    height_range = list(bounds or (75, 135))
    if maximum:
        limit = compositor.max_output_height_from_source_fraction(maximum, qualification['image_size'])
        height_range[1] = min(height_range[1], int(limit))
    return {
        **background,
        'ground_polygon_xy': compositor.output_polygon(geometry['ground_polygon'], qualification['image_size']),
        'output_content_xyxy': compositor.output_content_rectangle(qualification['image_size']),
        'source_image_size': qualification['image_size'], 'existing_bins': compositor.existing_bin_labels(qualification),
        'tested_height_range_output_px': bounds, 'max_apparent_height_fraction': maximum,
        'height_range': height_range, 'assessment': assessment,
    }


def placement_for(foreground, background, rng):
    assessment = background['assessment']
    floor = assessment.candidate_mask.astype(np.uint8)
    height_range = background['height_range']
    if height_range[0] > height_range[1]:
        raise ValueError('stored height interval leaves no candidate')
    for _ in range(3):
        _, _, _, left, top, width, height, _, _ = compositor.ground_placement(
            foreground['crop'], foreground['masks'], [(background, None, floor)], rng,
            height_range=height_range)
        request = [left, top, width, height]
        if not height_range[0] <= height <= height_range[1]:
            continue
        placement, scaled, feet, _ = compositor._scaled_group_layout(
            foreground['masks'], request, background, assessment, foreground['evidence'])
        x, y, w, h = placement
        x0, y0, x1, y1 = background['output_content_xyxy']
        if not (x0 <= x < x + w <= x1 and y0 <= y < y + h <= y1):
            continue
        if any(assessment.contact_status(points[1], contact_width=points[2][0]-points[0][0]) != 'candidate'
               for points in feet.values()):
            continue
        diluted = [cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                   for mask, _ in foreground['evidence'].values()]
        if any(mask.sum() < 24 or (opening and compositor.mouth_thickness(mask) < compositor.MIN_MOUTH_THICKNESS_PX)
               for mask, (_, opening) in zip(diluted, foreground['evidence'].values())):
            continue
        boxes = []
        for raster in scaled.values():
            bx, by, bw, bh = cv2.boundingRect(raster)
            boxes.append([x+bx, y+by, x+bx+bw, y+by+bh])
        compositor.reject_existing_bin_overlap(boxes, background['existing_bins'])
        return request, placement, boxes
    raise ValueError('no tested placement keeps all lid evidence at final size')


def make_state(index, foreground, background, layout, tone):
    request, applied, boxes = layout
    kind = source_kind(background)
    bins = foreground['bins']
    gain, saturation = TONES[tone]
    center_x = applied[0] + applied[2]/2
    center = 'left' if center_x < 255 else 'right' if center_x > 385 else 'center'
    mean_height = sum(y2-y1 for _,y1,_,y2 in boxes) / len(boxes)
    size = 'small' if mean_height < 90 else 'middle' if mean_height < 120 else 'large'
    lid_layout = 'mixed' if len(set(bins.values())) > 1 else next(iter(bins.values()))
    label_signature = ','.join(f'{color}:{state}' for color,state in sorted(bins.items()))
    return {
        'id': f'representative_{index:04d}', 'conditions': [], 'min_count': 0, 'max_count': 1,
        'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1,
        'failure_cost': 1.1 if foreground['occlusion'] == 'neighbor_overlap' else 1,
        'failure_event': 'bin_state_label_unavailable',
        'axes': {'source_and_quantity': f'{kind}:{len(bins)}', 'source_kind': kind,
                 'quantity': len(bins), 'lid_layout': lid_layout, 'color_lid': label_signature,
                 'tone': tone, 'position': center, 'size': size, 'view': foreground['view'],
                 'occlusion': foreground['occlusion']},
        'background': {key: background[key] for key in ('manifest', 'id', 'frame_id', 'record_id')},
        'target': {'render_manifest': foreground['manifest'], 'render_image': foreground['image'], 'bins': bins},
        'labels': {'mode': 'candidate', 'open': 0, 'closed': 1},
        'augmentation': {'value_gain': gain, 'saturation_gain': saturation, 'shadow': 'simulated_gaussian'},
        'parameters': {'placement_xywh': request},
        'sources': [background['image'], foreground['path'], foreground['mask_path']],
    }


def declaration(budget, seed, additional_manifests):
    _, source = load_task_config(TASK)
    rng = random.Random(seed)
    background_rows = background_choices(source, rng)
    database, eligible = compositor.eligible_ground_backgrounds(source, background_rows, 'train')
    known = compositor.build_queue(ROOT)
    backgrounds = defaultdict(list)
    for row in background_rows:
        if row['id'] not in eligible or compositor.source_role(ROOT, ROOT/row['parent_image'], known)[0] == 'regression':
            continue
        background = prepared_background(row, *eligible[row['id']])
        background['perspective_scale_observation'] = compositor.current_scale_observation(
            database, row['frame_id'], background['source_image_size'])
        backgrounds[source_kind(row)].append(background)
    manifests = (source['current_selection']['render_manifest'], *additional_manifests)
    strata, rejections = [], Counter()
    for foreground in foregrounds(manifests):
        for kind, frames in sorted(backgrounds.items()):
            order = rng.sample(frames, len(frames))
            accepted = 0
            for background in order:
                try:
                    layout = placement_for(foreground, background, rng)
                except ValueError as error:
                    rejections[str(error)] += 1
                    continue
                strata.append(make_state(len(strata), foreground, background, layout, rng.choice(list(TONES))))
                accepted += 1
                if accepted == 3:
                    break
    return {
        'schema_version': 1, 'model_key': TASK, 'target_split': 'train', 'budget': budget,
        'purpose': 'A reduced bin-state dataset sample with real and rendered backgrounds',
        'independence_basis': 'camera parent groups and one shared Blender scene/asset family',
        'uncertainty_weight': .25, 'constraints': CONSTRAINTS, 'strata': strata,
        'planning': {'seed': seed, 'foreground_manifests': list(manifests),
                     'background_candidate_counts': {key: len(value) for key,value in backgrounds.items()},
                     'placement_rejection_counts': dict(rejections)},
    }


def pair_selection(request):
    """Allocate in two-image blocks so each selected color has both states."""
    budget = request['budget']
    if budget < 8 or budget % 2:
        raise ValueError('the composite budget must be even and at least eight')
    candidates = request['strata']
    groups = defaultdict(lambda: ([], []))
    for row in candidates:
        signature = tuple(sorted((color, state == 'open') for color,state in row['target']['bins'].items()))
        complement = tuple((color, not opening) for color,opening in signature)
        key = (row['axes']['source_kind'], min(signature, complement))
        groups[key][int(signature > complement)].append(row)
    pairs, definitions = {}, []
    axes = ['source_and_quantity', 'color_lid', 'tone', 'position', 'size', 'view', 'occlusion']
    for group_key, (a_rows, b_rows) in groups.items():
        for a in sorted(a_rows, key=lambda row: row['axes']['view'] != 'oblique'):
            if not b_rows:
                break
            b = min(b_rows, key=lambda row: (a['background']['id'] != row['background']['id'],
                                            a['axes']['view'] != row['axes']['view'],
                                            a['axes']['occlusion'] != row['axes']['occlusion'],
                                            a['axes']['position'] != row['axes']['position']))
            b_rows.remove(b)
            pair_id = f'lid_pair_{len(pairs):04d}'
            pairs[pair_id] = a, b
            values = {axis: '+'.join(sorted({a['axes'][axis], b['axes'][axis]}))
                      for axis in axes}
            definitions.append({'id': pair_id, 'conditions': [], 'min_count': 0, 'max_count': 1,
                                'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1,
                                'failure_cost': max(a['failure_cost'], b['failure_cost']),
                                'failure_event': 'complementary_bin_state_label_unavailable', 'axes': values})
    reserved = Counter()
    for color in compositor.COLORS:
        choices = [row for row in definitions if set(pairs[row['id']][0]['target']['bins']) == {color}]
        if not choices:
            raise ValueError('no feasible complementary single-bin pair for ' + color)
        row = min(choices, key=lambda row: reserved[row['axes']['source_and_quantity']])
        row['min_count'] = 1
        reserved[row['axes']['source_and_quantity']] += 1
    spec = {'axes': axes, 'stratify_by': 'source_and_quantity', 'prior': 1, 'seed': request['planning']['seed']}
    pair_request = {**request, 'budget': budget//2, 'strata': definitions, 'conditional_balance': spec}
    allocation = allocate(pair_request, root=ROOT)
    selected = {row['id'] for row in allocation['strata'] if row['planned_count']}
    strata = [{**row, 'min_count': 1, 'pair_id': pair_id}
              for pair_id, pair in pairs.items() if pair_id in selected for row in pair]
    planning = {**request['planning'], 'feasible_image_candidates': len(candidates),
                'feasible_pair_candidates': len(pairs), 'allocation_unit': 'one complementary lid pair (two frames)',
                'reserved_single_color_pairs': dict(reserved), 'pair_balance': spec}
    final = {**request, 'strata': strata, 'planning': planning}
    execution = plan_production(final, root=ROOT)
    audit = {
        **execution,
        'pair_allocation': allocation,
        'pair_members': {key: [row['id'] for row in pair] for key,pair in pairs.items()},
        'candidate_frames': candidates,
    }
    return final, execution, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--additional-render-manifest', action='append', default=[])
    args = parser.parse_args()
    candidates = declaration(args.budget, args.seed, args.additional_render_manifest)
    request, planned, audit = pair_selection(candidates)
    compositor._planned_inputs(planned)
    PLAN_PATH.write_text(json.dumps(request, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    ALLOCATION_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    selected = {row['id'] for row in planned['strata'] if row['planned_count']}
    counts = Counter(row['axes']['source_and_quantity'] for row in request['strata'] if row['id'] in selected)
    print(json.dumps({'planned_images': request['budget'], 'feasible_candidates': request['planning']['feasible_pair_candidates'],
                      'planned_by_source_and_quantity': dict(counts), 'plan': compositor.rel(PLAN_PATH)}))


if __name__ == '__main__':
    main()
