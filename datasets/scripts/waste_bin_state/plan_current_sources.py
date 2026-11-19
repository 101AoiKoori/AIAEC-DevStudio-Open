"""Compile the current bin composite plan from actual render masks and selected backgrounds.

This command writes only the task's current plan JSON. It never renders or
composites images and cannot substitute predicted masks for missing render output.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.blender_workspace import BINS_PHOTO_SCENE
from datasets.scripts.common.bayesian import plan_production
from datasets.scripts.common.paths import bin_state_foreground_dir, sample_dir
from datasets.scripts.waste_bin_state import generate_waste_v4_grouped as compositor
from datasets.scripts.waste_bin_state.qualified_pool import qualified_backgrounds


def _observed_placement(qualification, render_manifest, sample, mask_digest, masks):
    """Replay a reviewed nominal layout for the same render, if one was recorded."""
    reviewed = qualification['geometry'].get('validated_placement') or {}
    placement = reviewed.get('nominal_output_xywh')
    manifest = Path(render_manifest)
    render_paths = {(render_manifest, (manifest.parent / sample['image']).as_posix())}
    if manifest == bin_state_foreground_dir().relative_to(ROOT) / 'manifest.json':
        prior_dir = sample_dir('waste_bin_state').relative_to(ROOT)
        render_paths.add(((prior_dir / manifest.name).as_posix(), (prior_dir / sample['image']).as_posix()))
    if (not placement or (reviewed.get('render_manifest'), reviewed.get('render_image')) not in render_paths or
            reviewed.get('render_mask_visible_sha256') != mask_digest):
        return None
    compositor._placement(placement)
    left, top, width, height = placement
    crop_height, crop_width = next(iter(masks.values())).shape
    if abs(width / crop_width - height / crop_height) > max(1 / crop_width, 1 / crop_height):
        raise ValueError('reviewed placement does not fit the current render crop')
    _, supports = compositor._group_supports(masks, placement)
    return left, top, width, height, supports


def build(seed=1918, *, qualified_frame_ids=(), qualified_only=False, render_image=None, max_states=None):
    _, source = load_task_config('waste_bin_state')
    selection = source['current_selection']
    if (selection.get('qualified_pool', {}).get('split') != 'train' or
            selection['qualified_pool'].get('selection') != 'all_current_eligible'):
        raise ValueError('qualified pool selection differs from the task source contract')
    if qualified_only and not qualified_frame_ids:
        raise ValueError('qualified-only replay requires explicit frame IDs')
    render_manifest = ROOT / selection['render_manifest']
    if not render_manifest.is_file():
        raise ValueError('actual render manifest is missing; run config/render_plan.json and refresh the annotation index')
    rendered = json.loads(render_manifest.read_text(encoding='utf8'))
    if rendered.get('status') != 'complete_candidates' or rendered.get('split') != 'train':
        raise ValueError('a complete current train render receipt is required')
    if rendered.get('scene_file') != source['standalone_scene']:
        raise ValueError('render source differs from source.yaml')
    if rendered.get('scene_name') != BINS_PHOTO_SCENE:
        raise ValueError('current bin composition requires a Waste Bins Photo render receipt')
    candidates = qualified_backgrounds(source, 'train', frame_ids=qualified_frame_ids)
    _, eligible = compositor.eligible_ground_backgrounds(source, candidates, 'train')
    backgrounds = [(r, *eligible[r['id']]) for r in candidates if r['id'] in eligible]
    if not backgrounds:
        raise ValueError('no selected train background has eligible stored ground and labels')
    excluded_backgrounds = sorted(row['id'] for row in candidates if row['id'] not in eligible)
    rng = random.Random(seed)
    strata, excluded = [], []
    for sample in rendered['samples']:
        if render_image is not None and sample['image'] != render_image:
            continue
        names = sample.get('visible_bins', [])
        labels = sample.get('labels', [])
        if (not names or not sample.get('candidate_label') or not sample.get('mask_visible') or
                not sample.get('mask_state_visible') or any(item.get('label_eligible') is not True for item in labels)
                or len(labels) != len(names) or set(names) != {item['bin'] for item in labels}):
            excluded.append({'image': sample['image'], 'reason': 'empty_scene_or_incomplete_target_and_state_evidence'})
            continue
        image_path = render_manifest.parent/sample['image']
        mask_path = render_manifest.parent/sample['mask_visible']
        for path in (image_path, mask_path, render_manifest.parent/sample['mask_state_visible'],
                     render_manifest.parent/sample['candidate_label']):
            if not path.is_file():
                raise ValueError('render evidence is missing: '+str(path.relative_to(ROOT)))
        image = compositor.base.read_image(image_path)
        mask_digest = sha256(mask_path.read_bytes()).hexdigest()
        rgba = compositor.base.read_image(mask_path, cv2.IMREAD_UNCHANGED)
        if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.shape[:2] != image.shape[:2]:
            raise ValueError('render mask dimensions differ from the actual image')
        rgb = rgba[:, :, [2, 1, 0]]
        masks = {}
        for color in names:
            mask = rgba[:, :, 3] >= 191
            for channel, value in enumerate(compositor.base.COLORS[color]):
                mask &= rgb[:, :, channel] >= 191 if value else rgb[:, :, channel] <= 63
            if not mask.any():
                raise ValueError('selected bin mask is empty: '+color)
            masks[color] = mask
        union = np.logical_or.reduce(list(masks.values())).astype(np.uint8)
        x, y, width, height = cv2.boundingRect(union)
        crop = image[y:y+height, x:x+width]
        cropped_masks = {key: value[y:y+height, x:x+width] for key, value in masks.items()}
        for background, qualification, assessment in backgrounds:
            background = {**background,
                          'ground_polygon_xy': compositor.output_polygon(
                              qualification['geometry']['ground_polygon'], qualification['image_size']),
                          'existing_bins': compositor.existing_bin_labels(qualification)}
            frame = compositor.output_frame(compositor.base.read_image(ROOT/background['image']))
            floor = assessment.candidate_mask.astype(np.uint8)
            for tone, gain, saturation in (('normal', 1.0, 1.0), ('dark', .76, .72)):
                try:
                    reviewed = _observed_placement(
                        qualification, selection['render_manifest'], sample, mask_digest, cropped_masks)
                    if reviewed is None:
                        _, _, _, left, top, w, h, _, supports = compositor.ground_placement(
                            crop, cropped_masks, [(background, frame, floor)], rng)
                    else:
                        left, top, w, h, supports = reviewed
                    if any(assessment.contact_status(points[1], contact_width=points[2][0]-points[0][0])
                           != 'candidate' for points in supports.values()):
                        raise ValueError('planned support exceeds stored contact region')
                    compositor.reject_existing_bin_overlap(
                        [[left, top, left+w, top+h]], compositor.existing_bin_labels(qualification))
                except ValueError as error:
                    excluded.append({'image': sample['image'], 'background': background['id'],
                                     'tone': tone, 'reason': str(error)})
                    continue
                state = {'id': f'composite_{len(strata):05d}', 'conditions': [],
                    'min_count': 1, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
                    'exposure_weight': 1, 'failure_cost': 1, 'failure_event': 'bin_state_label_unavailable',
                    'background': {'manifest': background.get('manifest', source['background_manifest']),
                                   'id': background['id'],
                                   'frame_id': qualification['frame_id'],
                                   'record_id': qualification['record_id']},
                    'target': {'render_manifest': selection['render_manifest'], 'render_image': sample['image'],
                               'bins': {item['bin']: item['state'] for item in labels}},
                    'labels': {'mode': 'candidate', 'open': 0, 'closed': 1},
                    'augmentation': {'value_gain': gain, 'saturation_gain': saturation, 'shadow': 'simulated_gaussian'},
                    'parameters': {'placement_xywh': [left, top, w, h]},
                    'sources': [background['image'], image_path.relative_to(ROOT).as_posix(), mask_path.relative_to(ROOT).as_posix()]}
                strata.append(state)
    if max_states is not None:
        if max_states < 1:
            raise ValueError('max-states must be positive')
        strata = strata[:max_states]
    if not strata:
        raise ValueError('render contains no eligible nonempty source; no production plan was written')
    result = {'schema_version': 1, 'model_key': 'waste_bin_state', 'target_split': 'train',
        'budget': len(strata), 'purpose': 'Current real-background compositions from actual rendered bin evidence',
        'independence_basis': 'one source model and selected recorded background parents; not independent capture',
        'uncertainty_weight': 0, 'constraints': {'resolution': [640, 480], 'label_mode': 'candidate', 'ground_placement': 'stored_contact_mask'},
        'strata': strata, 'planning': {'seed': seed, 'render_manifest': selection['render_manifest'],
                                      'excluded_render_samples': excluded,
                                      'background_without_stored_qualification': excluded_backgrounds,
                                      'background_count': len(backgrounds),
                                      'qualified_frame_ids': list(qualified_frame_ids)}}
    planned = plan_production(result, root=ROOT)
    compositor._planned_inputs(planned)
    return result


def check_existing_backgrounds(split):
    """Read current complete frames through the production background adapter."""
    _, source = load_task_config('waste_bin_state')
    database = ROOT / source['databases']['frame_environment']
    rows = compositor.query_frames(task='waste_bin_state', role='composite_background',
                                   split=split, database=database)
    frames = []
    for row in rows:
        image = row.get('source_image') or row['paths'][0]
        if row['label'].get('presence') != 'present':
            continue
        frames.append({'id': row['frame_id'], 'image': image,
                       'parent_image': row.get('parent_image') or image,
                       'video': row.get('parent_video'), 'split': split})
    _, eligible = compositor.eligible_ground_backgrounds(source, frames, split)
    if len(eligible) != len(frames) or not frames:
        raise ValueError('positive background qualification is incomplete for ' + split)
    instances = 0
    for frame in frames:
        review, assessment = eligible[frame['id']]
        actual = compositor.output_frame(compositor.base.read_image(ROOT / frame['image']))
        if actual.shape != (480, 640, 3) or assessment.candidate_mask.shape != (480, 640):
            raise ValueError('positive background transform is inconsistent')
        instances += len(compositor.existing_bin_labels(review))
    return {'split': split, 'frames': len(frames), 'retained_bin_labels': instances,
            'read_only': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1918)
    parser.add_argument('--check-existing-backgrounds', choices=('train', 'val', 'test'))
    parser.add_argument('--qualified-frame-id', action='append', default=[])
    parser.add_argument('--qualified-only', action='store_true')
    parser.add_argument('--render-image')
    parser.add_argument('--max-states', type=int)
    parser.add_argument('--output-plan', type=Path)
    args = parser.parse_args()
    if args.check_existing_backgrounds:
        print(json.dumps(check_existing_backgrounds(args.check_existing_backgrounds), ensure_ascii=False))
        return
    _, source = load_task_config('waste_bin_state')
    expected = ROOT/'datasets/scripts/waste_bin_state/config/production_plan.json'
    output = (args.output_plan or expected).resolve()
    if not output.is_relative_to((ROOT/'datasets/scripts/waste_bin_state/config').resolve()) or output.suffix != '.json':
        raise ValueError('bin plan output must be a task config JSON')
    if args.output_plan is None and (ROOT/source['production_plan']).resolve() != expected:
        raise ValueError('current bin plan must use its fixed task config path')
    result = build(args.seed, qualified_frame_ids=args.qualified_frame_id,
                   qualified_only=args.qualified_only, render_image=args.render_image,
                   max_states=args.max_states)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'planned_images': result['budget'], 'plan': output.relative_to(ROOT).as_posix()}))


if __name__ == '__main__':
    main()
