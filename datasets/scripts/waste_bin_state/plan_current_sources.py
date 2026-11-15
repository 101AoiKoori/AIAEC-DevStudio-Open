"""Compile the current bin composite plan from actual render masks and selected backgrounds.

This command writes only the task's current plan JSON. It never renders or
composites images and cannot substitute predicted masks for missing render output.
"""
import argparse
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
from datasets.scripts.common.bayesian import plan_production
from datasets.scripts.waste_bin_state import generate_waste_v4_grouped as compositor


def build(seed=1918):
    _, source = load_task_config('waste_bin_state')
    selection = source['current_selection']
    render_manifest = ROOT / selection['render_manifest']
    if not render_manifest.is_file():
        raise ValueError('actual render manifest is missing; run config/render_plan.json after production resumes, then refresh the sample index')
    rendered = json.loads(render_manifest.read_text(encoding='utf8'))
    if rendered.get('status') != 'complete_candidates' or rendered.get('split') != 'train':
        raise ValueError('a complete current train render receipt is required')
    if rendered.get('scene_file') != source['standalone_scene']:
        raise ValueError('render source differs from source.yaml')
    all_backgrounds = json.loads((ROOT/source['background_manifest']).read_text(encoding='utf8'))['frames']
    backgrounds = [r for r in all_backgrounds if r['id'] in selection['background_ids']]
    if len(backgrounds) != len(selection['background_ids']) or any(r['split'] != 'train' for r in backgrounds):
        raise ValueError('selected train backgrounds differ from the configured manifest')
    rng = random.Random(seed)
    strata, excluded = [], []
    for sample in rendered['samples']:
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
        for background in backgrounds:
            frame = compositor.base.read_image(ROOT/background['image'])
            if frame.shape != (480, 640, 3):
                raise ValueError('background differs from its 640x480 contract')
            floor = np.zeros((480, 640), np.uint8)
            cv2.fillPoly(floor, [np.asarray(background['ground_polygon_xy'], np.int32)], 1)
            floor = cv2.erode(floor, np.ones((11, 11), np.uint8))
            for tone, gain, saturation in (('normal', 1.0, 1.0), ('dark', .76, .72)):
                try:
                    _, _, _, left, top, w, h, _, _ = compositor.ground_placement(
                        crop, cropped_masks, [(background, frame, floor)], rng)
                except ValueError as error:
                    excluded.append({'image': sample['image'], 'background': background['id'],
                                     'tone': tone, 'reason': str(error)})
                    continue
                state = {'id': f'composite_{len(strata):05d}', 'conditions': [],
                    'min_count': 1, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
                    'exposure_weight': 1, 'failure_cost': 1, 'failure_event': 'bin_state_label_unavailable',
                    'background': {'manifest': source['background_manifest'], 'id': background['id']},
                    'target': {'render_manifest': selection['render_manifest'], 'render_image': sample['image'],
                               'bins': {item['bin']: item['state'] for item in labels}},
                    'labels': {'mode': 'candidate', 'open': 0, 'closed': 1},
                    'augmentation': {'value_gain': gain, 'saturation_gain': saturation, 'shadow': 'simulated_gaussian'},
                    'parameters': {'placement_xywh': [left, top, w, h]},
                    'sources': [background['image'], image_path.relative_to(ROOT).as_posix(), mask_path.relative_to(ROOT).as_posix()]}
                strata.append(state)
    if not strata:
        raise ValueError('render contains no eligible nonempty source; no production plan was written')
    result = {'schema_version': 1, 'model_key': 'waste_bin_state', 'target_split': 'train',
        'budget': len(strata), 'purpose': 'Current real-background compositions from actual rendered bin evidence',
        'independence_basis': 'one source model and selected recorded background parents; not independent capture',
        'uncertainty_weight': 0, 'constraints': {'resolution': [640, 480], 'label_mode': 'candidate', 'ground_placement': 'polygon'},
        'strata': strata, 'planning': {'seed': seed, 'render_manifest': selection['render_manifest'],
                                      'excluded_render_samples': excluded, 'background_count': len(backgrounds)}}
    planned = plan_production(result, root=ROOT)
    compositor._planned_inputs(planned)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1918)
    args = parser.parse_args()
    _, source = load_task_config('waste_bin_state')
    output = (ROOT/source['production_plan']).resolve()
    expected = ROOT/'datasets/scripts/waste_bin_state/config/production_plan.json'
    if output != expected:
        raise ValueError('current bin plan must use its fixed task config path')
    result = build(args.seed)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'planned_images': result['budget'], 'plan': output.relative_to(ROOT).as_posix()}))


if __name__ == '__main__':
    main()
