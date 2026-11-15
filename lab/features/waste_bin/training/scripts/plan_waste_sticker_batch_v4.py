"""Plan or, when separately authorized, produce full-frame sticker composites.

Planning reads task-use relations and writes only JSON. Production uses the
workspace guard; it is never started by the planning mode.
"""
import argparse
from collections import Counter
import json
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard, plan_production

ART = 'datasets/annotated/waste_bin/waste_sticker_artwork_source_manifest_20261114.jsonl'
BACKGROUNDS = 'datasets/annotated/waste_bin/waste_sticker_video_background_roles_20261114.jsonl'
MAIN = 'lab/features/waste_bin/training/config/waste_sticker_source_plan_20261114.json'
CONSTRAINTS = {'placement': 'four_nonoverlapping_full_frame_quadrants', 'image_size': [1280, 720],
               'labels': 'yolo_visible_paper_boxes', 'resize': 'stretch_to_output'}
FULL_FRAME_PRODUCTION_ENABLED = False


def relations(path, role):
    result = []
    for index, line in enumerate((ROOT / path).read_text(encoding='utf8').splitlines()):
        row = json.loads(line)
        selected = [use for use in row.get('task_use_status', [])
                    if use.get('task') == 'waste_sticker' and use.get('split') == 'train'
                    and use.get('role') == role and use.get('status') == 'ready']
        if selected:
            if len(selected) != 1:
                raise ValueError('ambiguous task-use relation')
            image = (ROOT / row['image']).resolve()
            if not image.is_relative_to(ROOT) or not image.is_file():
                raise ValueError('input is missing or outside workspace: ' + row['image'])
            with Image.open(image) as img:
                img.verify()
            result.append((row, selected[0], index))
    return result


def declaration(seed):
    main = json.loads((ROOT / MAIN).read_text(encoding='utf8'))['v4_full_combination_plan']
    arts = relations(ART, 'target_foreground')
    backgrounds = relations(BACKGROUNDS, 'background')
    ids = {use['class_id']: row['image'] for row, use, _ in arts}
    if set(ids) != set(range(24)):
        raise ValueError('the fixed foreground pool must cover classes 0..23')
    pools = {}
    for row, use, index in backgrounds:
        if use.get('background_item_status') != 'ai_visual_no_visible_waste_item_sticker':
            raise ValueError('background has no recorded full-frame item-sticker observation')
        pools.setdefault(use['parent_id'], []).append((row['image'], index))
    expected = {'001': (767, 0), '002': (255, 511), '003': (767, 0),
                '004': (255, 511), '005': (0, 512), '006': (0, 512)}
    if set(pools) != set(expected):
        raise ValueError('background pool does not match planned parent groups')
    rng = random.Random(seed)
    counts = Counter()
    for parent, (first, second) in expected.items():
        for sub, n in (('multi_bin_compatible_visibility', first),
                       ('oblique_compatible_visibility' if parent in ('001', '002', '003', '004')
                        else 'gray_single_bin_close_view', second)):
            for _ in range(n):
                image, index = rng.choice(pools[parent])
                counts[(parent, sub, image, index)] += 1
    states = []
    for n, ((parent, subgroup, image, index), count) in enumerate(sorted(counts.items())):
        params = {'image': image, 'class_pool': list(range(24)), 'parent_id': parent,
                  'subgroup': subgroup, 'background_record_index': index}
        states.append({'id': f'frame_{n:03d}', 'min_count': count, 'max_count': count,
          'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
          'failure_event': 'declared_composite_not_produced',
          'background': {'kind': 'video_frame', 'image': image, 'role_manifest': BACKGROUNDS, 'record_index': index},
          'target': {'kind': 'four_fixed_artworks', 'class_ids': list(range(24))},
          'labels': {'format': 'yolo_detection', 'scope': 'inserted_paper_boxes'},
          'augmentation': {'placement': CONSTRAINTS['placement'], 'selection': 'seeded_with_replacement'},
          'parameters': params, 'sources': [image] + [ids[i] for i in range(24)]})
    result = {'schema_version': 1, 'model_key': 'waste_sticker', 'target_split': 'train',
              'purpose': 'Empty-bin video backgrounds with fixed artwork; full-frame bounded placement',
              'budget': sum(counts.values()), 'uncertainty_weight': 0,
              'independence_basis': 'six source video files; capture-session independence unknown',
              'constraints': CONSTRAINTS, 'strata': states,
              'planning': {'seed': seed, 'selection': 'per parent draw existing frames with replacement',
                 'parent_quotas': {k: sum(v) for k, v in expected.items()},
                 'background_relations': BACKGROUNDS, 'foreground_relations': ART,
                 'source_plan': MAIN, 'target_images': main['target_images'],
                 'subgroup_scope': 'oblique group uses the named high/low-view video parents 002 and 004; no camera angle is measured',
                 'image_generation_performed': False}}
    result['planning']['branch_capacity'] = [
        {'branch': 'direct_camera', 'target': 42, 'arranged': 1, 'gap': 41,
         'entry_plan': 'datasets/annotated/waste_bin/waste_sticker_002016_candidate_package_spec_20261114.json',
         'constraint': 'One complete AI-label candidate; 41 additional source-frame labels not established. Historical exposure is unknown.'},
        {'branch': 'multi_bin_compatible_visibility', 'target': 2044, 'arranged': 2044, 'gap': 0},
        {'branch': 'oblique_compatible_visibility', 'target': 1022, 'arranged': 1022, 'gap': 0,
         'constraint': 'Uses parents named high/low view, without a calibrated angle.'},
        {'branch': 'gray_single_bin_close_view', 'target': 1024, 'arranged': 1024, 'gap': 0},
        {'branch': 'blender_front_view', 'target': 1534, 'arranged': 0, 'gap': 1534},
        {'branch': 'blender_oblique_view', 'target': 768, 'arranged': 0, 'gap': 768},
        {'branch': 'blender_partially_occluded_but_identifiable', 'target': 766, 'arranged': 0, 'gap': 766,
         'constraint': 'Existing scene renderer does not consume the animated sticker asset with sticker-specific labels.'}]
    result['planning']['blender_gap'] = 'Connect the animated sticker asset and sticker label geometry to a guarded Blender scene entry; scene assets are not rendered images.'
    validate(result, ids)
    return result


def validate(plan, ids):
    if plan['model_key'] != 'waste_sticker' or plan['target_split'] != 'train' or plan['constraints'] != CONSTRAINTS:
        raise ValueError('unsupported plan')
    eligible = {(row['image'], use['parent_id'], index) for row, use, index in relations(BACKGROUNDS, 'background')}
    if any(use.get('background_item_status') != 'ai_visual_no_visible_waste_item_sticker'
           for _, use, _ in relations(BACKGROUNDS, 'background')):
        raise ValueError('background item-sticker observation is missing')
    for state in plan['strata']:
        p = state['parameters']
        if (p['image'], p['parent_id'], p['background_record_index']) not in eligible:
            raise ValueError('background role relation changed')
        if p['class_pool'] != list(range(24)) or state['sources'] != [p['image']] + [ids[i] for i in range(24)]:
            raise ValueError('foreground role relation changed')
        if state['min_count'] != state['max_count'] or state['min_count'] < 1:
            raise ValueError('invalid fixed frame quota')
    if sum(s['min_count'] for s in plan['strata']) != plan['budget']:
        raise ValueError('budget mismatch')


def boxes(rng):
    """Four separate quadrants ensure no inserted-object occlusion or crop."""
    result = []
    for left, top in ((0, 0), (640, 0), (0, 360), (640, 360)):
        side = rng.randint(36, 150)
        x = rng.randint(left, left + 640 - side)
        y = rng.randint(top, top + 360 - side)
        result.append((x, y, x + side, y + side))
    return result


def _retired_produce(plan_path, out, seed):
    raise RuntimeError("Retired production entry; use the task scripts in datasets/scripts with its source.yaml")


def main():
    raise RuntimeError("Retired production entry; use the task scripts in datasets/scripts with its source.yaml")


if __name__ == '__main__':
    main()
