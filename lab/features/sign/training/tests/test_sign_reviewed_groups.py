import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/features/sign/training/scripts'))
sys.path.insert(0, str(ROOT / 'lab/training_common'))
import bayesian_allocation as allocation
import compose_sign_v4_measured as measured
import plan_sign_reviewed_groups as planner


class ReviewedSignGroupsTests(unittest.TestCase):
    def test_retired_plan_ids_are_rejected_before_guard(self):
        for name in ('photo-full_0001', 'photo-clip_0010'):
            with self.assertRaisesRegex(ValueError, 'Retired'):
                measured.generate(ROOT / 'datasets/exports/sign/unused', 1918, False,
                                  {'strata': [{'id': name}]})

    def test_shadow_changes_only_background_and_is_bounded(self):
        background = np.full((480, 640, 3), 180, np.uint8)
        sprite = np.full((100, 40, 4), 255, np.uint8)
        p = {'height': 100, 'left': 100, 'top': 250,
             'ground_polygon': [[90, 330], [160, 330], [160, 370], [90, 370]],
             'occlusion_polygons': [], 'visible_kind': 'fixture', 'visibility_evidence': 'fixture',
             'shadow_strength': .25}
        shadowed, box, actual = measured._compose_placed(background, sprite, 'normal', p)
        plain, plain_box, _ = measured._compose_placed(background, sprite, 'normal', dict(p, shadow_strength=0))
        self.assertEqual(box, plain_box)
        self.assertEqual(actual['shadow_strength'], .25)
        self.assertTrue(np.any(shadowed != plain))
        for invalid in (-.1, .36, float('nan')):
            with self.assertRaises(ValueError):
                measured._placed_parameters(dict(p, shadow_strength=invalid))

    def test_group_prior_changes_allocations_and_supports_full_footprint(self):
        plan = planner.build(24, 1918)
        result = plan['group_allocation']
        keys = ('schema_version', 'model_key', 'target_split', 'budget', 'purpose',
                'independence_basis', 'uncertainty_weight')
        group_keys = ('id', 'failure_event', 'conditions', 'min_count', 'max_count',
                      'prior', 'exposure_weight', 'failure_cost')
        request = {k: result[k] for k in keys}
        request['strata'] = [{k: s[k] for k in group_keys} for s in result['strata']]
        changed = copy.deepcopy(request)
        changed['strata'][0]['prior'] = {'alpha': 50, 'beta': 1}
        changed['strata'][0]['failure_cost'] = 10
        counts = lambda r: [x['planned_count'] for x in allocation.allocate(r)['strata']]
        self.assertNotEqual(counts(request), counts(changed))
        for state in plan['strata']:
            p = state['parameters']
            sprite = measured.base.cutout(state['target']['mask'])
            width = round(sprite.shape[1] * p['height'] / sprite.shape[0])
            x1, y1 = p['ground_polygon'][0]
            x2, y2 = p['ground_polygon'][2]
            self.assertTrue(x1 <= p['left'] and p['left'] + width <= x2)
            self.assertTrue(y1 <= p['top'] + p['height'] <= y2)

    def test_expanded_pool_avoids_protected_background_objects(self):
        plan = planner.build(120, 1918)
        frames = {item['image']: item for item in plan['background_inputs']}
        self.assertEqual(len(frames), 10)
        self.assertEqual(len({item['parent_video'] for item in frames.values() if item.get('parent_video')}), 3)
        self.assertEqual(len([item for item in frames.values() if item.get('source_group') == planner.PHOTO_PARENT]), 2)
        for frame in frames.values():
            if frame.get('source_group') == planner.PHOTO_PARENT:
                self.assertNotIn('parent_video', frame)
                self.assertEqual(frame['prior_sign_observation']['state'], 'unknown')
                self.assertEqual(frame['task_use_status'][0]['status'], 'candidate')
        used = set()
        for state in plan['strata']:
            name = state['background']['image']
            used.add(name)
            p = state['parameters']
            sprite = measured.base.cutout(state['target']['mask'])
            width = round(sprite.shape[1] * p['height'] / sprite.shape[0])
            for x1, y1, x2, y2 in frames[name]['protected_rectangles_xyxy']:
                self.assertFalse(p['left'] < x2 and p['left'] + width > x1
                                 and p['top'] < y2 and p['top'] + p['height'] > y1)
        self.assertEqual(used, set(frames))


if __name__ == '__main__':
    unittest.main()
