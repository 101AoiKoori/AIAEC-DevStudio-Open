"""Check observed-view grouping, independent state recipes and visible-mask geometry."""
from collections import Counter
from unittest.mock import patch
import unittest

import numpy as np

from datasets.scripts.waste_bin_state import reshoot_groups as groups


def plan():
    return {'seed': 42, 'background_domain_cycle': ['camera', 'camera', 'camera', 'blender'],
            'train': {'real_cutout_composites': 576, 'composition_groups': [
                {'count': 1, 'images': 256, 'heights_px': [90, 110, 150, 200]},
                {'count': 2, 'images': 192, 'heights_px': [90, 110, 140, 170]},
                {'count': 4, 'images': 128, 'heights_px': [90, 110, 125, 140]}]},
            'augmentation': {'value_gain': [.8, .95, 1.05, 1.15], 'saturation_gain': [.8, 1.]}}


def photos():
    return [{'source_record_id': f'{c}_{s}_{v}', 'color': c, 'state': s,
             'image': f'{c}_{s}_{v}.jpg', 'mask': 'mask.png', 'cutout': f'{c}_{s}_{v}.png',
             'view_observation': {'compatibility_group': v}, 'composite_eligible': True}
            for c in groups.COLORS for s in groups.STATES
            for v in ('medium_front', 'medium_rear', 'medium_side')]


def rectangle(cutout, height):
    rgba = np.full((height, round(height*.94), 4), 140, dtype=np.uint8)
    rgba[:, :, 3] = 255
    return rgba


class RealGroupTests(unittest.TestCase):
    def test_every_color_and_state_has_identical_size_view_layout_recipes(self):
        jobs = groups.schedule(photos(), plan())
        self.assertEqual(Counter(j['count'] for j in jobs), {1: 256, 2: 192, 4: 128})
        self.assertEqual(Counter((s['color'], s['state']) for j in jobs for s in j['sources']),
                         Counter({(c, s): 144 for c in groups.COLORS for s in groups.STATES}))
        recipes = {}
        for j in jobs:
            if j['count'] == 1:
                continue
            self.assertTrue(all(s['view_observation']['compatibility_group'] == j['view_group']
                                for s in j['sources']))
            for s in j['sources']:
                recipes.setdefault((j['count'], s['color'], s['state']), Counter())[
                    j['height'], j['view_group'], j['layout']] += 1
        for count in (2, 4):
            expected = recipes[count, 'blue', 'closed']
            self.assertTrue(all(recipes[count, c, s] == expected for c in groups.COLORS for s in groups.STATES))
        single = [s['source_record_id'] for j in jobs if j['count'] == 1 for s in j['sources']]
        self.assertTrue(any('medium_side' in s for s in single))

    def test_truncated_sources_and_missing_compatible_views(self):
        rows = photos()
        rows.append({**rows[0], 'source_record_id': 'truncated', 'composite_eligible': False})
        self.assertTrue(all(s['source_record_id'] != 'truncated' for j in groups.schedule(rows, plan())
                            for s in j['sources']))
        bad = [r for r in rows if not (r['color'] == 'red' and r['state'] == 'closed'
                                      and r['view_observation']['compatibility_group'] == 'medium_rear')]
        with self.assertRaisesRegex(ValueError, 'No compatible observed view'):
            groups.schedule(bad, plan())

    def test_state_changes_keep_color_slots_and_recipe_controls(self):
        jobs = [j for j in groups.schedule(photos(), plan()) if j['count'] == 4 and j['recipe_index'] == 2]
        with patch.object(groups, '_patch', side_effect=rectangle):
            positions = [groups._group(j, j['layout'])[3] for j in jobs]
        self.assertTrue(all(p == positions[0] for p in positions))
        self.assertEqual(len(jobs), 16)

    def test_single_views_are_paired_across_states_before_source_reuse(self):
        rows = photos()
        template = rows[0]
        rows += [{**template, 'source_record_id': 'unmatched_low_closed', 'color': 'gray',
                  'state': 'closed', 'view_observation': {'compatibility_group': 'low_front'}},
                 {**template, 'source_record_id': 'unmatched_side_open', 'color': 'red',
                  'state': 'open', 'view_observation': {'compatibility_group': 'side_open_only'}},
                 # Extra photos in one state may alter source reuse, never view quotas.
                 {**template, 'source_record_id': 'extra_front_closed', 'color': 'green',
                  'state': 'closed', 'view_observation': {'compatibility_group': 'medium_front'}}]
        jobs = [j for j in groups.schedule(rows, plan()) if j['count'] == 1]
        ids = {s['source_record_id'] for j in jobs for s in j['sources']}
        self.assertNotIn('unmatched_low_closed', ids)
        self.assertNotIn('unmatched_side_open', ids)
        self.assertIn('extra_front_closed', ids)
        for color in groups.COLORS:
            by_state = [Counter((j['view_group'], j['height'], j['layout']) for j in jobs
                                if j['sources'][0]['color'] == color and j['sources'][0]['state'] == state)
                        for state in groups.STATES]
            self.assertEqual(by_state[0], by_state[1])

    def test_body_occlusion_keeps_lid_region_and_reports_visible_boxes(self):
        job = next(j for j in groups.schedule(photos(), plan()) if j['count'] == 4 and j['recipe_index'] == 2)
        floor = np.ones((480, 640), dtype=bool)
        bg = {'image': 'camera.jpg', 'source_kind': 'direct_empty_negative',
              'ground_polygon_xy': [[0, 0], [639, 0], [639, 479], [0, 479]],
              'parent_group': 'camera', 'evidence': 'reviewed', 'existing_bins': []}
        pixels = np.full((480, 640, 3), 100, dtype=np.uint8)
        with patch.object(groups, '_patch', side_effect=rectangle):
            image, bins, trace = groups.render(job, [(bg, pixels, floor)], plan())
        self.assertEqual(image.shape, (480, 640, 3))
        self.assertEqual(len(bins), 4)
        self.assertEqual(trace['actual_layout'], 'light_overlap')
        self.assertTrue(any(b['visible_fraction'] < 1 for b in bins))
        self.assertTrue(all(.8 < b['visible_fraction'] <= 1 for b in bins))
        self.assertTrue(all(s['actual_height_px'] == job['height'] for s in trace['foreground_sources']))
        self.assertTrue(all(b['visible_fraction'] == s['visible_fraction'] and
                            b['occlusion_fraction'] == s['occlusion_fraction'] and
                            b['actual_height_px'] == s['actual_height_px']
                            for b, s in zip(bins, trace['foreground_sources'])))
        for b in bins:
            x1, y1, x2, y2 = b['bbox_xyxy']
            self.assertTrue(0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480)
        # Tone affects the completed scene, including pixels with no pasted bin.
        self.assertTrue(np.all((image[0, 0] >= 104) & (image[0, 0] <= 105)))


if __name__ == '__main__':
    unittest.main()
