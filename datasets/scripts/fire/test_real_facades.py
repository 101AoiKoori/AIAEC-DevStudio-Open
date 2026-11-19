"""Regressions for current-database fire production and spherical camera coverage."""
import copy
import math
import unittest

import numpy as np
import cv2

from datasets.scripts.common.config import load_task_config
from datasets.scripts.fire import compose_real_facades as producer
from datasets.scripts.fire.camera_trajectory import orange_peel_pose
from datasets.scripts.fire.wall_projection import transform_points


class FireSampleContracts(unittest.TestCase):
    def test_current_database_builds_both_camera_branches(self):
        source = load_task_config('fire')[1]
        uses = producer.current_inputs(source)
        self.assertEqual({u['branch'] for u in uses.values()},
                         {'direct_real', 'real_background_composite'})
        for building in 'ABC':
            counts = {u['count'] for u in uses.values() if u['branch'] == 'real_background_composite'
                      and u['target_building'] == building}
            self.assertEqual(counts, {0, 1, 3})

    def test_existing_multi_stickers_survive_the_normal_output_transform(self):
        source = load_task_config('fire')[1]
        use = next(u for u in producer.current_inputs(source).values()
                   if u['branch'] == 'direct_real' and u['count'] > 1)
        image, instances, transform = producer.compose_camera(use, None, source['sample_recipe'], 431)
        self.assertEqual(len(instances), len(use['row']['label']['instances']))
        self.assertTrue(all(i['origin'] == 'existing' for i in instances))
        for item in instances:
            producer.yolo_row(item['xyxy'], image.shape[1], image.shape[0])
        self.assertEqual(transform['crop_xyxy'], [0, 0, *use['row']['image_size']])

    def test_multi_paste_labels_all_slots_and_leaves_whole_building_unknown(self):
        source = load_task_config('fire')[1]
        use = next(u for u in producer.current_inputs(source).values()
                   if u['branch'] == 'real_background_composite' and u['count'] == 3)
        image, instances, _ = producer.compose_camera(
            use, np.full((210, 150, 3), 255, np.uint8), source['sample_recipe'], 431)
        self.assertEqual(len({i['slot_id'] for i in instances}), 3)
        counts = producer.counts_for(use['visible_buildings'], instances, 'crop')
        self.assertEqual(counts[use['target_building']]['visible_count'], 3)
        self.assertIsNone(counts[use['target_building']]['whole_building_count'])
        lower, upper = source['sample_recipe']['sticker_size']['height_fraction'][use['target_building']]
        self.assertGreater(len({i['height_fraction'] for i in instances}), 1)
        for item in instances:
            quad = np.array(item['quad_wall_xy'])
            height = (np.linalg.norm(quad[3]-quad[0]) + np.linalg.norm(quad[2]-quad[1])) / 2
            self.assertAlmostEqual(height/item['reference_facade_height_wall_units'], item['height_fraction'], places=5)
            self.assertTrue(lower <= item['height_fraction'] <= upper)
            self.assertGreater(item['scale_from_reference'], 1)
            inverse = np.linalg.inv(np.array(item['wall_homography']))
            restored = transform_points(item['quad_source_px'], inverse)
            np.testing.assert_allclose(restored, quad, atol=.002)
            np.testing.assert_allclose(restored[0, 1], restored[1, 1], atol=.002)
            np.testing.assert_allclose(restored[1, 0], restored[2, 0], atol=.002)
            for point in item['quad_source_px']:
                self.assertGreaterEqual(cv2.pointPolygonTest(np.float32(item['wall_boundary_px']), tuple(point), True),
                                        source['sample_recipe']['wall_projection']['margin_source_px'])

    def test_corner_facades_keep_every_paper_on_its_assigned_plane(self):
        source = load_task_config('fire')[1]
        use = next(u for u in producer.current_inputs(source).values()
                   if u['branch'] == 'real_background_composite' and u['count'] == 3
                   and u['target_building'] == 'C' and len(u['row']['geometry']['wall_surfaces']) == 2)
        _, instances, _ = producer.compose_camera(
            use, np.full((210, 150, 3), 255, np.uint8), source['sample_recipe'], 431)
        self.assertEqual({item['wall_id'] for item in instances}, {'C_left', 'C_right'})
        for item in instances:
            for point in item['quad_source_px']:
                self.assertGreater(cv2.pointPolygonTest(np.float32(item['wall_boundary_px']), tuple(point), True), 0)

    def test_invalid_source_box_cannot_become_an_empty_label(self):
        source = load_task_config('fire')[1]
        use = copy.deepcopy(next(u for u in producer.current_inputs(source).values()
                                 if u['branch'] == 'direct_real'))
        use['row']['label']['instances'][0]['xyxy'] = [-1, 0, 10, 10]
        with self.assertRaises(ValueError):
            producer.compose_camera(use, None, source['sample_recipe'], 431)

    def test_camera_samples_cover_the_spherical_strip(self):
        config = load_task_config('fire')[1]['camera_trajectory']
        poses = [orange_peel_pose(config, index, .27) for index in range(9)]
        self.assertEqual({p['azimuth_deg'] for p in poses}, set(config['azimuth_deg']))
        self.assertEqual({round(p['position_m'][2], 6) for p in poses}, set(config['camera_height_m']))
        for pose in poses:
            radius = math.dist(pose['position_m'], pose['target_m'])
            self.assertAlmostEqual(radius, config['radius_m'])
        midway = orange_peel_pose(config, .5, .27)
        self.assertAlmostEqual(midway['azimuth_deg'], -15)
        for index in range(9):
            global_view = orange_peel_pose(config, index, 0, .54)
            self.assertGreaterEqual(global_view['position_m'][2], min(config['camera_height_m']) - 1e-6)


if __name__ == '__main__':
    unittest.main()
