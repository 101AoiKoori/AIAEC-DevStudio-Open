"""Geometry and allocation contracts for the bounded rooftop augmentation."""
import copy
import random
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/features/building_name_paper/training/scripts'))
from compose_rooftop_papers import augment_scene


class AugmentationContracts(unittest.TestCase):
    def test_homography_updates_labels_without_clipping(self):
        image = np.zeros((150, 240, 3), dtype=np.uint8)
        quad = [[70, 50], [160, 50], [160, 90], [70, 90]]
        for seed in range(30):
            papers = [{'quad_xy': copy.deepcopy(quad), 'box_xyxy': [70, 50, 160, 90]}]
            result, actual = augment_scene(image, papers, random.Random(seed))
            expected = cv2.perspectiveTransform(np.float32([quad]), np.asarray(actual['homography']))[0]
            self.assertTrue(np.allclose(expected, papers[0]['quad_xy']))
            x1, y1, x2, y2 = papers[0]['box_xyxy']
            self.assertTrue(0 <= x1 <= expected[:, 0].min() <= expected[:, 0].max() <= x2 <= 240)
            self.assertTrue(0 <= y1 <= expected[:, 1].min() <= expected[:, 1].max() <= y2 <= 150)
            self.assertEqual(result.shape, image.shape)
            self.assertTrue(1.02 <= actual['scale'] <= 1.12)

    def test_prior_changes_elastic_allocation(self):
        states = [dict(id=f'state_{i}', failure_event='unknown', min_count=10, max_count=90,
                       prior={'alpha': 1, 'beta': 1}, exposure_weight=1, failure_cost=1)
                  for i in range(2)]
        plan = dict(schema_version=1, model_key='building_name_paper', target_split='train',
                    purpose='prior sensitivity', budget=100, uncertainty_weight=0,
                    independence_basis='fixture', strata=states)
        # plan_production is for generation states; use the same allocator for generic states.
        from lab.training_common.bayesian_allocation import allocate
        before = allocate(plan, ROOT)
        plan['strata'][0]['prior'] = {'alpha': 9, 'beta': 1}
        after = allocate(plan, ROOT)
        self.assertNotEqual([x['planned_count'] for x in before['strata']],
                            [x['planned_count'] for x in after['strata']])


if __name__ == '__main__':
    unittest.main()
