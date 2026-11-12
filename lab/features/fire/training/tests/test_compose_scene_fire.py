import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compose_scene_fire import add_two_papers, building_mask, distribute, ground_line, transform
from compose_scene_fire_eval import SOURCES, flame_paper, name_paper, paste
from vary_name_papers import make_name, write_paper


class SceneFireTest(unittest.TestCase):
    def test_fixed_proportions(self):
        self.assertEqual(distribute(4000, [20, 30, 25, 25]), [800, 1200, 1000, 1000])
        self.assertEqual(sum(distribute(13, [20, 30, 25, 25])), 13)

    def test_ground_and_silhouette(self):
        image = np.full((640, 640, 3), 120, np.uint8)
        image[100:500, 200:400] = 180
        image[200:240, 230:270] = 20
        mask = building_mask(image)
        self.assertEqual(int(mask[210, 240]), 255)
        self.assertEqual(int(mask[10, 10]), 0)
        scene = np.zeros((640, 640, 3), np.uint8)
        scene[400:404, :] = 255
        self.assertEqual(ground_line(scene), 403)

    def test_transformed_boxes_and_floor_alignment(self):
        import random
        image = np.zeros((640, 640, 3), np.uint8)
        mask = np.zeros((640, 640), np.uint8)
        mask[100:500, 200:400] = 255
        instances = [{'state': 'visible', 'pixel_bbox_xyxy': [250, 110, 350, 140]}]
        _, result, items, affine = transform(image, mask, instances, random.Random(42), 410)
        self.assertEqual(len(items), 1)
        self.assertGreaterEqual(np.nonzero(result)[0].max(), 430)
        self.assertLessEqual(np.nonzero(result)[0].max(), 465)
        x1, y1, x2, y2 = items[0][1]
        self.assertTrue(0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 640)
        self.assertGreater(affine['scale'], 0)

    def test_two_papers_are_distinct_and_labelled(self):
        import random
        image = np.zeros((640, 640, 3), np.uint8)
        mask = np.zeros((640, 640), np.uint8)
        mask[100:500, 200:400] = 255
        icon = np.full((140, 100, 3), 220, np.uint8)
        result, objects = add_two_papers(image, mask, [], icon, random.Random(11))
        self.assertEqual(len(objects), 2)
        self.assertNotEqual(objects[0]['pixel_bbox_xyxy'], objects[1]['pixel_bbox_xyxy'])
        for obj in objects:
            x1, y1, x2, y2 = obj['pixel_bbox_xyxy']
            self.assertEqual(obj['kind'], 'fire_sticker')
            self.assertTrue((result[y1:y2, x1:x2] > 0).all())

    def test_eval_paper_warp_and_separate_source_groups(self):
        import random
        image = np.zeros((540, 960, 3), np.uint8)
        paper = flame_paper(random.Random(5))
        self.assertEqual(paper.shape, (140, 100, 3))
        box = paste(image, paper, [[50, 50], [90, 50], [90, 100], [50, 100]])
        self.assertEqual(box, [50, 50, 91, 101])
        self.assertGreater(int(image[70, 70].sum()), 0)
        self.assertEqual(name_paper('星海东南大厦').shape, (160, 600, 3))
        self.assertEqual(len({v['source_group'] for v in SOURCES.values()}), 2)

    def test_name_generation_and_paper_extent(self):
        self.assertEqual(len({make_name(i) for i in range(1000)}), 1000)
        image = np.full((640, 640, 3), 180, np.uint8)
        write_paper(image, [150, 150, 300, 190], make_name(123))
        self.assertTrue(np.all(image[:140] == 180))
        self.assertLess(int(image[150:190, 150:300].min()), 180)


if __name__ == '__main__':
    unittest.main()
