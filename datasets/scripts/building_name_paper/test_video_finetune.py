"""Check complete-paper crops, panel labels and the new-video-only split."""
from copy import deepcopy
import unittest

import numpy as np

from datasets.scripts.building_name_paper import build_video_finetune as build


class VideoFinetuneTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((720, 1280, 3), np.uint8)
        self.objects = [{'building_id': 'A', 'xyxy': [90, 40, 260, 110],
                         'quad': [[90, 40], [260, 40], [260, 110], [90, 110]]}]

    def test_positive_crop_preserves_all_paper_and_transforms_quad(self):
        original = deepcopy(self.objects)
        for variant in (0, 1):
            image, objects, roi = build.crop_box(self.image, self.objects, variant)
            self.assertEqual(image.shape, self.image.shape)
            self.assertTrue(0 <= objects[0]['xyxy'][0] < objects[0]['xyxy'][2] <= 1280)
            self.assertTrue(0 <= objects[0]['xyxy'][1] < objects[0]['xyxy'][3] <= 720)
            self.assertAlmostEqual(objects[0]['quad'][0][0], (90-roi[0])*1280/(roi[2]-roi[0]))
        self.assertEqual(original, self.objects)

    def test_negative_crop_rejects_any_paper_overlap(self):
        _, objects, _ = build.crop_box(self.image, self.objects, 0, negative=True)
        self.assertEqual(objects, [])
        overlapping = [{'xyxy': [610, 580, 700, 650]}]
        with self.assertRaisesRegex(ValueError, 'intersects'):
            build.crop_box(self.image, overlapping, 0, negative=True)

    def test_two_and_three_paper_panels_have_distinct_complete_labels(self):
        bases = {key: {'id': key, 'objects': deepcopy(self.objects)} for key in 'ABC'}
        for count in (2, 3):
            state = {'id': 'panel'+str(count), 'parameters': {
                'branch': 'multiview_montage', 'parents': list('ABC')[:count]}}
            _, objects, _ = build.render_state(state, bases, dict.fromkeys(bases, self.image))
            self.assertEqual(len(objects), count)
            self.assertTrue(all(objects[i]['xyxy'][2] <= objects[i+1]['xyxy'][0] for i in range(count-1)))

    def test_plan_has_only_new_videos_and_no_cross_file_split_or_parent(self):
        plan = build.read(build.PLAN)
        self.assertEqual(plan['original_training_images'], 0)
        bases = {r['id']: r for r in plan['base_frames']}
        allowed = {'buildingname.mp4', 'buildingname1.mp4', 'buildingname2.mp4'}
        files = {}
        for row in bases.values():
            filename = build.Path(row['video']).name
            self.assertIn(filename, allowed)
            self.assertEqual(files.setdefault(filename, row['split']), row['split'])
        self.assertEqual(files['buildingname.mp4'], 'val')
        self.assertEqual(files['buildingname1.mp4'], 'train')
        self.assertEqual(files['buildingname2.mp4'], 'train')
        for state in plan['strata']:
            splits = {bases[parent]['split'] for parent in state['parameters']['parents']}
            self.assertEqual(len(splits), 1)
            if state['parameters']['branch'] != 'video_frame':
                self.assertEqual(splits, {'train'})

    def test_visible_box_clips_paper_at_camera_boundary(self):
        box = build.visible_box([[30, -10], [100, -10], [100, 20], [30, 20]])
        self.assertEqual(box[1], 0)
        self.assertGreater(box[3], 0)


if __name__ == '__main__':
    unittest.main()
