import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from person_scene_plan import compose_scene


def actor(name, x, *, view='front', category='community', kind='person', y=200):
    item = {'id': name, 'kind': kind, 'source': f'fixture/{name}.png',
            'foot_xy': [x, y], 'contact_width': 12, 'size_wh': [30, 80],
            'angles_degrees': {'yaw': 0, 'pitch': 0, 'roll': 0},
            'category': category, 'view': view, 'identity': name}
    if kind == 'person' and view == 'front':
        item.update(identity_evidence=f'fixture/{name}_identity', min_identity_visible=0.8)
    return item


class PersonScenePlanTest(unittest.TestCase):
    def setUp(self):
        self.background = np.zeros((240, 320, 3), dtype=np.uint8)
        self.sprite = np.full((80, 30, 4), 255, dtype=np.uint8)
        self.marker = np.zeros((80, 30), dtype=np.uint8)
        self.marker[5:20, 5:25] = 255
        self.plan = {'background_source': 'fixture/background.png',
                     'background_assessment': {'person_scope': 'confirmed_empty', 'evidence': 'fixture/review'},
                     'ground_polygon': [[0, 150], [319, 150], [319, 239], [0, 239]],
                     'obstacle_polygons': [], 'actors': [], 'required_overlaps': []}

    def compose(self, plan):
        foregrounds = {item['id']: self.sprite.copy() for item in plan['actors']}
        markers = {item['id']: self.marker.copy() for item in plan['actors']
                   if item['kind'] == 'person' and item['view'] == 'front'}
        return compose_scene(self.background, plan, foregrounds, markers)

    def test_explicit_front_vehicle_overlap_and_nonfrontal_negative(self):
        person = actor('front', 90, category='noncommunity')
        back = actor('back', 190, view='back', category=None)
        car = actor('car', 95, kind='vehicle', category='fallen', y=220)
        car['size_wh'] = [45, 30]
        self.plan['actors'] = [person, back, car]
        self.plan['required_overlaps'] = [['front', 'car']]
        result = self.compose(self.plan)
        self.assertEqual(len(result['labels']), 1)
        self.assertTrue(result['labels'][0].startswith('1 '))
        self.assertLess(result['objects'][0]['visible_fraction'], 1)
        self.assertEqual(result['objects'][0]['identity_visible_fraction'], 1)
        self.assertTrue(np.any(result['full_masks'][0] & result['full_masks'][2]))
        self.assertFalse(np.any(result['visible_masks'][0] & result['visible_masks'][2]))
        self.assertEqual(result['objects'][1]['view'], 'back')
        self.assertFalse(self.background.any())

    def test_eight_explicit_people_keep_independent_identity(self):
        self.plan['actors'] = [actor(f'p{i}', 25 + 38*i) for i in range(8)]
        result = self.compose(self.plan)
        self.assertEqual(len(result['labels']), 8)
        self.assertEqual(len({r['identity'] for r in result['objects']}), 8)

    def test_rejects_unreviewed_background_and_missing_identity_evidence(self):
        self.plan['actors'] = [actor('p', 80)]
        bad = copy.deepcopy(self.plan)
        bad['background_assessment']['person_scope'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'background requires evidence'):
            self.compose(bad)
        with self.assertRaisesRegex(ValueError, 'missing identity evidence mask'):
            compose_scene(self.background, self.plan, {'p': self.sprite}, {})

    def test_rejects_unidentifiable_occlusion_and_wrong_order(self):
        self.plan['actors'] = [actor('p', 90), actor('car', 90, kind='vehicle', category='upright')]
        self.plan['required_overlaps'] = [['p', 'car']]
        with self.assertRaisesRegex(ValueError, 'identity evidence is obscured'):
            self.compose(self.plan)
        self.plan['required_overlaps'] = [['car', 'p']]
        with self.assertRaisesRegex(ValueError, 'back-to-front order'):
            self.compose(self.plan)

    def test_ground_and_obstacle_intersect_full_contact_segment(self):
        self.plan['actors'] = [actor('p', 90)]
        self.plan['obstacle_polygons'] = [[[89, 190], [91, 190], [91, 210], [89, 210]]]
        with self.assertRaisesRegex(ValueError, 'intersects obstacle'):
            self.compose(self.plan)
        self.plan['obstacle_polygons'] = []
        self.plan['actors'][0]['foot_xy'] = [3, 200]
        with self.assertRaisesRegex(ValueError, 'outside ground'):
            self.compose(self.plan)

    def test_transform_preserves_identity_evidence_and_records_geometry(self):
        item = actor('p', 90)
        item['size_wh'] = [40, 100]
        item['angles_degrees'] = {'yaw': 10, 'pitch': 4, 'roll': 2}
        self.plan['actors'] = [item]
        result = self.compose(self.plan)
        self.assertEqual(len(result['labels']), 1)
        self.assertEqual(result['objects'][0]['angles_degrees'], item['angles_degrees'])
        self.assertEqual(result['objects'][0]['identity_visible_fraction'], 1)

    def test_thin_obstacle_crossing_does_not_depend_on_pixel_sampling(self):
        self.plan['actors'] = [actor('p', 90)]
        self.plan['obstacle_polygons'] = [[[90.2, 190], [90.3, 190], [90.3, 210], [90.2, 210]]]
        with self.assertRaisesRegex(ValueError, 'intersects obstacle'):
            self.compose(self.plan)

    def test_invalid_side_category_and_no_overlap_are_rejected(self):
        self.plan['actors'] = [actor('side', 60, view='side')]
        with self.assertRaisesRegex(ValueError, 'must not have a target category'):
            self.compose(self.plan)
        self.plan['actors'] = [actor('p', 60), actor('car', 200, kind='vehicle', category='fallen')]
        self.plan['required_overlaps'] = [['p', 'car']]
        with self.assertRaisesRegex(ValueError, 'no intersecting pixels'):
            self.compose(self.plan)

    def test_empty_scene_and_duplicate_identity(self):
        self.assertEqual(self.compose(self.plan)['labels'], [])
        self.plan['actors'] = [actor('p1', 60), actor('p2', 120)]
        self.plan['actors'][1]['identity'] = 'p1'
        with self.assertRaisesRegex(ValueError, 'duplicate frontal person identity'):
            self.compose(self.plan)

    def test_existing_person_labels_are_preserved_and_protected(self):
        self.plan['background_assessment'] = {
            'person_scope': 'complete_existing', 'evidence': 'fixture/review',
            'existing_people': [{'class_id': 1, 'xyxy': [5, 80, 40, 180]}],
            'avoid_boxes': [[120, 100, 150, 180]]}
        self.plan['actors'] = [actor('p', 90)]
        result = self.compose(self.plan)
        self.assertEqual(len(result['labels']), 2)
        self.assertTrue(result['labels'][0].startswith('1 '))
        self.assertTrue(result['labels'][1].startswith('0 '))
        self.plan['actors'][0]['foot_xy'] = [25, 170]
        with self.assertRaisesRegex(ValueError, 'protected background object'):
            self.compose(self.plan)
        self.plan['actors'][0]['foot_xy'] = [130, 200]
        with self.assertRaisesRegex(ValueError, 'protected background object'):
            self.compose(self.plan)

    def test_existing_background_supervision_rejects_missing_or_invalid_boxes(self):
        self.plan['background_assessment'] = {
            'person_scope': 'complete_existing', 'evidence': 'fixture/review', 'existing_people': []}
        with self.assertRaisesRegex(ValueError, 'requires existing people'):
            self.compose(self.plan)
        self.plan['background_assessment']['existing_people'] = [{'class_id': 2, 'xyxy': [0, 1, 20, 40]}]
        with self.assertRaisesRegex(ValueError, 'unsupported existing person class'):
            self.compose(self.plan)


if __name__ == '__main__':
    unittest.main()
