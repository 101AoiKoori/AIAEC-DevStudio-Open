import sys
import unittest
import random
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import synthesize_real_ground as compositor
from lab.training_common.bayesian_allocation import plan_generation
from synthesize_real_ground import (alpha_place, background_choice, background_kind_for, exposure,
                                    car_foot_y, choose_v4_left, crop_v4, require_distinct_v4_people,
                                    has_requested_overlap, illumination_stratum, planar_warp,
                                    planned_scene_indices,
                                    project_corners, sample_person_anchors, scenario_occurrence,
                                    scenarios, stratified_lighting_level, truncated_gauss,
                                    validate_actor_placement, validate_person_production_plan,
                                    visible_masks, yolo_row)


class RealGroundSynthesisTest(unittest.TestCase):
    def test_planned_production_uses_allocated_state_and_records_sample(self):
        target = {'kind': 'single', 'people': 1, 'vehicles': 0, 'relation': 'none',
                  'first_person_class': 'community', 'first_vehicle_class': None}
        indices = planned_scene_indices('ground_only', target, 'normal')
        state = {'id': 'single', 'planned_count': 2, 'sources': ['group'],
                 'parameters': {'format': 'test'}}
        run = SimpleNamespace(plan={'strata': [state]}, source_refs=[], record_sample=mock.Mock())
        mask = np.ones((10, 10), dtype=bool)
        def fake_scene(index, rng, *_):
            image = np.full((10, 10, 3), rng.randint(0, 255), dtype=np.uint8)
            objects = [{'kind': 'person', 'category': 'community', 'source': 'figure.png',
                        'visible_bbox_xyxy': [0, 0, 10, 10]}]
            return image, [mask], [mask], objects, {'image': 'bg'}, {'level': 'normal'}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(compositor, 'ROOT', root), \
                 mock.patch.object(compositor, 'eligible_backgrounds', return_value=[]), \
                 mock.patch.object(compositor, 'load_sources', return_value=({}, {})), \
                 mock.patch.object(compositor, 'generate_one',
                                   side_effect=fake_scene) as generate:
                result = compositor._produce_planned_person(root/'batch', 42, run, {'single': indices})
            self.assertEqual(result['completed'], 2)
            self.assertEqual([call.args[0] for call in generate.call_args_list], list(indices[:2]))
            self.assertTrue((root/'batch/labels/person_000001.txt').read_text().startswith('0 '))
            first, second = run.record_sample.call_args_list
            self.assertEqual(first.args, ('single',))
            self.assertEqual(first.kwargs['actual']['cycle_index'], indices[0])
            self.assertEqual(first.kwargs['sources'], ['group'])
            self.assertNotEqual(first.kwargs['actual']['seed'], second.kwargs['actual']['seed'])

    def test_person_plan_refuses_unimplemented_constraints_and_source_groups(self):
        state = {'id': 's1', 'failure_event': 'miss', 'min_count': 1, 'max_count': 1,
                 'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
                 'conditions': [], 'background': {'kind': 'ground_only'},
                 'target': {'kind': 'occlusion', 'people': 1, 'vehicles': 1,
                            'relation': 'vehicle_front', 'first_person_class': 'noncommunity',
                            'first_vehicle_class': 'fallen'},
                 'labels': {'person': 'visible_yolo', 'vehicle': 'not_emitted'},
                 'augmentation': {'lighting': 'dim'},
                 'parameters': {'resolution': [960, 480], 'geometry': 'bounded_planar',
                                'sampling': 'bounded_gaussian'},
                 'sources': ['scene_video_49s_shared', 'person.team_assets',
                             'e_bike.multiview_masks_refined_20261110_v2']}
        request = {'schema_version': 1, 'model_key': 'person', 'target_split': 'train',
                   'purpose': 'contract test', 'budget': 1, 'uncertainty_weight': 0,
                   'independence_basis': 'contract test only',
                   'constraints': {'person_scope': 'near_frontal', 'person_box': 'visible',
                                   'white_back': 'not_synthesized', 'distinct_person_sources': True},
                   'strata': [state]}
        self.assertTrue(validate_person_production_plan(request)['s1'])
        self.assertEqual(plan_generation(request)['strata'][0]['planned_count'], 1)
        with self.assertRaises(ValueError):
            validate_person_production_plan({**request, 'strata': [{**state, 'conditions': ['rain']}]})
        with self.assertRaises(ValueError):
            validate_person_production_plan({**request, 'strata': [{**state, 'sources': ['other']}]})
        with self.assertRaises(ValueError):
            validate_person_production_plan({**request, 'strata': [{**state, 'observations': []}]})
        with self.assertRaises(ValueError):
            validate_person_production_plan({**request, 'constraints': {**request['constraints'], 'rain': True}})

    def test_explicit_scene_state_selects_only_matching_cycle_indices(self):
        target = {'kind': 'occlusion', 'people': 1, 'vehicles': 1,
                  'relation': 'vehicle_front', 'first_person_class': 'noncommunity',
                  'first_vehicle_class': 'fallen'}
        indices = planned_scene_indices('ground_only', target, 'dim')
        self.assertTrue(indices)
        for index in indices:
            self.assertEqual(scenarios(index), {key: target[key] for key in
                             ('kind', 'people', 'vehicles', 'relation')})
            self.assertEqual(background_kind_for(index, scenarios(index)), 'ground_only')
            self.assertEqual(stratified_lighting_level(index), 'dim')
            self.assertEqual(scenario_occurrence(index) % 2, 1)
            self.assertEqual((scenario_occurrence(index) // 2) % 2, 1)
        with self.assertRaises(ValueError):
            planned_scene_indices('sign_context', {**target, 'people': 3}, 'dim')
        with self.assertRaises(ValueError):
            planned_scene_indices('ground_only', target, 'unimplemented')

    def test_repeated_front_source_is_rejected(self):
        require_distinct_v4_people([{'kind':'person', 'source':'a'},
                                    {'kind':'person_back', 'source':{'foreground':'a'}}])
        with self.assertRaises(ValueError):
            require_distinct_v4_people([{'kind':'person', 'source':'a'}, {'kind':'person', 'source':'a'}])

    def test_car_background_crop_preserves_native_pixels_and_foot_bounds(self):
        source = np.random.default_rng(7).integers(0, 256, (480,640,3), dtype=np.uint8)
        padded = cv2.copyMakeBorder(source, 0, 0, 160, 160, cv2.BORDER_REPLICATE)
        image, _, _, _ = crop_v4(padded, [], [], [], 160)
        self.assertTrue(np.array_equal(image, source))
        rng = random.Random(11)
        self.assertTrue(all(480*.73 <= car_foot_y(rng) <= 480*.94 for _ in range(100)))

    def test_v4_crop_recomputes_visible_box_and_preserves_occlusion(self):
        canvas = np.zeros((480, 960, 3), np.uint8)
        person = np.zeros((480, 960), bool); person[100:300, 200:400] = True
        vehicle = np.zeros_like(person); vehicle[200:320, 300:450] = True
        shown = visible_masks([person, vehicle])
        image, full, visible, meta = crop_v4(canvas, [person, vehicle], shown,
                                             [{'kind':'person', 'foot_x':350}, {'kind':'vehicle'}], 250)
        self.assertEqual(image.shape, (480, 640, 3))
        self.assertEqual(meta[0]['visible_bbox_xyxy'], [0, 100, 150, 300])
        self.assertFalse(np.any(visible[0] & visible[1]))
        self.assertAlmostEqual(meta[0]['crop_retained_fraction'], .75)
        self.assertEqual((meta[0]['foot_x_canvas_960'], meta[0]['foot_x']), (350, 100))
        self.assertAlmostEqual(meta[0]['visible_fraction'], float(visible[0].sum()/full[0].sum()))

    def test_v4_crop_planner_and_white_back_do_not_change_class_contract(self):
        back = np.zeros((480,960), bool); back[100:260, 100:180] = True
        person = np.zeros_like(back); person[110:300, 550:630] = True
        left = choose_v4_left([back, person], random.Random(4))
        self.assertLessEqual(left, 100)
        self.assertGreaterEqual(left + 640, 630)
        far = np.zeros_like(back); far[10:20,900:920] = True
        with self.assertRaises(ValueError):
            choose_v4_left([back,far], random.Random(0))
        empty, _, _, result = crop_v4(np.zeros((480,960,3), np.uint8), [back], [back],
                                      [{'kind':'person_back','category':'white_back'}], 0)
        self.assertEqual(empty.shape[:2], (480,640))
        self.assertEqual(result[0]['kind'], 'person_back')

    def test_planar_angles_keep_orientation_and_are_bounded(self):
        for yaw in (-12, 0, 12):
            for pitch in (-6, 0, 6):
                for roll in (-3, 0, 3):
                    p = project_corners(100, 200, yaw, pitch, roll)
                    first, second = p[1]-p[0], p[2]-p[1]
                    self.assertGreater(first[0]*second[1]-first[1]*second[0], 0)
        with self.assertRaises(ValueError):
            project_corners(100, 200, 180, 0, 0)

    def test_planar_warp_does_not_mirror_asymmetric_image(self):
        image = np.zeros((80, 40, 4), np.uint8)
        image[:, :, 3] = 255
        image[:, :15, 2] = 255
        result = planar_warp(image, 8, 4, 2)
        ys, xs = np.where((result[:, :, 2] > 200) & (result[:, :, 3] > 127))
        self.assertTrue(len(xs) > 0)
        self.assertLess(xs.mean(), result.shape[1] / 2)

    def test_depth_order_keeps_only_visible_pixels(self):
        back = np.zeros((20, 30), bool); back[2:18, 4:20] = True
        front = np.zeros((20, 30), bool); front[8:15, 12:27] = True
        a, b = visible_masks([back, front])
        self.assertTrue(np.array_equal(a, back & ~front))
        self.assertTrue(np.array_equal(b, front))
        self.assertFalse(np.any(a & b))
        self.assertTrue(has_requested_overlap([back, front], ['person', 'vehicle'], 'vehicle_front'))
        self.assertFalse(has_requested_overlap([back, front], ['person', 'vehicle'], 'person_front'))
        self.assertFalse(has_requested_overlap([back, ~back & front], ['person', 'vehicle'], 'vehicle_front'))

    def test_alpha_place_and_yolo_box(self):
        canvas = np.zeros((30, 30, 3), np.uint8)
        rgba = np.ones((10, 12, 4), np.uint8)*255
        mask = alpha_place(canvas, rgba, 15, 20)
        ys, xs = np.nonzero(mask)
        self.assertEqual([9, 10, 21, 20], [xs.min(), ys.min(), xs.max()+1, ys.max()+1])
        self.assertEqual('0 0.500000 0.500000 0.400000 0.333333', yolo_row(0, [9, 10, 21, 20], 30, 30))
        with self.assertRaises(ValueError):
            alpha_place(canvas, rgba, 2, 5)

    def test_scene_coverage_and_exposure_boundaries(self):
        cases = [scenarios(i) for i in range(20)]
        self.assertEqual({'single': 5, 'multi': 6, 'occlusion': 6, 'no_person': 3},
                         dict(Counter(x['kind'] for x in cases)))
        self.assertEqual({'single', 'multi', 'occlusion', 'no_person'},
                         {x['kind'] for x in cases[:10]})
        self.assertEqual({'vehicle_front', 'person_front'},
                         {x['relation'] for x in cases if x['kind'] == 'occlusion'})
        source = np.full((4, 4, 3), 128, np.uint8)
        self.assertLess(exposure(source, .82, 1.06).mean(), 128)
        self.assertGreater(exposure(source, 1.15, .95).mean(), 128)
        with self.assertRaises(ValueError):
            exposure(source, .2, 1)

    def test_truncated_gauss_is_reproducible_bounded_and_not_clipped(self):
        first, second = random.Random(123), random.Random(123)
        values = [truncated_gauss(first, 0, 5, -12, 12) for _ in range(1000)]
        self.assertEqual(values, [truncated_gauss(second, 0, 5, -12, 12) for _ in range(1000)])
        self.assertTrue(all(-12 <= value <= 12 for value in values))
        self.assertFalse(any(value in (-12, 12) for value in values))
        self.assertLess(abs(sum(values)/len(values)), 1)
        with self.assertRaises(ValueError):
            truncated_gauss(random.Random(1), 5, 0, -1, 1)

    def test_each_type_gets_each_light_and_both_person_categories(self):
        combinations = {}
        for index in range(240):
            case = scenarios(index); occurrence = scenario_occurrence(index)
            if case['people']:
                combinations.setdefault(case['kind'], set()).add((occurrence % 2, stratified_lighting_level(index)))
        expected = {(cls, level) for cls in (0, 1) for level in ('dim', 'bright', 'normal')}
        for kind in ('single', 'multi', 'occlusion'):
            self.assertEqual(expected, combinations[kind])
        paired = {(scenario_occurrence(i) % 2, (scenario_occurrence(i)//2) % 2)
                  for i in range(80) if scenarios(i)['kind'] == 'occlusion'}
        self.assertEqual({(0, 0), (0, 1), (1, 0), (1, 1)}, paired)

    def test_3500_exact_quotas_and_all_feasible_strata_have_all_lights(self):
        counts = Counter(scenarios(i)['kind'] for i in range(3500))
        self.assertEqual({'single': 875, 'multi': 1050, 'occlusion': 1050, 'no_person': 525}, dict(counts))
        per_stratum = {}
        for i in range(3500):
            key = illumination_stratum(i)
            self.assertEqual(key, illumination_stratum(i % 240))
            per_stratum.setdefault(key, Counter())[stratified_lighting_level(i)] += 1
        for key, lights in per_stratum.items():
            self.assertEqual({'dim', 'bright', 'normal'}, set(lights), key)
            self.assertLessEqual(max(lights.values())-min(lights.values()), 1, key)
        # All logically possible class pairs and available backgrounds are present.
        for kind, backgrounds, person_values, vehicle_values in (
                ('single', ('ground_only', 'track_context', 'sign_context'), (0, 1), (None,)),
                ('multi', ('ground_only', 'track_context'), (0, 1), (None,)),
                ('occlusion', ('ground_only', 'track_context', 'sign_context'), (0, 1), (0, 1)),
                ('no_person', ('ground_only', 'track_context', 'sign_context'), (None,), (None, 0, 1))):
            for background in backgrounds:
                for person in person_values:
                    for vehicle in vehicle_values:
                        self.assertTrue(any(key[0] == kind and key[1] == background and key[3:] == (person, vehicle)
                                            for key in per_stratum), (kind, background, person, vehicle))
        self.assertEqual({'dim', 'bright', 'normal'}, {stratified_lighting_level(i) for i in range(10)})

    def test_background_supervision_and_placement_contract(self):
        backgrounds = [{'path': Path('ground.jpg'), 'frame': 1, 'seconds': .03}]
        ground = background_choice(0, scenarios(0), backgrounds, random.Random(1))
        track = background_choice(1, scenarios(1), backgrounds, random.Random(1))
        sign = background_choice(2, scenarios(2), backgrounds, random.Random(1))
        self.assertEqual(('ground_only', True), (ground[1], ground[3]))
        self.assertEqual(('track_context', False), (track[1], track[3]))
        self.assertEqual(('sign_context', False), (sign[1], sign[3]))
        anchors = sample_person_anchors(random.Random(7), 3, 'track_context')
        self.assertTrue(all(410 <= x <= 820 for x in anchors))
        self.assertTrue(all(abs(x-y) >= 90 for i, x in enumerate(anchors) for y in anchors[i+1:]))
        sprite = np.zeros((120, 80, 4), np.uint8)
        validate_actor_placement({'foot_x': 450, 'foot_y': 455, 'sprite': sprite}, 'sign_context')
        with self.assertRaises(ValueError):
            validate_actor_placement({'foot_x': 400, 'foot_y': 455, 'sprite': sprite}, 'sign_context')
        with self.assertRaises(ValueError):
            validate_actor_placement({'foot_x': 450, 'foot_y': 300, 'sprite': sprite}, 'sign_context')


if __name__ == '__main__':
    unittest.main()
