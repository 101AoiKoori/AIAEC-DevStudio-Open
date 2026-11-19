"""Visible-box and photometric checks for natural near-frame truncation."""
import unittest

import numpy as np

from datasets.scripts.sign import compose_sign_v4_measured as compositor


class SampleGeometryTest(unittest.TestCase):
    def test_occlusion_window_diagnostic_distinguishes_closed_holes_from_edge_cover(self):
        from datasets.scripts.sign.audit_current_sample import enclosed_pixels
        fixture=np.ones((12,12),dtype=bool)
        window=fixture.copy();window[4:8,4:8]=False
        edge_cover=fixture.copy();edge_cover[:6,:6]=False
        self.assertEqual(enclosed_pixels(fixture),0)
        self.assertEqual(enclosed_pixels(window),16)
        self.assertEqual(enclosed_pixels(edge_cover),0)

    def test_portrait_direct_frame_preserves_aspect_and_label_mapping(self):
        self.assertEqual(compositor.output_content_rectangle([1080,1920]),[185,0,455,480])
        review={'image_size':[1080,1920],'label':{'presence':'present','instances':[
            {'class_id':0,'xyxy':[100,100,1000,1800],'object_id':'fixture'}]}}
        self.assertEqual(compositor.existing_sign_boxes(review)[0]['output_box_xyxy'],[210,25,435,450])

    def test_real_occluder_recomputes_visible_box(self):
        background=np.zeros((480,640,3),dtype=np.uint8)
        sprite=np.full((100,40,4),255,dtype=np.uint8)
        _,box,geometry=compositor._compose_placed(background,sprite,'normal',{
            'height':200,'left':50,'top':150,
            'ground_polygon':[[0,300],[639,300],[639,479],[0,479]],
            'occlusion_polygons':[[[50,150],[130,150],[130,230],[50,230]]],
            'visible_kind':'fixture','visibility_evidence':'existing foreground surface'})
        self.assertEqual(box,[50,231,130,350])
        self.assertEqual(geometry['visible_alpha_pixels'],80*119)

    def test_background_with_old_fixture_cannot_claim_empty_label_contract(self):
        frame = {'image': 'background.png', 'sign_absence': 'existing_labelled',
                 'existing_sign_boxes': [{'class_id': 0, 'output_box_xyxy': [10, 20, 30, 80]}]}
        state = {'background': {'image': 'background.png'},
                 'target': {'state': 'negative', 'mask': 'none'},
                 'labels': {'class_id': 0, 'kind': 'retained_background_boxes'},
                 'augmentation': {'tone': 'normal'}, 'parameters': {'kind': 'empty_background'}}
        self.assertEqual(compositor.validate_planned_stratum(state, [frame]), frame)
        state['labels']['kind'] = 'empty_candidate'
        with self.assertRaisesRegex(ValueError, 'label contract'):
            compositor.validate_planned_stratum(state, [frame])

    def test_near_fixture_is_clipped_by_image_without_deleting_source_alpha(self):
        background = np.zeros((480, 640, 3), dtype=np.uint8)
        sprite = np.full((100, 40, 4), 255, dtype=np.uint8)
        before = sprite.copy()
        image, box, geometry = compositor._compose_placed(background, sprite, 'normal', {
            'height': 620, 'left': 200, 'top': -160,
            'ground_polygon': [[0, 350], [639, 350], [639, 479], [0, 479]],
            'occlusion_polygons': [], 'visible_kind': 'fixture',
            'visibility_evidence': 'fixture extends above the image', 'layout': 'top_clipped'})
        self.assertEqual(box, [200, 0, 448, 460])
        self.assertTrue(np.array_equal(sprite, before))
        self.assertTrue(np.all(image[0, 200:448] == 255))
        self.assertFalse(geometry['retain_lower_half'])

    def test_brightness_and_saturation_can_vary_independently(self):
        self.assertEqual(compositor.TONE_GAINS['dark'][1], 1)
        self.assertEqual(compositor.TONE_GAINS['low_saturation'][0], 1)
        self.assertLess(compositor.TONE_GAINS['dark'][0], 1)
        self.assertLess(compositor.TONE_GAINS['low_saturation'][1], 1)


if __name__ == '__main__':
    unittest.main()
