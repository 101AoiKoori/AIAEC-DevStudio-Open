"""Exercise physical layer composition and the live low-view hybrid recipe."""
import unittest
from copy import deepcopy

import cv2
import numpy as np

from datasets.scripts.building_name_paper import hybrid_sample as hybrid


class HybridSampleTests(unittest.TestCase):
    def test_split_reallocation_reuses_pixels_but_physical_changes_require_render(self):
        before=hybrid.recipe();after=deepcopy(before)
        after['plates'][0]['source_split']='val'
        self.assertTrue(hybrid.same_render_recipe(before,after))
        after['plates'][0]['ground_polygon'][0][0]+=1
        self.assertFalse(hybrid.same_render_recipe(before,after))

    def test_shadow_attenuates_only_supported_floor_and_preserves_opaque_subject(self):
        photo = np.full((8, 8, 3), 160, np.uint8)
        lit = np.full_like(photo, 180)
        shadowed = hybrid.encoded(hybrid.linear(lit.astype(np.float32)/255)*.5)
        foreground = np.zeros((8,8,4),np.uint8)
        foreground[3:5,3:5] = [40,100,210,255]
        support = np.zeros((8,8),np.uint8)
        support[4:] = 1
        result, attenuation, _ = hybrid.compose_layers(photo,foreground,shadowed,lit,
            strength=.8,gain=[1,1,1],blur_sigma=0,shadow_support=support)
        np.testing.assert_array_equal(result[:3],photo[:3])
        np.testing.assert_array_equal(result[3:5,3:5],foreground[3:5,3:5,:3])
        self.assertLess(float(result[6,0,0]),160)
        self.assertTrue(np.all(attenuation[:4] == 1))

    def test_soft_edges_blur_premultiplied_colour_without_a_white_halo(self):
        photo = np.zeros((15,15,3),np.uint8)
        light = np.full_like(photo,180)
        foreground = np.full((15,15,4),255,np.uint8)
        foreground[:,:,3] = 0
        foreground[5:10,5:10] = [0,0,220,255]
        result, _, alpha = hybrid.compose_layers(photo,foreground,light,light,
            strength=.8,gain=[1,1,1],blur_sigma=.7)
        self.assertTrue(np.any((alpha>0)&(alpha<1)))
        self.assertEqual(int(result[:,:,:2].max()),0)

    def test_current_recipe_covers_all_building_combinations_on_both_real_plates(self):
        config = hybrid.recipe()
        plates = hybrid.qualified_plates(config)
        plan = hybrid.declaration(config,plates)
        self.assertEqual(len(plan['strata']),config['budget'])
        self.assertEqual(set(config['combinations']),{'A','B','C','AB','AC','BC','ABC'})
        for plate in plates:
            states = [s for s in plan['strata'] if s['background']['plate_id']==plate['id']]
            self.assertEqual(len(states),14)
            self.assertEqual({s['parameters']['camera_height_m'] for s in states},{.14,.20})
            self.assertEqual({s['parameters']['yaw_degrees'] for s in states},{-70,-50,0})
            self.assertTrue(200 < plate['perspective']['horizon_y_px'] < 300)
            self.assertEqual(plate['source_review']['label']['instances'],[])

    def test_layer_shape_mismatch_fails_before_compositing(self):
        with self.assertRaisesRegex(ValueError,'exact image coordinates'):
            hybrid.compose_layers(np.zeros((8,8,3),np.uint8),np.zeros((4,8,4),np.uint8),
                                  np.zeros((8,8,3),np.uint8),np.zeros((8,8,3),np.uint8),
                                  strength=.8,gain=[1,1,1],blur_sigma=0)
