"""Verify paper geometry, untouched pixels, and preservation of fixed artwork."""
import unittest
from pathlib import Path
import numpy as np
from datasets.scripts.waste_sticker.build_hard_negative_finetune import (
    ROOT, complete_empty, ground_stripes, paper_on_image, pixel_box,
)
from datasets.scripts.waste_sticker.source_inputs import artwork_catalog


class HardNegativeGeometryTests(unittest.TestCase):
    def test_white_card_and_target_share_box_but_have_different_ink(self):
        frame=np.full((100,150,3),65,np.uint8)
        face={'occluder_polygons':[]};quad=[[25,20],[85,22],[80,80],[24,77]]
        blank=frame.copy();positive=frame.copy()
        a=paper_on_image(blank,face,quad)
        b=paper_on_image(positive,face,quad,ROOT/artwork_catalog()[0])
        self.assertEqual(a,b)
        self.assertTrue(np.array_equal(blank[:15],frame[:15]))
        self.assertFalse(np.array_equal(blank,positive))

    def test_ground_projection_keeps_pixels_outside_patch(self):
        frame=np.full((90,160,3),70,np.uint8)
        result=ground_stripes(frame,[[5,55],[150,55],[158,88],[1,88]],123)
        np.testing.assert_array_equal(result[:50],frame[:50])
        self.assertTrue(np.any(result[60:]!=frame[60:]))

    def test_positive_or_unknown_source_cannot_become_empty(self):
        for label in [{},{'presence':'present','completeness':'complete'},
                      {'presence':'absent','completeness':'partial'}]:
            with self.assertRaises(ValueError): complete_empty({'frame_id':'fixture','label':label})

    def test_decimal_rounding_is_clipped_but_bad_geometry_is_rejected(self):
        self.assertEqual(pixel_box([-.00001,0,100.00002,60],100,60),[0,0,100,60])
        with self.assertRaises(ValueError): pixel_box([-2,0,100,60],100,60)


if __name__=='__main__': unittest.main()
