import unittest
import numpy as np
from lab.features.e_bike.training.scripts.build_focused_inputs import safe_crop,contains_no_vehicle,crop_resize


class FocusedInputTest(unittest.TestCase):
    def test_crop_closes_over_intersecting_vehicle_boxes(self):
        objects=[{'class_id':0,'xyxy':[30,30,60,60]},{'class_id':1,'xyxy':[55,40,80,70]}]
        rect=safe_crop([25,25,65,65],objects,100,100)
        frame,mapped=crop_resize(np.zeros((100,100,3),np.uint8),objects,rect)
        self.assertEqual(len(mapped),2);self.assertEqual(frame.shape,(480,640,3))
        self.assertGreaterEqual(rect[2],80)

    def test_background_negative_rejects_partial_vehicle_intersection(self):
        obj=[{'class_id':0,'xyxy':[30,30,60,60]}]
        self.assertFalse(contains_no_vehicle([0,0,31,31],obj))
        self.assertTrue(contains_no_vehicle([0,0,20,20],obj))

    def test_crop_does_not_silently_truncate_labels(self):
        with self.assertRaisesRegex(ValueError,'truncates'):
            crop_resize(np.zeros((100,100,3),np.uint8),[{'class_id':0,'xyxy':[30,30,60,60]}],[0,0,40,40])

    def test_fractional_boundary_is_closed_before_integer_crop(self):
        objects=[{'class_id':0,'xyxy':[20,20,25.1,40]},{'class_id':1,'xyxy':[30,30,60,60]}]
        rect=safe_crop([25.2,15,65,65],objects,100,100)
        _,mapped=crop_resize(np.zeros((100,100,3),np.uint8),objects,rect)
        self.assertEqual(len(mapped),2)

    def test_label_roundoff_at_image_boundary_is_tolerated(self):
        _,mapped=crop_resize(np.zeros((360,640,3),np.uint8),[{'class_id':0,'xyxy':[0,-.00018,640+.0001,70]}],[0,0,640,360])
        self.assertEqual(len(mapped),1)


if __name__=='__main__':unittest.main()
