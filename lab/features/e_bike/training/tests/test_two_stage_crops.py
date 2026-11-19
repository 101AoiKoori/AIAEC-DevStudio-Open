import unittest
import numpy as np

from lab.features.e_bike.training.scripts.two_stage_experiment import crop, compact


class VehicleCropTest(unittest.TestCase):
    def test_wide_vehicle_keeps_both_ends_with_square_padding(self):
        image = np.zeros((30,120,3),np.uint8)
        image[10:20,10:20] = [0,0,255]
        image[10:20,100:110] = [0,255,0]
        patch,_ = crop(image,[10,10,110,20])
        self.assertEqual(patch.shape,(160,160,3))
        self.assertTrue(np.any((patch[:,:,2]>200)&(patch[:,:,1]<30)))
        self.assertTrue(np.any((patch[:,:,1]>200)&(patch[:,:,2]<30)))
        self.assertTrue(np.all(patch[0]==114))

    def test_edge_context_clamps_to_image(self):
        image = np.full((40,50,3),200,np.uint8)
        patch,box = crop(image,[0,0,50,40])
        self.assertEqual(box,[0,0,50,40])
        self.assertEqual(patch.shape,(160,160,3))

    def test_invalid_detection_is_not_silently_classified(self):
        with self.assertRaises(ValueError):
            crop(np.zeros((30,30,3),np.uint8),[100,100,110,110])

    def test_pose_error_does_not_become_entity_detection_failure(self):
        row = {'source_image':'example.png','ground_truth':[{'class_id':0,'xyxy':[0,0,20,20]}],
               'predictions':[{'class_id':1,'xyxy':[0,0,20,20]}],
               'matches':[],'missed_gt_indices':[0],'false_prediction_indices':[0]}
        result = compact([row])
        self.assertEqual(result['exact_images'],0)
        self.assertEqual(result['entity_counts_and_matching']['all_matched'],1)


if __name__=='__main__':
    unittest.main()
