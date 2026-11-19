import unittest
from unittest import mock
import numpy as np
from lab.features.e_bike.training.scripts.keypoint_experiment import anchor_suppress, projected_seat, projected_points, parts_suppress


def prediction(box,center,seat,score=.8):
    return {'xyxy':box,'keypoints':[center,seat],'confidence':score}


class AnchorSuppressionTest(unittest.TestCase):
    def test_multiple_parts_keep_order_under_mirroring(self):
        rgba=np.full((20,40,4),255,np.uint8)
        spec={'yaw':0,'pitch':0,'roll':0,'size_px':80,'foot_x':100,'foot_y':100,'mirrored':False}
        module='lab.features.e_bike.training.scripts.keypoint_experiment'
        points=[[5,5],[30,5],None,[15,15]]
        with mock.patch(module+'.read_image',return_value=rgba),mock.patch(module+'.load_rgba',return_value=rgba):
            _,normal=projected_points(None,points,spec,'fallen')
            _,mirrored=projected_points(None,points,{**spec,'mirrored':True},'fallen')
        self.assertIsNone(mirrored[2])
        for index in (0,1,3):
            self.assertAlmostEqual(normal[index][0]+mirrored[index][0],199)
            self.assertAlmostEqual(normal[index][1],mirrored[index][1])

    def test_four_parts_distinguish_overlapping_vehicle_instances(self):
        points=[[20,80,.9],[60,80,.9],[60,25,.9],[20,25,.9]]
        first={'xyxy':[0,0,100,100],'keypoints':points,'confidence':.9}
        second={'xyxy':[0,0,100,100],'keypoints':[[x+20,y,c] for x,y,c in points],'confidence':.8}
        self.assertEqual(len(parts_suppress([first,second],.15)),2)
        duplicate={**second,'keypoints':[[x+1,y,c] for x,y,c in points]}
        self.assertEqual(parts_suppress([duplicate,first],.15),[first])

    def test_four_part_unknowns_fall_back_to_box_suppression(self):
        first={'xyxy':[0,0,100,100],'keypoints':[[0,0,0]]*4,'confidence':.9}
        second={**first,'confidence':.8}
        self.assertEqual(parts_suppress([first,second],.15),[first])

    def test_seat_projection_mirrors_with_sprite_placement(self):
        rgba=np.full((20,40,4),255,np.uint8)
        spec={'yaw':0,'pitch':0,'roll':0,'size_px':80,'foot_x':100,'foot_y':100,'mirrored':False}
        module='lab.features.e_bike.training.scripts.keypoint_experiment'
        with mock.patch(module+'.read_image',return_value=rgba),mock.patch(module+'.load_rgba',return_value=rgba):
            normal,p1=projected_seat(None,[10,8],spec,'fallen')
            mirrored,p2=projected_seat(None,[10,8],{**spec,'mirrored':True},'fallen')
        np.testing.assert_array_equal(normal[:,::-1],mirrored)
        self.assertAlmostEqual(p1[0]+p2[0],199)
        self.assertAlmostEqual(p1[1],p2[1])

    def test_overlapping_boxes_with_distinct_seats_survive(self):
        first=prediction([0,0,100,100],[50,50,.9],[30,30,.9])
        second=prediction([0,0,100,100],[50,50,.9],[70,30,.9],.7)
        self.assertEqual(len(anchor_suppress([first,second],'center',.15)),1)
        self.assertEqual(len(anchor_suppress([first,second],'seat',.15)),2)

    def test_duplicate_seat_keeps_highest_scoring_instance(self):
        first=prediction([0,0,100,100],[50,50,.9],[30,30,.9])
        second=prediction([0,0,100,100],[55,55,.9],[31,30,.9],.7)
        self.assertEqual(anchor_suppress([second,first],'seat',.15),[first])

    def test_unknown_zero_points_do_not_merge_distant_vehicles(self):
        first=prediction([10,10,50,50],[0,0,0],[0,0,0])
        second=prediction([70,10,110,50],[0,0,0],[0,0,0],.7)
        self.assertEqual(len(anchor_suppress([first,second],'seat',.15)),2)


if __name__=='__main__':
    unittest.main()
