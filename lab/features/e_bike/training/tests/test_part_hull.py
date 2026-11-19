import unittest
import cv2
from lab.features.e_bike.training.scripts.part_hull_experiment import associate,constrain,thick_hull


def vehicle(points,confidence=.9):
    return {'xyxy':[0,0,100,100],'confidence':confidence,'class_id':0,
            'keypoints':[[x,y,1] for x,y in points]}


class PartHullTest(unittest.TestCase):
    def test_duplicate_semantic_parts_have_one_owner(self):
        a=vehicle([(20,80),(70,80),(60,40),(25,35)])
        b=vehicle([(21,80),(71,80),(61,40),(26,35)],.8)
        kept,removed=constrain(associate([a,b]),True)
        self.assertEqual((len(kept),removed[0]['reason']),(1,'two_shared_parts'))

    def test_overlapping_hulls_with_distinct_parts_are_retained(self):
        a=vehicle([(20,80),(70,80),(60,40),(25,35)])
        b=vehicle([(35,80),(85,80),(75,40),(40,35)],.8)
        kept,removed=constrain(associate([a,b]),True)
        self.assertEqual((len(kept),len(removed)),(2,0))

    def test_unknown_parts_are_not_deletion_evidence(self):
        a=vehicle([(20,80),(70,80)]);a['keypoints']=[[20,80,0],[70,80,0]]
        kept,removed=constrain(associate([a,a]),True)
        self.assertEqual((len(kept),len(removed)),(2,0))

    def test_two_collinear_points_have_positive_thickness(self):
        hull=thick_hull([[0,0],[10,0]],2)
        self.assertGreater(cv2.contourArea(hull),40)

    def test_one_shared_part_requires_hull_evidence(self):
        a=vehicle([(20,80),(70,80),(60,40),(25,35)])
        b=vehicle([(20,80),(80,80),(70,40),(35,35)],.8)
        candidates=associate([a,b])
        self.assertEqual(len(constrain(candidates,False)[0]),2)
        self.assertEqual(constrain(candidates,True)[1][0]['reason'],'shared_part_and_hull')


if __name__=='__main__':unittest.main()
