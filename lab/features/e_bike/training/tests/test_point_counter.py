import unittest
from unittest import mock
import numpy as np
import torch
from lab.features.e_bike.training.scripts.point_counter_experiment import letterbox,point_targets,point_set_loss,match_points


class IndependentPointCounterTest(unittest.TestCase):
    def test_duplicate_points_match_only_one_vehicle(self):
        gt=[{'class_id':0,'xyxy':[0,0,20,20]}]
        predictions=[{'class_id':0,'point_xy':[10,10]},{'class_id':0,'point_xy':[10,10]}]
        matches,missed,extra=match_points(gt,predictions)
        self.assertEqual((len(matches),len(missed),len(extra)),(1,0,1))

    def test_point_matching_does_not_ignore_posture(self):
        matched,missed,extra=match_points([{'class_id':0,'xyxy':[0,0,20,20]}],[{'class_id':1,'point_xy':[10,10]}])
        self.assertEqual((len(matched),len(missed),len(extra)),(0,1,1))

    def test_letterbox_and_target_coordinates_round_trip(self):
        _,transform=letterbox(np.zeros((100,200,3),np.uint8))
        with mock.patch('lab.features.e_bike.training.scripts.point_counter_experiment.truths',return_value=[{'class_id':1,'xyxy':[90,40,110,60]}]):
            points,labels=point_targets({},transform)
        np.testing.assert_allclose(points.numpy(),[[.5,.5]])
        self.assertEqual(labels.tolist(),[2])
        self.assertAlmostEqual((float(points[0,0])*640-transform['left'])/transform['scale_x'],100)
        self.assertAlmostEqual((float(points[0,1])*640-transform['top'])/transform['scale_y'],50)

    def test_hungarian_supervision_makes_one_foreground_and_one_background(self):
        logits=torch.zeros((1,2,3),requires_grad=True)
        outputs={'logits':logits,'points':torch.tensor([[[.5,.5],[.5,.5]]],requires_grad=True)}
        targets=[{'points':torch.tensor([[.5,.5]]),'labels':torch.tensor([1])}]
        loss,_=point_set_loss(outputs,targets);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients=logits.grad[0,:,0]
        self.assertEqual(int((gradients>0).sum()),1)
        self.assertEqual(int((gradients<0).sum()),1)

    def test_empty_image_has_finite_background_loss(self):
        outputs={'logits':torch.zeros((1,2,3),requires_grad=True),'points':torch.zeros((1,2,2),requires_grad=True)}
        loss,_=point_set_loss(outputs,[{'points':torch.empty((0,2)),'labels':torch.empty(0,dtype=torch.long)}])
        loss.backward();self.assertTrue(torch.isfinite(loss))


if __name__=='__main__':unittest.main()
