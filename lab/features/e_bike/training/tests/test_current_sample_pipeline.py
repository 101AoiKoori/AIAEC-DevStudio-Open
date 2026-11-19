import unittest
import numpy as np

from datasets.scripts.e_bike.build_current_sample import direct_objects
from datasets.scripts.e_bike.compose_v4_grounded_train import draw_contact_shadow


class CurrentSamplePipelineTest(unittest.TestCase):
    def test_direct_mirror_preserves_posture_and_reflects_visible_boxes(self):
        text='0 0.25 0.5 0.2 0.4\n1 0.7 0.6 0.1 0.2\n'
        normal=direct_objects(text,640,480)
        mirror=direct_objects(text,640,480,True)
        for first,second in zip(normal,mirror):
            x1,y1,x2,y2=first['bbox_xyxy']
            self.assertEqual(second['bbox_xyxy'],[640-x2,y1,640-x1,y2])
            self.assertEqual(first['class'],second['class'])
            self.assertNotIn('visible_fraction',second)

    def test_shadow_follows_off_centre_alpha_contact(self):
        frame=np.full((100,100,3),200,np.uint8)
        sprite=np.zeros((30,40,4),np.uint8)
        sprite[10:30,2:7,3]=255
        actor={'sprite':sprite,'foot_x':60,'foot_y':80}
        draw_contact_shadow(frame,actor)
        self.assertLess(frame[79,44,0],180)
        self.assertGreater(frame[79,60,0],195)


if __name__=='__main__':unittest.main()
