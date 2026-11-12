"""Small tests for contrast screening and source-point geometry."""
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from audit_meter_source_quality import needle_contrast
from generate_meter_from_renders import transform


class MeterQualityTests(unittest.TestCase):
    def test_needle_contrast_detects_washed_out_line(self):
        row=[0,.5,.5,.8,.8, .2,.5,2, .8,.5,2, .5,.2,2, .5,.8,2]+[0,0,0]*4
        crisp=np.full((1024,1024),200,np.uint8)
        washed=crisp.copy()
        cv2.line(crisp,(205,512),(819,512),20,4)
        cv2.line(washed,(205,512),(819,512),180,4)
        self.assertGreater(needle_contrast(crisp,row,0),100)
        self.assertLess(needle_contrast(washed,row,0),35)

    def test_perspective_points_are_reversible(self):
        H=np.array([[1.04,.07,23],[-.03,.96,17],[.0001,-.00005,1]],dtype=np.float64)
        points=np.array([[10,25],[500,300],[800,900]],dtype=np.float32)
        restored=transform(transform(points,H),np.linalg.inv(H))
        self.assertLess(float(np.max(np.abs(restored-points))),.001)


if __name__=='__main__':
    unittest.main()
