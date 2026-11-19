"""Endpoint direction, long arcs, independent unknown states and equal ticks."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'ws/src/aec_vision/src'))
from aec_vision.meter import equal_angle_scale, read_pose_dials

SCALES={'temperature':dict(minimum=-30,maximum=50,unit='°C',tick_step=1,major_step=10,direction='clockwise'),
        'humidity':dict(minimum=0,maximum=100,unit='%RH',tick_step=2,major_step=10,direction='clockwise')}
def point(degree,radius=80,center=(100,100)):
    a=math.radians(degree)
    return [center[0]+radius*math.cos(a),center[1]+radius*math.sin(a),.99]
def pose(temp_angle=270,hum_angle=90):
    return np.array([[100,100,.99],point(temp_angle),[100,200,.99],point(hum_angle,30,(100,200)),
                     point(150),point(30),point(-30,30,(100,200)),point(210,30,(100,200))])

class PoseReadoutTests(unittest.TestCase):
    def test_midpoints_and_long_arcs(self):
        r=read_pose_dials(pose(),SCALES)
        self.assertAlmostEqual(r['temperature']['value'],10)
        self.assertAlmostEqual(r['humidity']['value'],50)
        self.assertEqual(len(r['temperature']['scale']['ticks']),81)
        self.assertEqual(len(r['humidity']['scale']['ticks']),51)
        self.assertAlmostEqual(r['temperature']['scale']['sweep_deg'],240)

    def test_both_endpoints_and_blank_arc(self):
        for ta,ha,tv,hv in ((150,-30,-30,0),(30,210,50,100)):
            r=read_pose_dials(pose(ta,ha),SCALES)
            self.assertAlmostEqual(r['temperature']['value'],tv)
            self.assertAlmostEqual(r['humidity']['value'],hv)
        r=read_pose_dials(pose(90,-90),SCALES)
        self.assertIsNone(r['temperature']['value'])
        self.assertIsNone(r['humidity']['value'])

    def test_partial_unknown_does_not_fill_another_dial(self):
        p=pose();p[1,2]=.1
        r=read_pose_dials(p,SCALES)
        self.assertEqual(r['temperature']['status'],'unknown')
        self.assertEqual(r['humidity']['status'],'estimated')
        self.assertIsNotNone(r['temperature']['scale'])
        p=pose();p[6,0]=float('nan')
        r=read_pose_dials(p,SCALES)
        self.assertEqual(r['humidity']['status'],'unknown')
        self.assertEqual(r['temperature']['status'],'estimated')

    def test_degenerate_and_counterclockwise(self):
        p=pose();p[1,:2]=p[0,:2]
        self.assertEqual(read_pose_dials(p,SCALES)['temperature']['reason'],'needle_tip_coincides_with_pivot')
        s=equal_angle_scale((0,0),point(30,1,(0,0))[:2],point(150,1,(0,0))[:2],0,100,2,direction='counterclockwise')
        self.assertAlmostEqual(s['sweep_deg'],240)
        angles=[t['angle_deg'] for t in s['ticks']]
        self.assertTrue(np.all(np.diff(angles)<0))
        with self.assertRaises(ValueError):equal_angle_scale((0,0),(1,0),(1,0),0,100,2)
        with self.assertRaises(ValueError):equal_angle_scale((0,0),(1,0),(0,1),0,100,3)

if __name__=='__main__':unittest.main()
