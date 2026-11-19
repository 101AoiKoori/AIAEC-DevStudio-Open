"""Verify that raster placement uses only applicable image-scale evidence."""

from types import SimpleNamespace
import unittest

import numpy as np

from datasets.scripts.waste_bin_state.generate_waste_v4_grouped import _scaled_group_layout


class BinScalePlacementTests(unittest.TestCase):
    def setUp(self):
        self.floor = np.zeros((480, 640), dtype=bool)
        self.floor[160:420] = True
        self.assessment = SimpleNamespace(candidate_mask=self.floor, contact_status=lambda *args, **kwargs: 'candidate')
        self.background = {'source_image_size': [640, 360], 'output_content_xyxy': [0, 60, 640, 420],
                           'ground_polygon_xy': [[0, 160], [639, 160], [639, 419], [0, 419]],
                           'perspective_scale_observation': {'evidence_id': 'frame-evidence',
                                                             'algorithm_version': 'v5', 'scale': {},
                                                             'local_stripe_scale': {}}}
        self.masks = {'blue': np.ones((20, 12), dtype=bool)}
        self.nominal = [190, 240, 48, 80]

    def layout(self):
        return _scaled_group_layout(self.masks, self.nominal, self.background, self.assessment, {})

    def test_horizon_scales_at_unchanged_contact_in_native_pixels(self):
        self.background['perspective_scale_observation']['scale'] = {
            'status': 'estimated', 'basis': 'ground_supported_vanishing_line', 'horizon_line_px': [0, 1, -100]}
        placement, _, supports, trace = self.layout()
        self.assertEqual(trace['foot_xy'], [214.0, 259.0])
        self.assertEqual(trace['reference_xy'], [214.0, 359.0])
        self.assertAlmostEqual(trace['ratio'], 159 / 259)
        self.assertEqual(placement[-1], 49)
        self.assertEqual(supports['blue'][1], [214, 319])

    def test_stripe_requires_axis_membership_and_records_uncertainty(self):
        stripe = {'status': 'estimated', 'basis': 'ground_supported_stripe_width_change',
                  'center_anchors_px': [[214, 220], [214, 320]], 'anchor_widths_px': [12, 20],
                  'width_error_sensitivity': [1.46, 1.91]}
        self.background['perspective_scale_observation']['local_stripe_scale'] = stripe
        placement, _, _, trace = self.layout()
        self.assertEqual(trace['scale_basis'], stripe['basis'])
        self.assertEqual(trace['width_error_sensitivity'], [1.46, 1.91])
        self.assertEqual(placement[-1], 60)
        stripe['center_anchors_px'] = [[217, 220], [217, 320]]
        placement, _, _, trace = self.layout()
        self.assertEqual(trace['scale_basis'], 'planned_approximation')
        self.assertIsNone(trace['ratio'])
        self.assertEqual(placement, self.nominal)

    def test_layout_outside_saved_height_range_keeps_plan(self):
        self.background['perspective_scale_observation']['scale'] = {
            'status': 'estimated', 'basis': 'ground_supported_vanishing_line', 'horizon_line_px': [0, 1, -100]}
        self.background['tested_height_range_output_px'] = [70, 90]
        placement, _, _, trace = self.layout()
        self.assertEqual(placement, self.nominal)
        self.assertEqual(trace['reason'], 'scaled_height_outside_tested_range')
        self.assertIsNone(trace['ratio'])
        self.assertAlmostEqual(trace['rejected_observation']['ratio'], 159 / 259)
