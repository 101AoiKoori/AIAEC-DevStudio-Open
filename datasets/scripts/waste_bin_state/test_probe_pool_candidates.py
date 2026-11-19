"""Batch footprint diagnostics must preserve contact and existing-bin constraints."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from datasets.scripts.common.paths import WORKSPACE
from datasets.scripts.common.review.placement import GroundPlacement
from datasets.scripts.waste_bin_state.probe_pool_candidates import (
    _current_label, _prior_indices, known_bin_boxes, probe_positions, relative_scale_cue,
)


class PoolCandidateProbeTests(unittest.TestCase):
    def test_finds_source_verified_contact_without_covering_an_existing_bin(self):
        candidate = np.zeros((480, 640), dtype=bool)
        candidate[260:430, 80:570] = True
        ground = GroundPlacement('candidate', 'fixture_floor', candidate, np.zeros_like(candidate))
        old = [110, 240, 310, 420]
        positions = probe_positions(ground, {'blue': np.ones((35, 25), dtype=bool)}, [old],
                                    heights=(90,), limit=3)
        self.assertTrue(positions)
        for position in positions:
            left, top, width, height = position['placement_xywh']
            self.assertFalse(min(left + width, old[2]) > max(left, old[0]) and
                             min(top + height, old[3]) > max(top, old[1]))
            points = position['supports']['blue']
            self.assertTrue(all(ground.contact_status(point, contact_width=position['contact_width_px'])
                                == 'candidate' for point in points))

    def test_no_supported_pixels_does_not_produce_a_candidate(self):
        empty = np.zeros((480, 640), dtype=bool)
        ground = GroundPlacement('unknown', 'fixture_floor', empty, empty.copy())
        self.assertEqual(probe_positions(ground, {'blue': np.ones((35, 25), dtype=bool)}, []), [])

    def test_previous_batches_are_excluded_by_pool_index(self):
        (WORKSPACE / 'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=WORKSPACE / 'tmp') as directory:
            first, second = (Path(directory) / name for name in ('first.json', 'second.json'))
            first.write_text(json.dumps({'records': [{'index': 20}, {'index': 45}]}), encoding='utf8')
            second.write_text(json.dumps({'records': [{'index': 45}, {'index': 95}]}), encoding='utf8')
            self.assertEqual(_prior_indices((first, second)), {20, 45, 95})

    def test_incomplete_and_excluded_bins_only_protect_known_pixels(self):
        label = {
            'completeness': 'partial', 'presence': 'present',
            'instances': [{'object_id': 'main', 'xyxy': [10, 20, 90, 200],
                           'lid_state': 'unknown'}],
            'excluded_instances': [{'object_id': 'distant', 'xyxy': [400, 40, 420, 70]},
                                   {'object_id': 'unlocated'}],
        }
        boxes = known_bin_boxes(label, [640, 480])
        self.assertEqual([item['annotation_group'] for item in boxes],
                         ['instances', 'excluded_instances'])
        self.assertIsNone(boxes[1]['reason'])
        candidate = np.zeros((480, 640), dtype=bool)
        candidate[260:450, 80:580] = True
        ground = GroundPlacement('candidate', 'fixture_floor', candidate, np.zeros_like(candidate))
        positions = probe_positions(ground, {'blue': np.ones((35, 25), dtype=bool)},
                                    [item['bbox_xyxy'] for item in boxes],
                                    heights=(90,), limit=1)
        self.assertTrue(positions)

    def test_partial_frame_label_is_reported_without_blocking_placement(self):
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('''CREATE TABLE observations
                        (frame_id TEXT, task TEXT, kind TEXT, current INTEGER,
                         evidence_id TEXT, payload_json TEXT)''')
        payload = {'scope': {'kind': 'frame'}, 'completeness': 'partial',
                   'presence': 'present', 'instances': [], 'excluded_instances': []}
        conn.execute('INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?)',
                     ('frame', 'waste_bin_state', 'label', 1, 'review', json.dumps(payload)))
        label, issue = _current_label(conn, 'frame')
        self.assertEqual(label[0]['evidence_id'], 'review')
        self.assertEqual(issue, 'complete_target_scope_label_pending')
        conn.close()

    def test_relative_scale_is_only_a_warning_against_existing_bins(self):
        old = [{'bin_id': 'near_old', 'bbox_xyxy': [30, 87, 196, 286]}]
        cue = relative_scale_cue([292, 256, 65, 90], old, content_height=360)
        self.assertEqual(cue['status'], 'rough_scale_too_small')
        self.assertGreater(cue['rough_expected_to_actual_ratio'], 2)
        self.assertEqual(relative_scale_cue([292, 256, 65, 90], [], content_height=360)
                         ['status'], 'no_comparable_existing_bin')


if __name__ == '__main__':
    unittest.main()
