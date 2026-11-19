import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_sign_video_candidates import DEFAULT_SOURCE, ROOT, candidate_box, yolo_line


class SignVideoCandidateTests(unittest.TestCase):
    def test_default_source_uses_categorized_sign_video(self):
        self.assertEqual(
            DEFAULT_SOURCE.relative_to(ROOT).as_posix(),
            'Archive/Assets/working_sources/legacy/sign/视频/禁止直行牌视频.mp4',
        )

    def test_missing_colored_assembly_is_unknown_not_an_empty_negative(self):
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        self.assertIsNone(candidate_box(frame))
        cv2.rectangle(frame, (400, 100), (450, 1100), (0, 255, 255), -1)
        self.assertIsNone(candidate_box(frame))

    def test_detects_assembly_and_box_stays_in_image(self):
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        cv2.rectangle(frame, (400, 100), (450, 1100), (0, 255, 255), -1)
        cv2.rectangle(frame, (300, 1100), (600, 1250), (0, 255, 0), -1)
        cv2.circle(frame, (430, 350), 150, (0, 0, 255), 20)
        box = candidate_box(frame)
        self.assertIsNotNone(box)
        self.assertLessEqual(box[0], 280)
        self.assertLessEqual(box[1], 100)
        self.assertGreaterEqual(box[2], 600)
        self.assertGreaterEqual(box[3], 1250)
        row = yolo_line(box).split()
        self.assertEqual(row[0], '0')
        self.assertTrue(all(0 < float(value) <= 1 for value in row[1:]))

    def test_rejects_wrong_dimensions_and_invalid_box(self):
        with self.assertRaises(ValueError):
            candidate_box(np.zeros((100, 100, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            yolo_line([10, 20, 5, 30])


if __name__ == '__main__':
    unittest.main()
