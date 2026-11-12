import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_name_ocr_candidates import check_rows, expected_rows, text_crop_state


class OcrCropTest(unittest.TestCase):
    def test_rejects_small_edge_and_low_contrast(self):
        image = np.full((80, 120, 3), 200, dtype=np.uint8)
        self.assertEqual(text_crop_state(image, [10, 10, 42, 22], '清清大厦'), 'too_small_for_text_review')
        self.assertEqual(text_crop_state(image, [0, 10, 80, 35], '清清大厦'), 'touches_image_edge_possible_truncation')
        self.assertEqual(text_crop_state(image, [10, 10, 90, 35], '清清大厦'), 'low_contrast_for_text_review')
        image[15:25, 30:70] = 0
        self.assertEqual(text_crop_state(image, [10, 10, 90, 35], '清清大厦'), 'candidate_requires_readability_review')

    def test_label_mismatch_is_not_silently_exported(self):
        expected = expected_rows([{'box': [10, 10, 90, 35]}], 120, 80)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'label.txt'
            path.write_text('\n'.join(expected), encoding='utf-8')
            check_rows(path, expected)
            path.write_text('0 0.5 0.5 0.1 0.1\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                check_rows(path, expected)


if __name__ == '__main__':
    unittest.main()
