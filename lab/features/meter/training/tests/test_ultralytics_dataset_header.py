import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_meter_ultralytics import validate_dataset_header


class DatasetHeaderTest(unittest.TestCase):
    def test_omitted_and_explicit_root(self):
        root = Path(__file__).resolve().parent
        data = {'kpt_shape': [8, 3], 'names': {0: 'meter'}}
        validate_dataset_header(data, root)
        validate_dataset_header({**data, 'path': str(root)}, root)
        with self.assertRaises(RuntimeError):
            validate_dataset_header({**data, 'path': str(root.parent)}, root)


if __name__ == '__main__':
    unittest.main()
