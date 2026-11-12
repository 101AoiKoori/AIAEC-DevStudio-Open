import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'extend_real_scene_candidates.py'
spec = importlib.util.spec_from_file_location('extend_real_scene_candidates', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CandidatePackageTest(unittest.TestCase):
    def test_label_bounds_and_classes(self):
        with tempfile.TemporaryDirectory() as folder:
            label = Path(folder) / 'person.txt'
            label.write_text('0 0.5 0.5 0.4 0.6\n1 0.2 0.2 0.1 0.2\n', encoding='utf-8')
            self.assertEqual(module.validate_yolo(label), 2)
            for line in ('2 0.5 0.5 0.2 0.2', '0 0.01 0.5 0.5 0.2', '0 nan 0.5 0.2 0.2'):
                label.write_text(line, encoding='utf-8')
                with self.assertRaises(ValueError):
                    module.validate_yolo(label)

    def test_source_cannot_escape_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve()
            (base / 'image.png').write_bytes(b'example')
            self.assertEqual(module.checked_file(base, 'image.png'), base / 'image.png')
            with self.assertRaises(ValueError):
                module.checked_file(base, '../outside.png')


if __name__ == '__main__':
    unittest.main()
