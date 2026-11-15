"""Test deterministic split selection without using production data."""
import importlib.util
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = ROOT / 'lab/features/e_bike/training/scripts/prepare_v4_parking_eval.py'
spec = importlib.util.spec_from_file_location('parking_eval', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ParkingEvalTest(unittest.TestCase):
    def test_balanced_distinct_selection_excludes_known_bad(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            occupied = root / 'occupied.png'
            occupied.write_bytes(b'occupied')
            rows = []
            for i in range(99):
                name = f'frame_{i:03}.png'
                (root / name).write_bytes(b'occupied' if i == 0 else f'candidate_{i}'.encode())
                rows.append({'image': name, 'tone': module.TONES[i % 3],
                             'plan': ('edge', 'person', 'mixed', 'fallen')[i % 4],
                             'background_parent': 'restricted.png' if i == 1 else 'frame.png',
                             'background_source_video': 'scene.mp4',
                             'objects': [{'class': 'fallen', 'foreground_source': {
                                 'path': 'bad.png' if i == 2 else 'good.png'}}]})
            with mock.patch.object(module, 'ROOT', root):
                chosen, skipped = module.select_rows(rows, {'bad'}, {'restricted.png'}, [occupied], 24, 48, 7)
                repeated, _ = module.select_rows(rows, {'bad'}, {'restricted.png'}, [occupied], 24, 48, 7)
            self.assertEqual([(s, i) for s, i, _ in chosen], [(s, i) for s, i, _ in repeated])
            self.assertEqual(Counter(s for s, _, _ in chosen), {'val': 24, 'test': 48})
            self.assertEqual(len({r['image'] for _, _, r in chosen}), 72)
            self.assertFalse({0, 1, 2} & {i for _, i, _ in chosen})
            self.assertEqual(skipped['restricted_or_bad_parent'], 2)
            self.assertEqual(skipped['identical_output'], 1)
            for split in ('val', 'test'):
                tone = Counter(r['tone'] for s, _, r in chosen if s == split)
                self.assertEqual(max(tone.values()), min(tone.values()))


if __name__ == '__main__':
    unittest.main()
