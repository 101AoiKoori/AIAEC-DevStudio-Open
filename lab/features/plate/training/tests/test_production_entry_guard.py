"""Exercise the plate entry with a real temporary SQLite receipt cache."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/training_common/tests'))
from test_production_guard import ProductionGuardTests, guard

SPEC = importlib.util.spec_from_file_location('plate_composite_test', ROOT / 'lab/features/plate/training/scripts/compose_plate_vehicle_pilot.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PlateEntryGuardTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ProductionGuardTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.output = self.root / 'datasets/exports/plate/output'
        self.args = dict(self.fixture.args, output_paths=[self.output])
        self.calls = 0

    def generate(self, output, refs):
        self.calls += 1
        output.mkdir(parents=True)
        (output / 'manifest.json').write_text('{"status":"complete_candidates","samples":[]}', encoding='utf-8')
        return {'status': 'complete_candidates', 'samples': []}

    def actual_guard(self, *args, **kwargs):
        return guard.generation_guard(**self.args)

    def test_repeat_uses_receipt_without_rewriting(self):
        with patch.object(module, 'ROOT', self.root), patch.object(module, 'EXPORTS', self.output.parent), \
             patch.object(module, 'generation_guard', self.actual_guard), patch.object(module, '_generate', self.generate):
            module.generate(self.output)
            before = (self.output / 'manifest.json').stat().st_mtime_ns
            module.generate(self.output)
        self.assertEqual(self.calls, 1)
        self.assertEqual(before, (self.output / 'manifest.json').stat().st_mtime_ns)

    def test_missing_database_does_not_write(self):
        (self.root / 'datasets/annotated/media_index/query_cache.sqlite3').unlink()
        with patch.object(module, 'ROOT', self.root), patch.object(module, 'EXPORTS', self.output.parent), \
             patch.object(module, 'generation_guard', self.actual_guard), patch.object(module, '_generate', self.generate):
            with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
                module.generate(self.output)
        self.assertEqual(self.calls, 0)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
