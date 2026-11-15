"""Check that audit output cannot run before its production guard."""
import importlib.util
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = ROOT / 'lab/features/meter/training/scripts/audit_station_mask_candidates.py'
spec = importlib.util.spec_from_file_location('meter_audit_guard_boundary', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AuditGuardBoundaryTest(unittest.TestCase):
    def test_reuse_and_database_failure_skip_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            batch = Path(directory)
            (batch / 'samples.jsonl').write_text('', encoding='utf8')
            calls = []

            @contextmanager
            def reused(*args, **kwargs):
                class Result:
                    reused = True
                yield Result()

            @contextmanager
            def failed(*args, **kwargs):
                raise RuntimeError('database unavailable')
                yield

            with patch.object(module, 'generation_guard', reused), patch.object(module, '_execute', lambda args: calls.append(1)), patch.object(sys, 'argv', ['audit', str(batch)]):
                module.main()
            with patch.object(module, 'generation_guard', failed), patch.object(module, '_execute', lambda args: calls.append(1)), patch.object(sys, 'argv', ['audit', str(batch)]):
                with self.assertRaisesRegex(RuntimeError, 'database unavailable'):
                    module.main()
            self.assertEqual(calls, [])
            self.assertFalse((batch / 'mask_audit.json').exists())


if __name__ == '__main__':
    unittest.main()
