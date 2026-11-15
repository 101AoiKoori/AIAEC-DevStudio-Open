"""Composition entrypoints honor the shared production gate before writing."""
import contextlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


class ComposeGuardTest(unittest.TestCase):
    def test_distinct_reuse_and_gate_failure(self):
        module = importlib.import_module('compose_distinct_sticker_pilot')
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            exports = root / 'datasets/exports/waste_bin'
            output = exports / 'fixture'
            writes = []
            receipts = set()

            @contextlib.contextmanager
            def guard(operation, inputs, outputs, parameters, **kwargs):
                reused = operation in receipts
                yield SimpleNamespace(reused=reused)
                receipts.add(operation)

            def produce(destination):
                writes.append(destination)
                destination.mkdir(parents=True)
                result = {'status': 'complete_candidates', 'samples': []}
                (destination / 'manifest.json').write_text(json.dumps(result), encoding='utf8')
                return result

            with patch.object(module, 'ROOT', root), patch.object(module, 'EXPORTS', exports), \
                 patch.object(module, '_generate', side_effect=produce), \
                 patch('lab.training_common.production_guard.generation_guard', guard):
                module.generate(output)
                before = (output / 'manifest.json').stat().st_mtime_ns
                module.generate(output)
                self.assertEqual(len(writes), 1)
                self.assertEqual((output / 'manifest.json').stat().st_mtime_ns, before)
                @contextlib.contextmanager
                def unavailable(*args, **kwargs):
                    raise ValueError('database unavailable')
                    yield
                with patch('lab.training_common.production_guard.generation_guard', unavailable):
                    with self.assertRaisesRegex(ValueError, 'database unavailable'):
                        module.generate(exports / 'rejected')
                self.assertFalse((exports / 'rejected').exists())


if __name__ == '__main__':
    unittest.main()
