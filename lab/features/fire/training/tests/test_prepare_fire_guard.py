"""The frame extractor must not write when a receipt is reused or DB fails."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/features/fire/training/scripts'))
import prepare_fire_videos as module


class PrepareGuardTests(unittest.TestCase):
    def test_prepare_reuses_without_rewriting_and_fails_closed(self):
        temp_root = ROOT / 'tmp'
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            base = Path(tmp)
            folder, output = base / 'source/train', base / 'output'
            folder.mkdir(parents=True)
            video = folder / 'sample.mp4'
            video.write_bytes(b'fixture')
            calls = []

            @contextmanager
            def guard(operation, inputs, outputs, params, **kwargs):
                self.assertEqual(inputs, [video])
                self.assertEqual(outputs, [output])
                yield SimpleNamespace(reused=output.exists(), source_refs={video.relative_to(ROOT).as_posix(): []})

            def produce(groups, destination, fps, refs):
                calls.append((groups, fps, refs))
                destination.mkdir()
                records = [{'source': video.relative_to(ROOT).as_posix(), 'frames': 1}]
                (destination / 'manifest.json').write_text(json.dumps(records), encoding='utf8')
                return records

            with patch.object(module, 'generation_guard', guard), patch.object(module, '_prepare_groups', produce):
                first = module.prepare(base / 'source', output, 1.0)
                stamp = (output / 'manifest.json').stat().st_mtime_ns
                second = module.prepare(base / 'source', output, 1.0)
                self.assertEqual(first, second)
                self.assertEqual(len(calls), 1)
                self.assertEqual((output / 'manifest.json').stat().st_mtime_ns, stamp)

            @contextmanager
            def unavailable(*args, **kwargs):
                raise ValueError('SQLite unavailable')
                yield

            rejected = base / 'rejected'
            with patch.object(module, 'generation_guard', unavailable), patch.object(module, '_prepare_groups') as body:
                with self.assertRaisesRegex(ValueError, 'SQLite unavailable'):
                    module.prepare(base / 'source', rejected, 1.0)
                body.assert_not_called()
                self.assertFalse(rejected.exists())


if __name__ == '__main__':
    unittest.main()
