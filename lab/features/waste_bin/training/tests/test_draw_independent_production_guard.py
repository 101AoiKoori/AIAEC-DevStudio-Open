"""Exercise the procedural artwork entry with a temporary SQLite index."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lab/scripts/media/tests'))
import test_media_index_sqlite as fixtures
from lab.training_common import production_guard

SCRIPT = ROOT / 'lab/features/waste_bin/training/scripts/draw_independent_sticker_eval.py'
spec = importlib.util.spec_from_file_location('draw_guard_test_target', SCRIPT)
draw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(draw)


class DrawingGuardTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CacheTests('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        config = self.root / 'config/perception_tasks.json'
        config.parent.mkdir(parents=True)
        config.write_text('{"tasks": []}', encoding='utf-8')
        (self.root / 'fixtures').mkdir(exist_ok=True)
        self.script = self.root / 'fixtures/draw.py'
        self.script.write_text('# fixture\n', encoding='utf-8')
        self.out = self.root / 'datasets/exports/waste_bin/fixture'
        self.calls = []
        def produce(args, out, font):
            self.calls.append(1)
            out.mkdir(parents=True)
            (out / 'manifest.json').write_text('{}', encoding='utf-8')
        for patch in (mock.patch.object(draw, 'ROOT', self.root),
                      mock.patch.object(draw, '__file__', str(self.script)),
                      mock.patch.object(production_guard, '_index', return_value=fixtures.cache),
                      mock.patch.object(sys, 'argv', ['draw', '--out', str(self.out)]),
                      mock.patch.object(draw, '_draw', side_effect=produce)):
            patch.start()
            self.addCleanup(patch.stop)

    def test_repeat_does_not_execute_or_rewrite(self):
        self.assertEqual(self.fixture.call(['sync'])[0], 0)
        with redirect_stdout(io.StringIO()):
            draw.main()
            result = self.out / 'manifest.json'
            before = result.stat().st_mtime_ns
            draw.main()
        self.assertEqual(self.calls, [1])
        self.assertEqual(result.stat().st_mtime_ns, before)

    def test_missing_database_never_writes(self):
        with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
            draw.main()
        self.assertFalse(self.out.exists())
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
