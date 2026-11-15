"""The sign adapter uses real receipts and fails before writing without a DB."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/scripts/media/tests'), str(ROOT / 'lab/features/sign/training/scripts')]
import test_media_index_sqlite as fixtures
import sign_production_guard as adapter
import compose_sign_scene as scene


class SignProductionGuardTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CacheTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        shard = self.root / self.fixture.parts['datasets/raw/camera']['shard']
        item = json.loads(shard.read_text(encoding='utf8'))
        stat = self.fixture.image.stat()
        item.update(bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
        shard.write_text(json.dumps(item) + '\n', encoding='utf8')
        self.assertEqual(self.fixture.call(['sync'])[0], 0)
        local_helper = self.root / 'lab/features/sign/training/scripts/sign_production_guard.py'
        local_helper.parent.mkdir(parents=True)
        shutil.copy2(adapter.__file__, local_helper)
        for name in ('lab/training_common/production_guard.py', 'lab/scripts/media/media_index.py',
                     'lab/scripts/media/media_index_sqlite.py'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture', encoding='utf8')
        self.old_root, self.old_file = adapter.ROOT, adapter.__file__
        adapter.ROOT, adapter.__file__ = self.root, str(local_helper)
        self.addCleanup(setattr, adapter, 'ROOT', self.old_root)
        self.addCleanup(setattr, adapter, '__file__', self.old_file)
        (self.root / 'fixtures').mkdir(exist_ok=True)
        self.code = self.root / 'fixtures/sign_fixture.py'
        self.code.write_text('fixture', encoding='utf8')
        self.output = self.root / 'datasets/exports/sign/fixture'

    def call(self, executed):
        with adapter.guarded('fixture', self.output, [self.fixture.image], [],
                             {'seed': 7}, [self.code]) as run:
            if not run.reused:
                executed.append(True)
                self.output.mkdir(parents=True)
                (self.output / 'manifest.json').write_text('{"status":"complete"}', encoding='utf8')
            return run

    def test_second_call_reuses_without_rewriting(self):
        executed = []
        first = self.call(executed)
        before = (self.output / 'manifest.json').stat().st_mtime_ns
        second = self.call(executed)
        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(len(executed), 1)
        self.assertEqual(before, (self.output / 'manifest.json').stat().st_mtime_ns)
        self.assertTrue(second.source_refs['datasets/raw/camera/a.png'])

    def test_missing_database_never_writes_output(self):
        fixtures.cache.DATABASE.unlink()
        with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
            self.call([])
        self.assertFalse(self.output.exists())


class SignEntryCoverageTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'tmp').mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / 'tmp')
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / 'output'

    def test_all_sign_data_entrypoints_guard_the_writing_body(self):
        entries = {
            'compose_sign_scene.py': ('generate', '_generate'),
            'compose_sign_val_procedural.py': ('generate', '_generate'),
            'compose_sign_v4_measured.py': ('generate', '_generate'),
            'compose_sign_vehicle_pilot.py': ('generate', '_generate'),
            'compose_sign_arena_v4.py': ('main', '_generate'),
            'prepare_sign_video_candidates.py': ('main', '_run'),
            'prepare_sign_v4_real_eval_draft.py': ('main', '_main'),
            'assemble_sign_v4_draft.py': ('main', '_main'),
            'render_sign_arena_backgrounds.py': ('main', '_main'),
        }
        scripts = ROOT / 'lab/features/sign/training/scripts'
        for name, (entry, writing_body) in entries.items():
            with self.subTest(entry=name):
                path = (ROOT / 'datasets/scripts/sign' / name
                        if name in {'compose_sign_v4_measured.py', 'compose_sign_scene.py', 'compose_sign_vehicle_pilot.py'} else scripts / name)
                tree = ast.parse(path.read_text(encoding='utf8'))
                fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry)
                guarded = [node for node in ast.walk(fn) if isinstance(node, ast.With) and any(
                    isinstance(item.context_expr, ast.Call) and
                    isinstance(item.context_expr.func, ast.Name) and item.context_expr.func.id == 'guarded'
                    for item in node.items)]
                self.assertEqual(len(guarded), 1)
                self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                                    node.func.id == writing_body for node in ast.walk(guarded[0])))

    def test_scene_entry_reuses_without_running_generator(self):
        @contextmanager
        def reused(*args, **kwargs):
            yield SimpleNamespace(reused=True)
        with mock.patch.object(adapter, 'guarded', reused), \
                mock.patch.object(adapter, 'existing_result', return_value={'status': 'complete'}) as read, \
                mock.patch.object(scene, '_generate', side_effect=AssertionError('must not run')), \
                mock.patch.object(scene, 'train_backgrounds', return_value=[]), \
                mock.patch.object(scene, '_prepare_plan', return_value={}), \
                mock.patch.object(scene.json, 'loads', return_value={'records': []}), \
                mock.patch.object(Path, 'read_text', return_value='{}'):
            self.assertEqual(scene.generate(self.output, 'train', 10, 7, {'budget': 10})['status'], 'complete')
            read.assert_called_once_with(self.output)
        self.assertFalse(self.output.exists())

    def test_scene_entry_database_failure_never_runs_generator(self):
        with mock.patch.object(adapter, 'guarded', side_effect=ValueError('database unavailable')), \
                mock.patch.object(scene, '_generate', side_effect=AssertionError('must not run')), \
                mock.patch.object(scene, 'train_backgrounds', return_value=[]), \
                mock.patch.object(scene, '_prepare_plan', return_value={}), \
                mock.patch.object(scene.json, 'loads', return_value={'records': []}), \
                mock.patch.object(Path, 'read_text', return_value='{}'):
            with self.assertRaisesRegex(ValueError, 'database unavailable'):
                scene.generate(self.output, 'train', 10, 7, {'budget': 10})
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
