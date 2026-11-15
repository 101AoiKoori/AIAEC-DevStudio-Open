"""Meter render guard contracts using temporary SQLite fixtures, without Blender."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS = ROOT / 'datasets/scripts/meter'
sys.path.insert(0, str(ROOT / 'lab/training_common/tests'))
sys.path.insert(0, str(SCRIPTS))
import test_production_guard as fixtures
from datasets.scripts.meter import meter_production_guard as meter


class MeterGuardTests(unittest.TestCase):
    def setUp(self):
        self.base = fixtures.ProductionGuardTests(methodName='runTest')
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.root = self.base.root
        self.old_root, self.old_file, self.old_guard = meter.ROOT, meter.__file__, meter.generation_guard
        self.addCleanup(setattr, meter, 'ROOT', self.old_root)
        self.addCleanup(setattr, meter, '__file__', self.old_file)
        self.addCleanup(setattr, meter, 'generation_guard', self.old_guard)
        meter.ROOT = self.root
        meter.__file__ = str(self.root / 'fixtures/meter_helper.py')
        Path(meter.__file__).write_text('fixture helper')
        meter.generation_guard = lambda *a, **kw: fixtures.guard.generation_guard(*a, **kw, root=self.root)
        self.args = SimpleNamespace(output=self.root / 'datasets/outputs/scene', count=2)

    def test_rebind_station_face_requires_exact_packed_source(self):
        face = SimpleNamespace(source='FILE', name='face.png', filepath='stale',
                               packed_file=SimpleNamespace(data=self.base.image.read_bytes()))
        bpy = SimpleNamespace(data=SimpleNamespace(images=[face]))
        meter.rebind_station_face(bpy, self.base.image)
        self.assertEqual(Path(face.filepath), self.base.image)
        face.filepath = 'stale'
        face.packed_file = SimpleNamespace(data=b'other')
        with self.assertRaisesRegex(ValueError, 'differs'):
            meter.rebind_station_face(bpy, self.base.image)
        self.assertEqual(face.filepath, 'stale')
        bpy.data.images = []
        with self.assertRaisesRegex(ValueError, 'expected one'):
            meter.rebind_station_face(bpy, self.base.image)

    def test_reuse_and_failed_database_do_not_execute_body(self):
        calls = []
        def execute():
            with meter.guarded('render', self.args, self.base.code_file, self.base.image,
                               extra_inputs=[self.base.config]) as run:
                if run.reused:
                    return
                calls.append('body')
                self.args.output.mkdir(parents=True)
                (self.args.output / 'manifest.json').write_text('{}')
        execute()
        path = self.args.output / 'manifest.json'
        before = path.stat().st_mtime_ns
        execute()
        self.assertEqual(calls, ['body'])
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.assertTrue(json.loads(path.read_text())['production_source_refs'])
        (self.root / 'datasets/annotated/media_index/query_cache.sqlite3').unlink()
        args = SimpleNamespace(output=self.root / 'datasets/outputs/blocked', count=2)
        with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
            with meter.guarded('render', args, self.base.code_file, self.base.image,
                               extra_inputs=[self.base.config]):
                self.fail('database failure entered body')
        self.assertFalse(args.output.exists())

    def test_preflight_result_reuse_without_output(self):
        expected = {'kind': 'station', 'summary': {'resolution': [640, 480], 'ground_angle_deg': 65.0}}
        calls = []
        for _ in range(2):
            with meter.guarded('preflight', self.args, self.base.code_file, self.base.image,
                               extra_inputs=[self.base.config], outputless=True) as run:
                if run.reused:
                    self.assertEqual(run.value, expected)
                else:
                    calls.append('scene mutation')
                    run.set_result(expected)
        self.assertEqual(calls, ['scene mutation'])
        self.assertFalse(self.args.output.exists())

    def test_blender_file_media_requires_accessible_workspace_source_when_packed(self):
        image = SimpleNamespace(source='FILE', name='face', filepath=str(self.base.image),
                                packed_file=SimpleNamespace(data=self.base.image.read_bytes()))
        bpy = SimpleNamespace(data=SimpleNamespace(images=[image]), path=SimpleNamespace(abspath=lambda p: p))
        self.assertEqual(meter.blender_media(bpy), [self.base.image])
        image.filepath = str(self.root / 'datasets/raw/camera/missing.png')
        with self.assertRaisesRegex(ValueError, 'packed media also requires'):
            meter.blender_media(bpy)
        image.filepath = ''
        with self.assertRaisesRegex(ValueError, 'no source path'):
            meter.blender_media(bpy)

    def test_changed_json_and_blender_media_fail_before_body(self):
        from contextlib import contextmanager
        real_guard = meter.generation_guard
        @contextmanager
        def changed_config(*args, **kwargs):
            self.base.config.write_text('{"seed":2}', encoding='utf8')
            with real_guard(*args, **kwargs) as run:
                yield run
        meter.generation_guard = changed_config
        with self.assertRaisesRegex(ValueError, 'JSON source changed'):
            with meter.guarded('race', self.args, self.base.code_file, self.base.image,
                               extra_inputs=[self.base.config]):
                self.fail('changed JSON entered body')
        self.assertFalse(self.args.output.exists())
        meter.generation_guard = real_guard
        source = [self.base.image]
        @contextmanager
        def changed_media(*args, **kwargs):
            with real_guard(*args, **kwargs) as run:
                source.append(self.base.image)
                yield run
        meter.generation_guard = changed_media
        with self.assertRaisesRegex(ValueError, 'media declarations changed'):
            with meter.guarded('media-race', self.args, self.base.code_file, self.base.image,
                               media_probe=lambda: list(source)):
                self.fail('changed media entered body')
        self.assertFalse(self.args.output.exists())

    def test_packed_media_must_match_registered_backing_bytes(self):
        image = SimpleNamespace(source='FILE', name='face', filepath=str(self.base.image),
                                packed_file=SimpleNamespace(data=b'other'))
        bpy = SimpleNamespace(data=SimpleNamespace(images=[image]), path=SimpleNamespace(abspath=lambda p: p))
        with self.assertRaisesRegex(ValueError, 'differs from workspace source'):
            meter.blender_media(bpy)

    def test_appended_blender_media_must_be_declared_and_refresh_unpacked(self):
        image = SimpleNamespace(source='FILE', name='face', filepath=str(self.base.image),
                                packed_file=None, reload=mock.Mock())
        bpy = SimpleNamespace(data=SimpleNamespace(images=[image]), path=SimpleNamespace(abspath=lambda p: p))
        run = SimpleNamespace(source_refs={}, staged_inputs={})
        with self.assertRaisesRegex(ValueError, 'not declared before the guard'):
            meter.ensure_registered_blender_media(bpy, run)
        image.reload.assert_not_called()
        run.source_refs[self.base.image.relative_to(self.root).as_posix()] = []
        meter.ensure_registered_blender_media(bpy, run)
        image.reload.assert_called_once()

    def test_nested_staged_face_uses_parent_receipt(self):
        child_code = self.root / 'fixtures/child.py'
        child_code.write_text('fixture child')
        with meter.guarded('parent', self.args, self.base.code_file, self.base.image,
                           extra_inputs=[self.base.config], code_dependencies=[child_code]) as parent:
            self.args.output.mkdir(parents=True)
            staged = self.args.output / 'face.png'
            staged.write_bytes(b'parent generated face')
            with mock.patch.dict(os.environ, parent.child_env(), clear=True):
                with meter.guarded('child', self.args, child_code, staged,
                                   extra_inputs=[self.base.config], inplace=True) as child:
                    self.assertIn(staged.relative_to(self.root).as_posix(), child.staged_inputs)
                    (self.args.output / 'manifest.json').write_text('{}')
            manifest = json.loads((self.args.output / 'manifest.json').read_text())
            self.assertTrue(manifest['production_staged_inputs'])
        manifest = json.loads((self.args.output / 'manifest.json').read_text())
        self.assertTrue(manifest['production_source_refs'])

    def test_nested_staged_face_in_subprocess(self):
        child_code = self.root / 'fixtures/child.py'
        child_code.write_text('fixture child')
        child = '''import functools, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import meter_production_guard as m
from lab.training_common.production_guard import generation_guard
root=Path(sys.argv[2]); m.ROOT=root; m.__file__=str(root/'fixtures/meter_helper.py')
m.generation_guard=functools.partial(generation_guard, root=root)
out=root/'datasets/outputs/scene'; source=out/'face.png'
with m.guarded('child', SimpleNamespace(output=out, count=2), root/'fixtures/child.py', source,
               extra_inputs=[root/'fixtures/production_config.json'], inplace=True) as run:
    assert source.relative_to(root).as_posix() in run.staged_inputs
    (out/'manifest.json').write_text('{}')
'''
        with meter.guarded('parent', self.args, self.base.code_file, self.base.image,
                           extra_inputs=[self.base.config], code_dependencies=[child_code]) as parent:
            self.args.output.mkdir(parents=True)
            (self.args.output / 'face.png').write_bytes(b'parent generated face')
            env = parent.child_env()
            env['PYTHONPATH'] = str(ROOT)
            proc = subprocess.run([sys.executable, '-c', child, str(SCRIPTS), str(self.root)],
                                  env=env, text=True, capture_output=True, timeout=20)
            self.assertEqual(proc.returncode, 0, proc.stderr)
        manifest = json.loads((self.args.output / 'manifest.json').read_text())
        self.assertTrue(manifest['production_staged_inputs'])
        self.assertTrue(manifest['production_source_refs'])


class PreflightEntryTests(unittest.TestCase):
    @staticmethod
    def entry(source, bpy):
        parsed = ast.parse(source.read_text(encoding='utf8'))
        node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        namespace = {'__file__': str(source), 'argparse': __import__('argparse'), 'json': json,
                     'Path': Path, 'sys': sys, 'bpy': bpy, 'TASK_HELPERS': SCRIPTS, 'ROOT': ROOT, 'STATION_SOURCE': ROOT/'fixture.blend'}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        return namespace

    def test_reuse_prints_actual_summary_before_scene_access(self):
        source = ROOT / 'datasets/scripts/meter/render_meter_candidates.py'
        namespace = self.entry(source, SimpleNamespace(data=SimpleNamespace(filepath='source.blend')))
        summary = {'resolution': [640, 480], 'ground_angle_deg': 65.0}
        run = SimpleNamespace(reused=True, value={'kind': 'station', 'summary': summary})
        from contextlib import nullcontext
        with mock.patch.object(sys, 'argv', ['blender', '--', '--output', 'out', '--endpoints', 'x', '--preflight']), \
             mock.patch.object(meter, 'guarded', return_value=nullcontext(run)), \
             mock.patch.object(meter, 'blender_media', return_value=[]), redirect_stdout(io.StringIO()) as output:
            namespace['main']()
        self.assertEqual(output.getvalue().strip(), 'STATION_V4_PREFLIGHT_OK ' + json.dumps(summary))

    def test_preflight_saves_computed_result_and_database_failure_stops_scene(self):
        source = ROOT / 'datasets/scripts/meter/render_meter_candidates.py'
        bpy = SimpleNamespace(data=SimpleNamespace(filepath='source.blend'))
        namespace = self.entry(source, bpy)
        from contextlib import nullcontext
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            root = Path(folder)
            config = root / 'input.json'
            config.write_text('{}')
            output = root / 'runs/meter/new'
            expected = {'resolution': [640, 480], 'ground_angle_deg': 65.0}
            scene = mock.Mock(return_value=expected)
            namespace.update(ROOT=root, render_station_v4=scene)
            run = mock.Mock(reused=False)
            argv = ['blender', '--', '--output', str(output), '--endpoints', str(config), '--station-v4', '--preflight']
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(meter, 'blender_media', return_value=[]), \
                 mock.patch.object(meter, 'guarded', return_value=nullcontext(run)):
                namespace['main']()
            run.set_result.assert_called_once_with({'kind': 'station', 'summary': expected})
            self.assertEqual(scene.call_count, 1)
            scene.reset_mock()
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(meter, 'blender_media', return_value=[]), \
                 mock.patch.object(meter, 'guarded', side_effect=ValueError('database missing')):
                with self.assertRaisesRegex(ValueError, 'database missing'):
                    namespace['main']()
            scene.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
