"""Exercise guarded meter entrypoints without touching workspace datasets."""
import importlib
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


class EntryGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'out'
        self.source = self.root / 'source'
        self.source.mkdir()
        self.video = self.root / 'video.mp4'
        self.face = self.root / 'face.png'
        self.config = self.root / 'geometry.json'
        self.video.write_bytes(b'fixture')
        self.face.write_bytes(b'fixture')
        self.config.write_text('{}', encoding='utf8')

    def cases(self):
        samples = [{'id': 'item', 'split': 'train'}]
        (self.source / 'manifest.json').write_text(json.dumps({'samples': samples}), encoding='utf8')
        (self.source / 'quality_report.json').write_text(
            json.dumps({'samples': [{'id': 'item', 'eligible': True}]}), encoding='utf8')
        animation = self.root / 'animation'
        animation.mkdir()
        (animation / 'manifest.json').write_text(json.dumps({
            'samples': [{'file': f'{i}.png'} for i in range(17)], 'render_resolution': [1024, 1024]
        }), encoding='utf8')
        return [
            ('generate_meter_from_renders', ['--sources', str(self.source), '--output', str(self.out)]),
            ('prepare_meter_rendered_sources', ['--animation', str(animation), '--partial', str(self.source), '--output', str(self.out)]),
            ('prepare_real_meter_validation', ['--video', str(self.video), '--output', str(self.out)]),
            ('apply_meter_glare', [str(self.source)]),
            ('complete_meter_source_endpoints', ['--sources', str(self.source), '--face', str(self.face), '--geometry', str(self.config)]),
        ]

    def test_duplicate_skips_all_writes(self):
        @contextmanager
        def reused(*args, **kwargs):
            yield SimpleNamespace(reused=True, source_refs={})
        for name, argv in self.cases():
            with self.subTest(entry=name):
                module = importlib.import_module(name)
                with mock.patch.object(module, 'generation_guard', reused), \
                     mock.patch.object(module, '_produce') as produce, \
                     mock.patch.object(sys, 'argv', [name, *argv]):
                    module.main()
                produce.assert_not_called()
                self.assertFalse(self.out.exists())

    def test_database_failure_prevents_body_and_output(self):
        @contextmanager
        def unavailable(*args, **kwargs):
            raise ValueError('SQLite cache missing')
            yield  # pragma: no cover
        for name, argv in self.cases():
            with self.subTest(entry=name):
                module = importlib.import_module(name)
                before = (self.source / 'manifest.json').read_bytes()
                with mock.patch.object(module, 'generation_guard', unavailable), \
                     mock.patch.object(module, '_produce') as produce, \
                     mock.patch.object(sys, 'argv', [name, *argv]):
                    with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
                        module.main()
                produce.assert_not_called()
                self.assertFalse(self.out.exists())
                self.assertEqual(before, (self.source / 'manifest.json').read_bytes())

    def test_pre_read_metadata_change_rejects_before_output(self):
        cases = self.cases()
        targets = {
            'generate_meter_from_renders': [self.source / 'manifest.json', self.source / 'quality_report.json'],
            'prepare_meter_rendered_sources': [self.root / 'animation' / 'manifest.json'],
            'apply_meter_glare': [self.source / 'manifest.json'],
            'complete_meter_source_endpoints': [self.source / 'manifest.json'],
        }
        for name, argv in cases:
            for target in targets.get(name, []):
                with self.subTest(entry=name, changed=target.name):
                    module = importlib.import_module(name)

                    @contextmanager
                    def changed(*args, **kwargs):
                        target.write_bytes(target.read_bytes() + b' ')
                        yield SimpleNamespace(reused=False, source_refs={})

                    with mock.patch.object(module, 'generation_guard', changed), \
                         mock.patch.object(module, '_produce') as produce, \
                         mock.patch.object(sys, 'argv', [name, *argv]):
                        with self.assertRaisesRegex(ValueError, 'changed during input enumeration'):
                            module.main()
                    produce.assert_not_called()
                    self.assertFalse(self.out.exists())

    def test_new_partial_label_after_enumeration_is_not_read(self):
        cases = dict(self.cases())
        module = importlib.import_module('prepare_meter_rendered_sources')

        @contextmanager
        def add_label(*args, **kwargs):
            directory = self.source / 'labels' / 'train'
            directory.mkdir(parents=True)
            (directory / 'late.txt').write_text('late', encoding='utf8')
            yield SimpleNamespace(reused=False, source_refs={})

        with mock.patch.object(module, 'generation_guard', add_label), \
             mock.patch.object(module, '_produce') as produce, \
             mock.patch.object(sys, 'argv', ['entry', *cases['prepare_meter_rendered_sources']]):
            module.main()
        self.assertEqual(produce.call_args.args[3], [])


class RealCacheEntryTests(unittest.TestCase):
    def test_video_entry_reuses_and_missing_database_refuses_before_write(self):
        from functools import partial
        workspace = Path(__file__).resolve().parents[5]
        sys.path.insert(0, str(workspace / 'lab/scripts/media/tests'))
        import test_media_index_sqlite as fixtures
        from lab.training_common.production_guard import generation_guard

        fixture = fixtures.CacheTests(methodName='runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        shard = fixture.root / fixture.parts['datasets/raw/camera']['shard']
        item = json.loads(shard.read_text(encoding='utf8'))
        stat = fixture.image.stat()
        item.update(bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
        shard.write_text(json.dumps(item) + '\n', encoding='utf8')
        self.assertEqual(fixture.call(['sync'])[0], 0)
        (fixture.root / 'fixtures').mkdir(exist_ok=True)
        code = fixture.root / 'fixtures' / 'fixture_meter_entry.py'
        code.write_text('fixture', encoding='utf8')
        output = fixture.root / 'datasets' / 'meter_out'
        module = importlib.import_module('prepare_real_meter_validation')

        def produce(args, refs):
            self.assertIn('datasets/raw/camera/a.png', refs)
            args.output.mkdir(parents=True)
            (args.output / 'result.json').write_text('{}', encoding='utf8')

        argv = ['entry', '--video', str(fixture.image), '--output', str(output)]
        with mock.patch.object(module, '__file__', str(code)), \
             mock.patch.object(module, 'generation_guard', partial(generation_guard, root=fixture.root)), \
             mock.patch.object(module, '_produce', side_effect=produce) as body, \
             mock.patch.object(sys, 'argv', argv):
            module.main()
            mtime = (output / 'result.json').stat().st_mtime_ns
            module.main()
            self.assertEqual(body.call_count, 1)
            self.assertEqual((output / 'result.json').stat().st_mtime_ns, mtime)
            db = fixture.root / 'datasets/annotated/media_index/query_cache.sqlite3'
            db.unlink()
            sys.argv = ['entry', '--video', str(fixture.image), '--output', str(fixture.root / 'datasets' / 'other_out')]
            with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
                module.main()
            self.assertEqual(body.call_count, 1)
            self.assertFalse((fixture.root / 'datasets' / 'other_out').exists())

    def test_inplace_endpoint_entry_reuses_its_post_write_state(self):
        from functools import partial
        workspace = Path(__file__).resolve().parents[5]
        sys.path.insert(0, str(workspace / 'lab/scripts/media/tests'))
        import test_media_index_sqlite as fixtures
        from lab.training_common.production_guard import generation_guard

        fixture = fixtures.CacheTests(methodName='runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        shard = fixture.root / fixture.parts['datasets/raw/camera']['shard']
        item = json.loads(shard.read_text(encoding='utf8'))
        stat = fixture.image.stat()
        item.update(bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
        shard.write_text(json.dumps(item) + '\n', encoding='utf8')
        self.assertEqual(fixture.call(['sync'])[0], 0)
        (fixture.root / 'fixtures').mkdir(exist_ok=True)
        code = fixture.root / 'fixtures' / 'fixture_endpoint_entry.py'
        code.write_text('fixture', encoding='utf8')
        geometry = fixture.root / 'fixtures' / 'geometry.json'
        geometry.write_text('{}', encoding='utf8')
        sources = fixture.root / 'datasets' / 'local_sources'
        sources.mkdir()
        manifest = sources / 'manifest.json'
        manifest.write_text('{"samples": []}', encoding='utf8')
        module = importlib.import_module('complete_meter_source_endpoints')

        def produce(args, path, data):
            data['endpoints_completed'] = True
            path.write_text(json.dumps(data), encoding='utf8')

        argv = ['entry', '--sources', str(sources), '--face', str(fixture.image), '--geometry', str(geometry)]
        with mock.patch.object(module, '__file__', str(code)), \
             mock.patch.object(module, 'generation_guard', partial(generation_guard, root=fixture.root)), \
             mock.patch.object(module, '_produce', side_effect=produce) as body, \
             mock.patch.object(sys, 'argv', argv):
            module.main()
            mtime = manifest.stat().st_mtime_ns
            module.main()
            self.assertEqual(body.call_count, 1)
            self.assertEqual(manifest.stat().st_mtime_ns, mtime)


if __name__ == '__main__':
    unittest.main()
