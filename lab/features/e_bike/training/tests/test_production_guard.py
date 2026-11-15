"""Exercise the production gate against a temporary indexed source and SQLite cache."""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import io
import json
import importlib.util
import types
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lab/scripts/media/tests'))
from lab.training_common import production_guard
import test_media_index_sqlite as sqlite_fixtures

SPEC = importlib.util.spec_from_file_location(
    'e_bike_source_entry', ROOT / 'lab/features/e_bike/training/scripts/prepare_e_bike_sources.py')
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)
sys.path.insert(0, str(ROOT / 'lab/features/e_bike/training/scripts'))
import compose_v4_scaled_parking as COMPOSITE
import e_bike_queue_inputs


class ProductionGuardTest(unittest.TestCase):
    def setUp(self):
        self.fixture = sqlite_fixtures.CacheTests('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self._refresh_fixture_stat(self.fixture.image)
        (self.root / 'fixtures').mkdir(exist_ok=True)
        self.code = self.root / 'fixtures' / 'generator.py'
        self.code.write_text('# temporary generator identity\n', encoding='utf8')
        self.output = self.root / 'datasets/exports/e_bike/fixture_output'
        self.arguments = dict(operation='e_bike_fixture', input_paths=[self.fixture.image],
                              output_paths=[self.output], parameters={'fps': 1},
                              code_paths=[self.code], root=self.root)
        patcher = mock.patch.object(production_guard, '_index', return_value=sqlite_fixtures.cache)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_repeat_reuses_recorded_output_without_executing_generator(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(self.fixture.call(['sync'])[0], 0)
        calls = []
        def generate():
            with production_guard.generation_guard(**self.arguments) as run:
                if run.reused:
                    return run
                calls.append('executed')
                self.output.mkdir(parents=True)
                (self.output / 'labels.txt').write_text('sample\n', encoding='utf8')
                return run
        first = generate()
        self.assertFalse(first.reused)
        self.assertEqual(first.source_refs['datasets/raw/camera/a.png'][0]['manifest'],
                         'datasets/exports/derived/manifest.json')
        target = self.output / 'labels.txt'
        before = (target.read_bytes(), target.stat().st_mtime_ns)
        second = generate()
        self.assertTrue(second.reused)
        self.assertEqual(calls, ['executed'])
        self.assertEqual((target.read_bytes(), target.stat().st_mtime_ns), before)

    def _refresh_fixture_stat(self, media):
        shard = self.root / 'datasets/annotated/media_index/shards/raw.jsonl'
        rows = [json.loads(line) for line in shard.read_text(encoding='utf8').splitlines()]
        self.assertEqual(len(rows), 1)
        rows[0]['bytes'] = media.stat().st_size
        rows[0]['modified_ns'] = media.stat().st_mtime_ns
        shard.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')

    def _video_fixture(self):
        video = self.fixture.image.with_suffix('.mp4')
        self.fixture.image.rename(video)
        for shard in (self.root / 'datasets/annotated/media_index/shards').glob('*.jsonl'):
            value = shard.read_text(encoding='utf8').replace('a.png', 'a.mp4')
            shard.write_text(value, encoding='utf8')
        self._refresh_fixture_stat(video)
        return video

    def _entry_guard(self, operation, inputs, outputs, parameters, *, code_paths):
        self.assertEqual(code_paths, [Path(ENTRY.__file__)])
        return production_guard.generation_guard(operation, inputs, outputs, parameters,
                                                 code_paths=[self.code], root=self.root)

    def test_prepare_entry_runs_once_and_reuses_output(self):
        video = self._video_fixture()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(self.fixture.call(['sync'])[0], 0)
        def extract(command, **kwargs):
            destination = Path(command[-1].replace('%06d', '000001'))
            destination.write_bytes(b'fixture frame')
            return types.SimpleNamespace(returncode=0, stderr='')
        with mock.patch.object(ENTRY, 'ROOT', self.root), \
             mock.patch.object(ENTRY, 'generation_guard', side_effect=self._entry_guard), \
             mock.patch.object(ENTRY.subprocess, 'run', side_effect=extract) as ffmpeg:
            first = ENTRY.prepare([video], self.output, 1)
            path = self.output / 'manifest.json'
            before = (path.read_bytes(), path.stat().st_mtime_ns)
            second = ENTRY.prepare([video], self.output, 1)
        self.assertEqual(first, second)
        self.assertEqual(ffmpeg.call_count, 1)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    def test_prepare_entry_rejects_unindexed_source_before_extracting(self):
        self._video_fixture()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(self.fixture.call(['sync'])[0], 0)
        unknown = self.root / 'datasets/raw/camera/unindexed.mp4'
        unknown.write_bytes(b'not in index')
        with mock.patch.object(ENTRY, 'ROOT', self.root), \
             mock.patch.object(ENTRY, 'generation_guard', side_effect=self._entry_guard), \
             mock.patch.object(ENTRY.subprocess, 'run') as ffmpeg:
            with self.assertRaisesRegex(ValueError, 'media missing from current index'):
                ENTRY.prepare([unknown], self.output, 1)
        ffmpeg.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_prepare_entry_rejects_stale_index_before_extracting(self):
        video = self._video_fixture()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(self.fixture.call(['sync'])[0], 0)
        shard = next((self.root / 'datasets/annotated/media_index/shards').glob('*.jsonl'))
        shard.write_text(shard.read_text(encoding='utf8') + '\n', encoding='utf8')
        with mock.patch.object(ENTRY, 'ROOT', self.root), \
             mock.patch.object(ENTRY, 'generation_guard', side_effect=self._entry_guard), \
             mock.patch.object(ENTRY.subprocess, 'run') as ffmpeg:
            with self.assertRaisesRegex(ValueError, 'stale'):
                ENTRY.prepare([video], self.output, 1)
        ffmpeg.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_prepare_entry_rejects_changed_source_before_extracting(self):
        video = self._video_fixture()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(self.fixture.call(['sync'])[0], 0)
        video.write_bytes(b'updated after indexing')
        with mock.patch.object(ENTRY, 'ROOT', self.root), \
             mock.patch.object(ENTRY, 'generation_guard', side_effect=self._entry_guard), \
             mock.patch.object(ENTRY.subprocess, 'run') as ffmpeg:
            with self.assertRaisesRegex(ValueError, 'indexed media changed'):
                ENTRY.prepare([video], self.output, 1)
        ffmpeg.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_prepare_entry_missing_database_never_extracts(self):
        video = self._video_fixture()
        with mock.patch.object(ENTRY, 'ROOT', self.root), \
             mock.patch.object(ENTRY, 'generation_guard', side_effect=self._entry_guard), \
             mock.patch.object(ENTRY.subprocess, 'run') as ffmpeg:
            with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
                ENTRY.prepare([video], self.output, 1)
        ffmpeg.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_composite_entry_stops_on_database_failure_or_reuse(self):
        review = self.root / 'review.json'
        selection = self.root / 'selection.json'
        vehicle_selection = self.root / 'vehicles.json'
        review.write_text(json.dumps({'rows': [{'image': 'datasets/raw/camera/a.png',
                                                 'source_frame': 'datasets/raw/camera/a.png',
                                                 'source_video': 'datasets/raw/camera/a.png'}]}), encoding='utf8')
        for path in (selection, vehicle_selection):
            path.write_text('{}', encoding='utf8')
        output = self.root / 'datasets/exports/e_bike/test_composite'
        arguments = ['compose_v4_scaled_parking.py', '--output', str(output), '--count', '1']
        for state in ('database_failure', 'reused'):
            with self.subTest(state=state), ExitStack() as patches:
                patches.enter_context(mock.patch.object(COMPOSITE, 'ROOT', self.root))
                patches.enter_context(mock.patch.object(COMPOSITE, 'REVIEW', review))
                patches.enter_context(mock.patch.object(COMPOSITE.bike, 'SELECTION', selection))
                patches.enter_context(mock.patch.object(COMPOSITE.bike.scene, 'VEHICLE_SELECTION', vehicle_selection))
                patches.enter_context(mock.patch.object(COMPOSITE.bike, 'sources', return_value=({}, {})))
                patches.enter_context(mock.patch.object(COMPOSITE, 'build_queue', return_value={}))
                patches.enter_context(mock.patch.object(e_bike_queue_inputs, 'queue_inputs', return_value=([], [])))
                patches.enter_context(mock.patch.object(sys, 'argv', arguments))
                produce = patches.enter_context(mock.patch.object(COMPOSITE, '_produce'))
                if state == 'database_failure':
                    guard = patches.enter_context(mock.patch.object(production_guard, 'generation_guard',
                                                                    side_effect=ValueError('SQLite cache missing')))
                    with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
                        COMPOSITE.main()
                else:
                    @contextmanager
                    def reused(*args, **kwargs):
                        yield types.SimpleNamespace(reused=True)
                    guard = patches.enter_context(mock.patch.object(production_guard, 'generation_guard', side_effect=reused))
                    COMPOSITE.main()
                guard.assert_called_once()
                self.assertIn(self.fixture.image.resolve(), guard.call_args.args[1])
                produce.assert_not_called()
                self.assertFalse(output.exists())

    def test_missing_database_fails_before_creating_output(self):
        self.assertFalse((self.root / 'datasets/annotated/media_index/query_cache.sqlite3').exists())
        with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
            with production_guard.generation_guard(**self.arguments):
                self.output.mkdir(parents=True)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
