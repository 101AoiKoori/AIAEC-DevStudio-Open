"""Exercise shared production entry guards against an isolated real SQLite cache."""
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/scripts/media'), str(ROOT / 'lab/scripts/data')]
import cv2
import numpy as np
from lab.scripts.media.tests import test_media_index_sqlite as fixtures
from lab.training_common import bayesian_allocation as allocation
from lab.training_common import camera_tone as tone
from lab.scripts.media import extract_capture_frames as capture
from lab.scripts.data import export_site_adaptation as adaptation
from datasets.scripts.common import working_inputs


class SharedGenerationEntriesTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CacheTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.source = self.fixture.image
        tone.write_image(self.source, np.full((16, 24, 3), (35, 100, 180), np.uint8))
        for module in (allocation, tone, capture, adaptation):
            original = Path(module.__file__)
            target = self.root / original.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            file_patch = mock.patch.object(module, '__file__', str(target))
            root_patch = mock.patch.object(module, 'ROOT', self.root)
            file_patch.start(); root_patch.start()
            self.addCleanup(file_patch.stop); self.addCleanup(root_patch.stop)
        for name in ('lab/scripts/data/v4_source_gate.py', 'lab/training_common/engine/config.py',
                     'lab/training_common/engine/data.py'):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        self.sync_source()

    def sync_source(self):
        shard = self.root / self.fixture.parts['datasets/raw/camera']['shard']
        item = json.loads(shard.read_text(encoding='utf8'))
        stat = self.source.stat()
        item.update(bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
        shard.write_text(json.dumps(item) + '\n', encoding='utf8')
        self.assertEqual(self.fixture.call(['sync'])[0], 0)

    def write(self, path, value):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf8')
        return path

    def tone_plan(self, source):
        states = []
        for profile, values in tone.PROFILES.items():
            states.append({'id': profile, 'failure_event': 'miss', 'min_count': 1, 'max_count': 1,
                           'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
                           'conditions': ['camera_tone:' + profile], 'background': {'source': source},
                           'target': {'kind': 'existing_image_content'}, 'labels': {'status': 'unknown'},
                           'augmentation': {'profile': profile}, 'sources': [source],
                           'parameters': dict(values)})
        return {'schema_version': 1, 'model_key': 'person', 'target_split': 'unassigned',
                'purpose': 'isolated fixture', 'budget': 2, 'uncertainty_weight': 0,
                'independence_basis': 'fixture', 'constraints': {'max_long_side': 1280,
                'max_short_side': 720, 'geometry': 'unchanged'}, 'strata': states}

    def test_bayesian_plan_reuses_and_recomputes_only_changed_request(self):
        request = {'schema_version': 1, 'model_key': 'fire_sticker', 'target_split': 'train',
                   'purpose': 'guard fixture', 'budget': 4, 'uncertainty_weight': .5,
                   'independence_basis': 'fixture groups', 'strata': [{
                       'id': 'normal', 'failure_event': 'miss', 'min_count': 1, 'max_count': 10,
                       'prior': {'alpha': 2, 'beta': 8}, 'exposure_weight': 1, 'failure_cost': 1,
                       'conditions': [], 'observations': []}]}
        input_path = self.write('request.json', request)
        args = ['--input', 'request.json', '--output', 'runs/plan.json']
        self.assertEqual(allocation.main(args), 0)
        result = self.root / 'runs/plan.json'
        original = result.read_bytes(), result.stat().st_mtime_ns
        with mock.patch.object(allocation, 'allocate', side_effect=AssertionError('must reuse')):
            self.assertEqual(allocation.main(args), 0)
        self.assertEqual(original, (result.read_bytes(), result.stat().st_mtime_ns))
        request['budget'] = 5
        input_path.write_text(json.dumps(request), encoding='utf8')
        self.assertEqual(allocation.main(args), 0)
        self.assertEqual(json.loads(result.read_text(encoding='utf8'))['planned_total'], 5)

    def test_camera_preview_reuses_without_processing_or_rewriting(self):
        source = self.source.relative_to(self.root).as_posix()
        out = 'datasets/exports/mixed_scene/preview'
        known = {'exposed_images': [], 'exposed_video_frames': []}
        plan = self.tone_plan(source)
        with mock.patch.object(tone, '_make_preview', wraps=tone._make_preview) as generated:
            first = tone.make_preview(self.root, [source], out, known=known, production_plan=plan)
            completed_run = generated.call_args.kwargs['run']
        with self.assertRaisesRegex(ValueError, 'active planned production run'):
            tone._make_preview(self.root, [source], 'datasets/exports/mixed_scene/late',
                               known=known, run=completed_run)
        self.assertFalse((self.root / 'datasets/exports/mixed_scene/late').exists())
        manifest = self.root / out / 'manifest.json'
        before = manifest.read_bytes(), manifest.stat().st_mtime_ns
        with mock.patch.object(tone, '_make_preview', side_effect=AssertionError('must reuse')):
            second = tone.make_preview(self.root, [source], out, known=known, production_plan=plan)
        self.assertEqual(first, second)
        self.assertEqual(before, (manifest.read_bytes(), manifest.stat().st_mtime_ns))
        with mock.patch.object(tone, '_make_preview', wraps=tone._make_preview) as called:
            tone.make_preview(self.root, [source], out, known=known, production_plan=plan, force_regression=True)
            self.assertEqual(called.call_count, 1)

    def test_camera_preview_rejects_unsupported_states_before_writing(self):
        source = self.source.relative_to(self.root).as_posix()
        out = 'datasets/exports/mixed_scene/refused-plan'
        known = {'exposed_images': [], 'exposed_video_frames': []}
        plan = self.tone_plan(source)
        plan['strata'][0]['augmentation']['yaw'] = 20
        with self.assertRaisesRegex(ValueError, 'Unsupported camera-tone'):
            tone.make_preview(self.root, [source], out, known=known, production_plan=plan)
        self.assertFalse((self.root / out).exists())

    def test_camera_preview_single_output_records_chosen_state(self):
        source = self.source.relative_to(self.root).as_posix()
        out = 'datasets/exports/mixed_scene/one'
        plan = self.tone_plan(source)
        plan['budget'] = 1
        for state in plan['strata']:
            state['min_count'] = 0
        report = tone.make_preview(self.root, [source], out, production_plan=plan,
                                   known={'exposed_images': [], 'exposed_video_frames': []})
        self.assertEqual(len(report['samples']), 1)
        state_db = self.root / 'datasets/annotated/media_index/production_state.sqlite3'
        with sqlite3.connect(str(state_db)) as db:
            receipt = json.loads(db.execute('SELECT receipt_json FROM production_operations WHERE operation=?',
                                            ('camera_tone:' + out,)).fetchone()[0])
        recorded = receipt['planned_samples']
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]['actual']['parent_image'], source)
        self.assertEqual(recorded[0]['actual']['record_index'], 0)
        self.assertEqual(recorded[0]['actual']['manifest'], out + '/manifest.json')

    def test_allocation_interruption_is_not_a_complete_receipt(self):
        self.write('request.json', {'strata': []})
        with mock.patch.object(allocation, 'allocate', side_effect=KeyboardInterrupt):
            self.assertEqual(allocation.main(['--input', 'request.json', '--output', 'runs/interrupted.json']), 130)
        result = self.root / 'runs/interrupted.json'
        self.assertEqual(json.loads(result.read_text(encoding='utf8'))['status'], 'interrupted')
        with mock.patch.object(allocation, 'allocate', side_effect=AssertionError('must reject unreceipted output')):
            self.assertEqual(allocation.main(['--input', 'request.json', '--output', 'runs/interrupted.json']), 1)

    def test_extraction_entry_reuses_before_frame_work(self):
        plan = self.write('datasets/annotated/segments.json', {'interval': 1})
        segment = types.SimpleNamespace(video=types.SimpleNamespace(path=self.source), source_row={'id': 'fixture'})
        out = self.root / 'datasets/annotated/extraction'
        def write_frames(*args):
            out.mkdir(parents=True)
            (out / 'summary.json').write_text('{"frames_indexed":1}', encoding='utf8')
            return 1
        with mock.patch.object(capture, '_extract', side_effect=write_frames) as work:
            self.assertEqual(capture.extract('fixture_ffmpeg', self.source.parent, plan, out, [segment]), 1)
            before = (out / 'summary.json').stat().st_mtime_ns
            self.assertEqual(capture.extract('fixture_ffmpeg', self.source.parent, plan, out, [segment]), 1)
            self.assertEqual(work.call_count, 1)
            self.assertEqual(before, (out / 'summary.json').stat().st_mtime_ns)

    def test_extraction_cli_accepts_archived_capture_with_annotated_output(self):
        batch = self.root / 'Archive/Assets/captures/site_capture_20261112'
        batch.mkdir(parents=True)
        plan = self.write('datasets/annotated/segments.csv', {})
        out = self.root / 'datasets/annotated/extraction'
        argv = ['extract_capture_frames.py', '--raw-batch', str(batch),
                '--segments', str(plan), '--output', str(out)]
        segments = [object()]
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch.object(capture.shutil, 'which', side_effect=lambda name: name), \
                mock.patch.object(capture, 'read_segments', return_value=segments) as reader, \
                mock.patch.object(capture, 'extract', return_value=1) as extract:
            capture.main()
        reader.assert_called_once_with(plan, batch, 'ffprobe')
        extract.assert_called_once_with('ffmpeg', batch, plan, out, segments)
        with self.assertRaises(ValueError):
            capture.path_under(batch, 'datasets/annotated')
        with self.assertRaises(ValueError):
            capture.path_under(self.root / 'Archive/technical/old_capture', 'datasets/raw', 'Archive/Assets')

    def test_site_export_entry_reuses_before_copying(self):
        self.write('config.json', {})
        self.write('data.yaml', {})
        plan = {'config': 'config.json', 'output': 'datasets/exports/fixture',
                'samples': [{'image': self.source.relative_to(self.root).as_posix()}]}
        out = self.root / plan['output']
        def write_export(*args):
            out.mkdir(parents=True)
            (out / 'summary.json').write_text('{"samples_added":1}', encoding='utf8')
        with mock.patch.object(adaptation, 'load_config', return_value={'data': 'data.yaml'}), \
                mock.patch.object(adaptation, 'resolve_sources', return_value=[]), \
                mock.patch.object(adaptation, '_export', side_effect=write_export) as work:
            adaptation.export(plan, set())
            before = (out / 'summary.json').stat().st_mtime_ns
            adaptation.export(plan, set())
            self.assertEqual(work.call_count, 1)
            self.assertEqual(before, (out / 'summary.json').stat().st_mtime_ns)

    def test_dual_role_source_uses_two_workspace_copies_across_splits(self):
        source = self.source.relative_to(self.root).as_posix()
        target = 'datasets/raw/person/targets/camera/batch/frames/frame.jpg'
        background = 'datasets/raw/shared/backgrounds/camera/batch/frame.jpg'
        for image in (target, background):
            copy = self.root / image
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.source, copy)
        target_manifest = 'datasets/annotated/person/targets.json'
        background_manifest = 'datasets/annotated/common/backgrounds.json'
        self.write(target_manifest, {'schema_version': 1, 'records': [
            {'source_image': source, 'image': target},
        ]})
        self.write(background_manifest, {'schema_version': 1, 'records': [
            {'source_image': source, 'image': background, 'tasks': {'person': {
                'state': 'ready', 'uses': ['composite_background'], 'split': 'train',
            }}},
        ]})
        config = {'target_working_manifest': target_manifest, 'background_pool_manifest': background_manifest}
        with mock.patch.object(working_inputs, 'WORKSPACE', self.root), \
             mock.patch.object(working_inputs, 'load_task_config', return_value=(None, config)):
            for split in ('train', 'val', 'test'):
                with self.subTest(split=split):
                    lookup = working_inputs.WorkingInputs(
                        'person', backgrounds=[source], targets=[source], split=split,
                    )
                    self.assertEqual(lookup.actual(source, role='target'), target)
                    self.assertEqual(lookup.actual(source, role='background'), background)
                    self.assertEqual(lookup.actual(source), source)
                    self.assertEqual(lookup.media, {self.root / target, self.root / background})

    def test_stale_database_rejects_before_any_output(self):
        shard = self.root / self.fixture.parts['datasets/raw/camera']['shard']
        shard.write_text(shard.read_text(encoding='utf8') + ' ', encoding='utf8')
        source = self.source.relative_to(self.root).as_posix()
        with mock.patch.object(tone, '_make_preview', side_effect=AssertionError('must not generate')):
            with self.assertRaisesRegex(ValueError, 'stale'):
                tone.make_preview(self.root, [source], 'datasets/exports/mixed_scene/refused',
                                  known={'exposed_images': [], 'exposed_video_frames': []},
                                  production_plan=self.tone_plan(source))
        self.assertFalse((self.root / 'datasets/exports/mixed_scene/refused').exists())


if __name__ == '__main__':
    unittest.main()
