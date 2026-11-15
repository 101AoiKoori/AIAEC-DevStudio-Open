"""Focused checks for provenance claims, relocation, and builder coordination."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import media_index


class MediaIndexTests(unittest.TestCase):
    def setUp(self):
        workspace_tmp = media_index.ROOT / 'tmp'
        workspace_tmp.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=workspace_tmp)
        self.root = Path(self.temp.name)
        self.old_root, self.old_index = media_index.ROOT, media_index.INDEX
        policy = self.old_root / media_index.LIFECYCLE_POLICY
        media_index.ROOT = self.root
        media_index.INDEX = self.root / 'datasets/annotated/media_index'
        copied_policy = self.root / media_index.LIFECYCLE_POLICY
        copied_policy.parent.mkdir(parents=True)
        copied_policy.write_text(policy.read_text(encoding='utf8'), encoding='utf8')
        sources = {
            'datasets/annotated/person/media_lifecycle_20261113.json': {'records': []},
            'datasets/annotated/waste_bin_state/v4_distribution_resume_20261113.json': {'dataset_lifecycle': {'items': []}},
            'datasets/exports/fire/fire_sticker_v4_candidate_entry_20261113/media_lifecycle_20261113.json': {'retired': []},
        }
        for name, data in sources.items():
            source = self.root / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(json.dumps(data), encoding='utf8')
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(setattr, media_index, 'ROOT', self.old_root)
        self.addCleanup(setattr, media_index, 'INDEX', self.old_index)

    def task_config_fixture(self):
        config = self.root / 'datasets/scripts/person/config'
        config.mkdir(parents=True)
        source = {'task': 'person', 'production_plan': 'datasets/scripts/person/config/production_plan.json',
                  'render_plan': 'datasets/scripts/person/config/render_plan.json',
                  'combination_plan': 'datasets/scripts/person/config/retired_plan.json',
                  'current_selection': {'excluded': [{'path': 'datasets/raw/person/unlabelled.png'}]}}
        (config / 'source.yaml').write_text(json.dumps(source), encoding='utf8')
        (config / 'dataset.yaml').write_text(json.dumps({'task': 'person', 'source_config': 'config/source.yaml'}), encoding='utf8')
        plan = {'budget': 1, 'model_key': 'person', 'target_split': 'train', 'strata': [
            {'id': 'first', 'sources': ['symbolic_background'],
             'background': {'candidates': [{'image': 'datasets/raw/person/background.png',
                                            'parent_source': 'datasets/raw/person/source.mp4'}]},
             'parameters': {'note': 'datasets/raw/person/background.png#record:0'},
             'target': {'asset': 'datasets/assets/blender/person/scene.blend'}}]}
        (config / 'render_plan.json').write_text(json.dumps(plan), encoding='utf8')
        (config / 'retired_plan.json').write_text('{"strata": []}', encoding='utf8')
        return config, plan

    def test_task_config_records_selection_pending_plans_and_nested_reverse_paths(self):
        config, plan = self.task_config_fixture()
        records = media_index.task_config_evidence(config / 'source.yaml')
        snapshot = records[0][1]['claims']['task_config']
        self.assertEqual(snapshot['schema_version'], 1)
        self.assertEqual(snapshot['source_config']['current_selection']['excluded'][0]['path'],
                         'datasets/raw/person/unlabelled.png')
        self.assertEqual([p['plan_key'] for p in snapshot['plans']], ['production_plan', 'render_plan'])
        self.assertFalse(snapshot['plans'][0]['present'])
        self.assertEqual(snapshot['plans'][1]['budget'], 1)
        self.assertEqual(len(snapshot['declared_files']), 4)
        self.assertTrue(all({'path', 'present', 'size', 'mtime_ns'} == set(p) for p in snapshot['declared_files']))
        references = [(path, record['reference_field']) for path, record in records
                      if record['relation'] == 'plan_input_reference']
        self.assertIn(('datasets/raw/person/background.png', '/background/candidates/0/image'), references)
        self.assertIn(('datasets/assets/blender/person/scene.blend', '/target/asset'), references)
        self.assertEqual(len(references), 3)
        self.assertFalse(any(path.endswith('unlabelled.png') for path, _ in records))
        states = [record['claims']['plan_state']['stratum'] for _, record in records
                  if record['relation'] == 'plan_state']
        self.assertEqual(states, plan['strata'])

    def test_task_plan_modification_and_missing_plan_creation_invalidate_local_snapshot(self):
        config, plan = self.task_config_fixture()
        task = config.parent
        snapshot = self.root / 'snapshot.jsonl'
        snapshot.write_text(''.join(media_index.input_lines(task, False, None, [])), encoding='utf8')
        self.assertTrue(media_index.input_matches(snapshot, task, False, None, []))
        plan['budget'] = 2
        (config / 'render_plan.json').write_text(json.dumps(plan), encoding='utf8')
        self.assertFalse(media_index.input_matches(snapshot, task, False, None, []))
        snapshot.write_text(''.join(media_index.input_lines(task, False, None, [])), encoding='utf8')
        (config / 'production_plan.json').write_text(json.dumps(plan), encoding='utf8')
        self.assertFalse(media_index.input_matches(snapshot, task, False, None, []))
        snapshot.write_text(''.join(media_index.input_lines(task, False, None, [])), encoding='utf8')
        source = config / 'source.yaml'
        source.write_text(source.read_text(encoding='utf8') + '\n', encoding='utf8')
        self.assertFalse(media_index.input_matches(snapshot, task, False, None, []))

    def test_task_config_rejects_cross_task_plan_and_invalid_yaml(self):
        config, _ = self.task_config_fixture()
        source = config / 'source.yaml'
        source.write_text(json.dumps({'task': 'person', 'production_plan': 'datasets/scripts/meter/config/production_plan.json'}), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'own config directory'):
            media_index.task_config_evidence(source)
        source.write_text(json.dumps({'task': 'meter'}), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'their own task'):
            media_index.task_config_evidence(source)
        source.write_text('selected: [broken', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'invalid task YAML'):
            media_index.task_config_evidence(source)

    def test_task_config_targeted_build_keeps_other_partition_and_has_no_media(self):
        config, _ = self.task_config_fixture()
        other = self.root / 'datasets/scripts/meter'
        other.mkdir(parents=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['build', '--scan-root', 'datasets/scripts',
                                             '--source', 'datasets/scripts/meter', '--no-video-probe', '--no-clean']), 0)
        old = json.loads((media_index.INDEX / 'catalog.json').read_text(encoding='utf8'))['partitions']['datasets/scripts/meter']
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['build', '--scan-root', 'datasets/scripts',
                                             '--source', 'datasets/scripts/person', '--no-video-probe', '--no-clean']), 0)
        catalog = json.loads((media_index.INDEX / 'catalog.json').read_text(encoding='utf8'))
        self.assertEqual(old, catalog['partitions']['datasets/scripts/meter'])
        part = catalog['partitions']['datasets/scripts/person']
        self.assertEqual((self.root / part['shard']).read_text(encoding='utf8'), '')
        records = [json.loads(line) for line in (self.root / part['evidence_shard']).read_text(encoding='utf8').splitlines()]
        self.assertTrue(any(row['evidence'][0]['relation'] == 'task_config' for row in records))

    def test_manifest_evidence_and_explicit_move(self):
        batch = self.root / 'datasets/exports/waste_bin/batch'
        batch.mkdir(parents=True)
        # PNG header provides dimensions without requiring an image library.
        (batch / 'a.png').write_bytes(b'\x89PNG\r\n\x1a\n' + b'\0' * 8 +
                                      (640).to_bytes(4, 'big') + (480).to_bytes(4, 'big'))
        (batch / 'manifest.json').write_text(json.dumps({'records': [{
            'image': 'a.png', 'parent_video': 'datasets/raw/clip.mp4',
            'split': 'train', 'label_state': 'partial',
            'instances': [{'task': 'person', 'bbox_xyxy': [1, 2, 10, 20]}],
            'rejected_instances': [{'task': 'person', 'reason': 'incorrect_location'}],
            'background_view': 'road_perspective', 'surface_role': 'ground_plane',
            'camera_angles': {'yaw': None, 'pitch': None, 'roll': None},
            'use_status': 'qualitative_query_hint_not_calibrated_geometry',
            'ground_state': 'unknown', 'visible_ground_boundary_status': 'unknown',
            'instances_scope': 'nonexhaustive'}]}), encoding='utf8')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = media_index.main(['build', '--scan-root', 'datasets/exports',
                                     '--source', 'datasets/exports/waste_bin',
                                     '--relocation', 'datasets/exports/old=datasets/exports/waste_bin',
                                     '--no-video-probe'])
        self.assertEqual(code, 0)
        with contextlib.redirect_stdout(output := io.StringIO()):
            self.assertEqual(media_index.main(['query', '--contains', 'a.png']), 0)
        record = json.loads(output.getvalue())
        self.assertEqual(record['image_size'], [640, 480])
        self.assertEqual(record['id'], 'workspace:datasets/exports/old/batch/a.png')
        self.assertEqual(record['path_history'][0]['basis'], 'explicit_relocation')
        self.assertEqual(record['evidence'][0]['claims']['label_state'], 'partial')
        self.assertEqual(record['evidence'][0]['claims']['instances'][0]['bbox_xyxy'], [1, 2, 10, 20])
        self.assertEqual(record['evidence'][0]['claims']['instances_scope'], 'nonexhaustive')
        self.assertEqual(record['evidence'][0]['claims']['rejected_instances'][0]['reason'], 'incorrect_location')
        self.assertEqual(record['evidence'][0]['claims']['surface_role'], 'ground_plane')
        self.assertIsNone(record['evidence'][0]['claims']['camera_angles']['yaw'])
        self.assertEqual(record['evidence'][0]['claims']['visible_ground_boundary_status'], 'unknown')
        self.assertEqual(record['evidence'][0]['claims']['parent_video'], 'datasets/raw/clip.mp4')
        self.assertEqual(record['recorded_claims']['sources_and_parents'], ['datasets/raw/clip.mp4'])
        self.assertEqual(record['recorded_claims']['evaluation_scope'], [])
        # A later incremental refresh retains the explicit path relationship.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['build', '--scan-root', 'datasets/exports',
                                               '--source', 'datasets/exports/waste_bin',
                                               '--no-video-probe']), 0)
            with contextlib.redirect_stdout(output := io.StringIO()):
                media_index.main(['query', '--contains', 'a.png'])
        self.assertEqual(json.loads(output.getvalue())['id'], record['id'])

    def test_concurrent_builder_is_rejected(self):
        media_index.INDEX.mkdir(parents=True)
        (media_index.INDEX / '.build.lock').write_text('busy', encoding='utf8')
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(media_index.main(['build', '--no-video-probe']), 1)
        self.assertTrue((media_index.INDEX / '.build.lock').exists())

    def test_unlock_requires_confirmed_same_host_dead_process(self):
        media_index.INDEX.mkdir(parents=True)
        lock = media_index.INDEX / '.build.lock'
        lock.write_text(json.dumps({'pid': 100, 'host': media_index.socket.gethostname()}), encoding='utf8')
        with patch.object(media_index, 'process_exists', return_value=None), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(media_index.main(['unlock']), 1)
        self.assertTrue(lock.exists())
        with patch.object(media_index, 'process_exists', return_value=False), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['unlock']), 0)
        self.assertFalse(lock.exists())

    def test_manifest_can_reference_a_different_partition(self):
        media = self.root / 'datasets/raw/clip.png'
        media.parent.mkdir(parents=True)
        media.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\0' * 8 +
                          (12).to_bytes(4, 'big') + (8).to_bytes(4, 'big'))
        manifest = self.root / 'datasets/exports/batch/manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'items': [{'path': 'datasets/raw/clip.png',
                                                  'source_group': 'known-video',
                                                  'generation_method': 'camera_capture'}]}), encoding='utf8')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['build', '--scan-root', 'datasets',
                                               '--no-video-probe']), 0)
        with contextlib.redirect_stdout(output := io.StringIO()):
            self.assertEqual(media_index.main(['evidence', 'datasets/raw/clip.png']), 0)
        evidence = json.loads(output.getvalue())
        self.assertEqual(evidence['evidence'][0]['manifest'], 'datasets/exports/batch/manifest.json')
        self.assertEqual(evidence['evidence'][0]['record_index'], 0)
        with contextlib.redirect_stdout(output := io.StringIO()):
            self.assertEqual(media_index.main(['evidence', '--source-group', 'known-video',
                                               '--generation-method', 'camera_capture']), 0)
        self.assertEqual(json.loads(output.getvalue())['claimed_path'], 'datasets/raw/clip.png')
        with contextlib.redirect_stdout(output := io.StringIO()):
            self.assertEqual(media_index.main(['query', '--source-group', 'known-video']), 0)
        self.assertEqual(json.loads(output.getvalue())['path'], 'datasets/raw/clip.png')

    def test_failed_new_snapshot_leaves_previous_catalog(self):
        batch = self.root / 'datasets/exports/task'
        batch.mkdir(parents=True)
        (batch / 'a.png').write_bytes(b'\x89PNG\r\n\x1a\n' + b'\0' * 8 +
                                      (32).to_bytes(4, 'big') + (24).to_bytes(4, 'big'))
        argv = ['build', '--scan-root', 'datasets/exports', '--source',
                'datasets/exports/task', '--no-video-probe']
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(argv), 0)
        catalog_path = media_index.INDEX / 'catalog.json'
        before = json.loads(catalog_path.read_text(encoding='utf8'))['partitions']['datasets/exports/task/@root']
        replace = media_index.os.replace
        def failing_replace(src, dst):
            if str(dst).endswith('.jsonl') and not str(dst).endswith('.evidence.jsonl'):
                raise OSError('simulated second shard failure')
            return replace(src, dst)
        with patch.object(media_index.os, 'replace', side_effect=failing_replace), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(media_index.main(argv + ['--rebuild']), 1)
        after = json.loads(catalog_path.read_text(encoding='utf8'))['partitions']['datasets/exports/task/@root']
        self.assertEqual(after['shard'], before['shard'])
        self.assertEqual(after['evidence_shard'], before['evidence_shard'])
        self.assertTrue((self.root / before['shard']).is_file())
        self.assertTrue((self.root / before['evidence_shard']).is_file())

    def test_unchanged_reuses_partitions_and_changes_are_local(self):
        first = self.root / 'datasets/exports/task/first'
        second = self.root / 'datasets/exports/task/second'
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        (first / 'one.png').write_bytes(b'one')
        (second / 'two.png').write_bytes(b'two')
        args = ['build', '--scan-root', 'datasets/exports', '--no-video-probe']
        def build(extra=()):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(media_index.main(args + list(extra)), 0)
            return json.loads((media_index.INDEX / 'catalog.json').read_text(encoding='utf8'))['partitions']
        before = build()
        a, b = 'datasets/exports/task/first', 'datasets/exports/task/second'
        old_files = {key: (before[key]['shard'], before[key]['evidence_shard'],
                           before[key]['input_shard'], before[key]['updated_utc']) for key in (a, b)}
        catalog_file = media_index.INDEX / 'catalog.json'
        catalog_before = (catalog_file.stat().st_mtime_ns, catalog_file.read_bytes())
        self.assertNotIn('last_full_build', json.loads(catalog_before[1]))
        with patch.object(media_index, 'build_partition', side_effect=AssertionError('unexpected rebuild')):
            same = build()
        self.assertEqual((catalog_file.stat().st_mtime_ns, catalog_file.read_bytes()), catalog_before)
        self.assertNotIn('last_full_build', json.loads(catalog_file.read_text(encoding='utf8')))
        with patch.object(media_index, 'build_partition', side_effect=AssertionError('unexpected rebuild')):
            build(['--source', a])
        self.assertEqual((catalog_file.stat().st_mtime_ns, catalog_file.read_bytes()), catalog_before)
        for key in (a, b):
            self.assertEqual(old_files[key], tuple(same[key][field] for field in
                              ('shard', 'evidence_shard', 'input_shard', 'updated_utc')))
        (first / 'one.png').write_bytes(b'changed')
        changed = build(['--source', a])
        self.assertNotEqual(changed[a]['shard'], before[a]['shard'])
        self.assertEqual(changed[b], same[b])
        (first / 'one.png').unlink()
        (first / 'new.png').write_bytes(b'new')
        updated = build()
        self.assertNotEqual(updated[a]['shard'], changed[a]['shard'])
        self.assertEqual(updated[b], changed[b])
        with (self.root / updated[a]['shard']).open(encoding='utf8') as f:
            self.assertEqual([Path(json.loads(line)['path']).name for line in f], ['new.png'])
        forced = build(['--rebuild'])
        self.assertNotEqual(forced[b]['shard'], updated[b]['shard'])

    def test_external_relative_path_changes_refresh_manifest_partition(self):
        first = self.root / 'datasets/exports/task/first'
        first.mkdir(parents=True)
        (first / 'manifest.json').write_text(json.dumps({'records': [{'image': 'sample.png', 'split': 'train'}]}), encoding='utf8')
        args = ['build', '--scan-root', 'datasets/exports', '--no-video-probe']
        def build():
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(media_index.main(args), 0)
            return json.loads((media_index.INDEX / 'catalog.json').read_text(encoding='utf8'))['partitions']['datasets/exports/task/first']
        before = build()
        (self.root / 'sample.png').write_bytes(b'external')
        after = build()
        self.assertNotEqual(after['shard'], before['shard'])
        with (self.root / after['evidence_shard']).open(encoding='utf8') as f:
            self.assertTrue(any(json.loads(line)['claimed_path'] == 'sample.png' for line in f))

    def test_targeted_unchanged_preserves_complete_full_snapshot(self):
        batch = self.root / 'datasets/exports/task/batch'
        batch.mkdir(parents=True)
        (batch/'image.png').write_bytes(b'fixture')
        with patch.object(media_index, 'DEFAULT_SOURCES', ('datasets/exports',)), \
                patch.object(media_index, 'ROOT_FILE_SOURCES', ()), \
                patch.object(media_index, 'PROJECT_ASSET_DIRS', ()), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['build', '--no-video-probe']), 0)
            path = media_index.INDEX/'catalog.json'
            before = (path.stat().st_mtime_ns, path.read_bytes())
            self.assertEqual(media_index.main(['build', '--source', 'datasets/exports/task/batch',
                                               '--no-video-probe']), 0)
            self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), before)
            data = json.loads(before[1])
            self.assertEqual(data['last_full_build']['state'], 'complete')
            self.assertEqual(data['latest_build_scope']['selected_sources'], [])

    def test_overlap_is_deduplicated_and_move_file_is_read(self):
        batch = self.root / 'datasets/exports/waste_bin/batch'
        batch.mkdir(parents=True)
        (batch / 'one.png').write_bytes(b'\x89PNG\r\n\x1a\n' + b'\0' * 8 +
                                        (8).to_bytes(4, 'big') + (8).to_bytes(4, 'big'))
        move_file = self.root / 'datasets/migration.json'
        move_file.write_text(json.dumps({'moves': [{'original_path': 'datasets/exports/old',
                                                     'current_path': 'datasets/exports/waste_bin/batch'}]}),
                             encoding='utf8')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['build', '--scan-root', 'datasets/exports',
                                               '--relocation-file', 'datasets/migration.json',
                                               '--no-video-probe']), 0)
            self.assertEqual(media_index.main(['build', '--scan-root', 'datasets/exports/waste_bin',
                                               '--no-video-probe']), 0)
        with contextlib.redirect_stdout(output := io.StringIO()):
            self.assertEqual(media_index.main(['query', '--contains', 'one.png']), 0)
        rows = [json.loads(x) for x in output.getvalue().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['id'], 'workspace:datasets/exports/old/one.png')

    def test_declared_retained_copy_preserves_original_source(self):
        batch = self.root / 'datasets/annotated/meter/base'
        batch.mkdir(parents=True)
        (batch / 'copy.png').write_bytes(b'\x89PNG\r\n\x1a\n' + b'\0' * 8 +
                                         (8).to_bytes(4, 'big') + (8).to_bytes(4, 'big'))
        (batch / 'manifest.json').write_text(json.dumps({'samples': [{
            'source': 'datasets/annotated/meter/base/copy.png',
            'original_source': 'datasets/exports/meter/old/frame.png',
            'source_relocation_basis': 'byte_identical_retained_copy'}]}), encoding='utf8')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['build', '--scan-root', 'datasets/annotated',
                                               '--no-video-probe']), 0)
        with contextlib.redirect_stdout(output := io.StringIO()):
            media_index.main(['query', '--contains', 'copy.png', '--with-evidence'])
        record = json.loads(output.getvalue())
        self.assertEqual(record['evidence'][0]['claims']['source_relocation_basis'],
                         'byte_identical_retained_copy')
        self.assertEqual(record['recorded_claims']['sources_and_parents'],
                         ['datasets/exports/meter/old/frame.png'])

    def test_old_partition_retired_only_after_all_batches_scanned(self):
        task = self.root / 'datasets/exports/waste_bin'
        (task / 'batch_a').mkdir(parents=True)
        (task / 'batch_b').mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            media_index.main(['build', '--scan-root', 'datasets/exports', '--no-video-probe'])
        catalog_path = media_index.INDEX / 'catalog.json'
        catalog = json.loads(catalog_path.read_text(encoding='utf8'))
        catalog['partitions']['datasets/exports/waste_bin'] = {
            'source': 'datasets/exports/waste_bin', 'updated_utc': '2000-01-01',
            'shard': catalog['partitions']['datasets/exports/waste_bin/batch_a']['shard']}
        missing = catalog['partitions'].pop('datasets/exports/waste_bin/batch_b')
        catalog_path.write_text(json.dumps(catalog), encoding='utf8')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['retire-overlaps']), 0)
        catalog = json.loads(catalog_path.read_text(encoding='utf8'))
        self.assertIn('datasets/exports/waste_bin', catalog['partitions'])
        catalog['partitions']['datasets/exports/waste_bin/batch_b'] = missing
        catalog_path.write_text(json.dumps(catalog), encoding='utf8')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['retire-overlaps']), 0)
        catalog = json.loads(catalog_path.read_text(encoding='utf8'))
        self.assertNotIn('datasets/exports/waste_bin', catalog['partitions'])

    def test_lifecycle_filters_old_snapshot_and_keeps_project_array_assets(self):
        parent = self.root / 'datasets/exports/person'
        retired = parent / 'bad'
        live = parent / 'pending'
        retired.mkdir(parents=True)
        live.mkdir()
        png = b'\x89PNG\r\n\x1a\n' + b'\0' * 8 + (8).to_bytes(4, 'big') * 2
        (retired / 'retired.png').write_bytes(png)
        (live / 'pending.png').write_bytes(png)
        with contextlib.redirect_stdout(io.StringIO()):
            media_index.main(['build', '--scan-root', 'datasets', '--no-video-probe'])
        lifecycle = self.root / 'datasets/annotated/person/media_lifecycle_20261113.json'
        lifecycle.parent.mkdir(parents=True, exist_ok=True)
        lifecycle.write_text(json.dumps({'records': [
            {'path': 'datasets/exports/person/bad', 'lifecycle_state': 'inactive_failed', 'evidence': []},
            {'path': 'datasets/exports/person/pending', 'lifecycle_state': 'quarantined_pending_policy', 'evidence': []}]}), encoding='utf8')
        (live / 'manifest.json').write_text(json.dumps({'records': [{
            'image': 'datasets/exports/person/bad/retired.png', 'source_group': 'retired'},
            {'image': 'datasets/exports/person/pending/pending.png', 'source_group': 'active'}]}), encoding='utf8')
        assets = self.root / 'datasets/exports/waste_bin/.cache/assets'
        assets.mkdir(parents=True)
        (assets / 'icon.npz').write_bytes(b'array cache')
        (assets / 'assets.json').write_text(json.dumps([{'file': 'icon.npz', 'name': 'icon', 'w': 8, 'h': 8}]), encoding='utf8')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['build', '--no-video-probe']), 0)
        with contextlib.redirect_stdout(output := io.StringIO()):
            media_index.main(['query', '--contains', 'retired.png', '--include-missing'])
        self.assertEqual(output.getvalue(), '')
        with contextlib.redirect_stdout(output := io.StringIO()):
            media_index.main(['query', '--contains', 'pending.png'])
        self.assertEqual(json.loads(output.getvalue())['path'], 'datasets/exports/person/pending/pending.png')
        with contextlib.redirect_stdout(output := io.StringIO()):
            media_index.main(['query', '--kind', 'array_asset'])
        self.assertEqual(json.loads(output.getvalue())['evidence'][0]['claims']['w'], 8)
        catalog = json.loads((media_index.INDEX / 'catalog.json').read_text(encoding='utf8'))
        for partition in catalog['partitions'].values():
            self.assertNotIn('retired.png', (self.root / partition['shard']).read_text(encoding='utf8'))
            if partition.get('evidence_shard'):
                self.assertNotIn('retired.png', (self.root / partition['evidence_shard']).read_text(encoding='utf8'))

    def test_catalog_publisher_waits_for_active_reader(self):
        media_index.INDEX.mkdir(parents=True)
        catalog_path = media_index.INDEX / 'catalog.json'
        media_index.atomic_json(catalog_path, {'version': 'old'})
        reader_ready, release_reader, writer_done = threading.Event(), threading.Event(), threading.Event()
        def read_snapshot():
            with media_index.catalog_guard():
                reader_ready.set()
                release_reader.wait(3)
        def publish():
            media_index.atomic_json(catalog_path, {'version': 'new'})
            writer_done.set()
        reader = threading.Thread(target=read_snapshot)
        writer = threading.Thread(target=publish)
        reader.start()
        self.assertTrue(reader_ready.wait(3))
        writer.start()
        self.assertFalse(writer_done.wait(.2))
        self.assertEqual(json.loads(catalog_path.read_text(encoding='utf8'))['version'], 'old')
        release_reader.set()
        reader.join(3)
        writer.join(3)
        self.assertTrue(writer_done.is_set())
        self.assertEqual(json.loads(catalog_path.read_text(encoding='utf8'))['version'], 'new')

    def test_parent_usage_is_queryable_without_relabeling_parent(self):
        parent = self.root / 'datasets/raw/parent.png'
        derived = self.root / 'datasets/exports/task/batch/derived.png'
        parent.parent.mkdir(parents=True, exist_ok=True)
        derived.parent.mkdir(parents=True, exist_ok=True)
        for path in (parent, derived):
            path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\0' * 8 + (8).to_bytes(4, 'big') * 2)
        (derived.parent / 'manifest.json').write_text(json.dumps({'records': [{
            'image': 'datasets/exports/task/batch/derived.png',
            'background_parent': 'datasets/raw/parent.png', 'generation_method': 'composite',
            'source_group': 'source-A', 'split': 'train',
            'source': 'datasets/raw/parent.png',
            'equivalent_upstream_image': 'datasets/raw/parent.png',
            'equivalent_upstream_basis': 'byte_identical'}]}), encoding='utf8')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['build', '--scan-root', 'datasets', '--no-video-probe']), 0)
        with contextlib.redirect_stdout(output := io.StringIO()):
            media_index.main(['query', '--contains', 'datasets/raw/parent.png', '--with-evidence'])
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        record = next(x for x in records if x['path'] == 'datasets/raw/parent.png')
        self.assertEqual(record['recorded_claims']['generation_methods'], [])
        self.assertEqual(record['recorded_claims']['declared_splits'], [])
        self.assertEqual(record['recorded_claims']['v4_declared_splits'], [])
        self.assertEqual(record['evidence'][0]['relation'], 'referenced_as')
        self.assertEqual(record['evidence'][0]['reference_field'], 'background_parent')
        self.assertEqual(record['evidence'][0]['claims']['generation_method'], 'composite')
        self.assertTrue(any(x.get('reference_field') == 'equivalent_upstream_image' for x in record['evidence']))
        self.assertTrue(any(x.get('reference_field') == 'source' and x.get('relation') == 'referenced_as'
                            for x in record['evidence']))

    def test_split_version_is_reported_without_assuming_v4(self):
        records = [
            {'manifest': 'old.json', 'record_index': 0, 'relation': 'describes',
             'manifest_context': {'dataset_version': 'v3'}, 'claims': {'split': 'train'}},
            {'manifest': 'unspecified.json', 'record_index': 0, 'relation': 'describes',
             'manifest_context': {}, 'claims': {'split': 'val'}},
            {'manifest': 'v4.json', 'record_index': 0, 'relation': 'describes',
             'manifest_context': {'dataset_version': 'v4'}, 'claims': {'split': 'test'}},
        ]
        result = media_index.structured_claims(records)
        self.assertEqual(result['declared_splits'], ['train', 'val', 'test'])
        self.assertEqual(result['v4_declared_splits'], ['test'])
        self.assertEqual([r['dataset_version'] for r in result['split_claims']], ['v3', None, 'v4'])

    def test_batch_evaluation_stays_at_batch_scope(self):
        batch = self.root / 'datasets/annotated/report'
        batch.mkdir(parents=True)
        (batch / 'distribution.json').write_text(json.dumps({
            'scope': 'synthetic stress set',
            'data_distribution': {'shared_source_groups_by_split': {'train|test': ['shared-video']}},
            'evaluation': {'status': 'not_run', 'data_scope': 'synthetic_stress', 'metrics': None},
            'records': [{'image': 'x.png', 'split': 'test'}]}), encoding='utf8')
        (batch / 'x.png').write_bytes(b'\x89PNG\r\n\x1a\n' + b'\0' * 8 + (8).to_bytes(4, 'big') * 2)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['build', '--scan-root', 'datasets/annotated', '--no-video-probe']), 0)
        with contextlib.redirect_stdout(output := io.StringIO()):
            self.assertEqual(media_index.main(['batches', '--evaluation-scope', 'synthetic_stress']), 0)
        batch_record = json.loads(output.getvalue())
        self.assertEqual(batch_record['relation'], 'batch_context')
        self.assertEqual(batch_record['batch_context']['evaluation']['status'], 'not_run')
        with contextlib.redirect_stdout(output := io.StringIO()):
            media_index.main(['query', '--contains', 'x.png'])
        image = json.loads(output.getvalue())
        self.assertEqual(image['recorded_claims']['result_evidence'], [])

    def test_json_suffix_with_jsonl_content_is_read_safely(self):
        batch = self.root / 'datasets/annotated/lines'
        batch.mkdir(parents=True)
        (batch / 'a.png').write_bytes(b'\x89PNG\r\n\x1a\n' + b'\0' * 8 + (8).to_bytes(4, 'big') * 2)
        (batch / 'b.png').write_bytes(b'\x89PNG\r\n\x1a\n' + b'\0' * 8 + (8).to_bytes(4, 'big') * 2)
        (batch / '3_manifest.json').write_text('{"image":"a.png","source_group":"first"}\n'
                                                '{"image":"b.png","source_group":"second"}\n', encoding='utf8')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['build', '--scan-root', 'datasets/annotated', '--no-video-probe']), 0)
        with contextlib.redirect_stdout(output := io.StringIO()):
            media_index.main(['query', '--contains', 'b.png', '--with-evidence'])
        self.assertEqual(json.loads(output.getvalue())['evidence'][0]['claims']['source_group'], 'second')

    def test_evidence_and_batches_do_not_repeat_overlapping_snapshot(self):
        batch = self.root / 'datasets/exports/task/batch'
        batch.mkdir(parents=True)
        (batch / 'x.png').write_bytes(b'\x89PNG\r\n\x1a\n' + b'\0' * 8 + (8).to_bytes(4, 'big') * 2)
        (batch / 'manifest.json').write_text(json.dumps({
            'scope': 'batch scope', 'records': [{'image': 'x.png', 'split': 'train'}]}), encoding='utf8')
        with contextlib.redirect_stdout(io.StringIO()):
            media_index.main(['build', '--scan-root', 'datasets/exports', '--no-video-probe'])
        catalog_path = media_index.INDEX / 'catalog.json'
        catalog = json.loads(catalog_path.read_text(encoding='utf8'))
        part = catalog['partitions']['datasets/exports/task/batch']
        catalog['partitions']['datasets/exports/task/overlap'] = dict(part)
        catalog_path.write_text(json.dumps(catalog), encoding='utf8')
        with contextlib.redirect_stdout(output := io.StringIO()):
            media_index.main(['evidence', 'datasets/exports/task/batch/x.png', '--limit', '0'])
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        with contextlib.redirect_stdout(output := io.StringIO()):
            media_index.main(['batches', '--contains', 'batch scope', '--limit', '0'])
        self.assertEqual(len(output.getvalue().splitlines()), 1)

    def test_windows_lifecycle_matching_ignores_case(self):
        if media_index.os.name != 'nt':
            self.skipTest('Windows path case behavior')
        media_index.load_lifecycle_rules()
        self.assertTrue(media_index.lifecycle_reason(
            'DATASETS/EXPORTS/WASTE_BIN/real_scene_bin_train_4000_20261110/a.png'))

    def test_v3_run_data_is_excluded_without_losing_shared_v4_source(self):
        run = self.root / 'runs/plate_detection/plate_detection/v3_20261111_152409_363091'
        (run / 'data/images').mkdir(parents=True)
        (run / 'training').mkdir()
        png = b'\x89PNG\r\n\x1a\n' + b'\0' * 8 + (8).to_bytes(4, 'big') * 2
        (run / 'data/images/old.png').write_bytes(png)
        (run / 'training/plot.png').write_bytes(png)
        shared = self.root / 'datasets/exports/plate/shared.png'
        shared.parent.mkdir(parents=True)
        shared.write_bytes(png)
        (run / 'data/samples.json').write_text(json.dumps([{
            'workspace_image': 'datasets/exports/plate/shared.png', 'split': 'train'}]), encoding='utf8')
        (shared.parent / 'manifest.json').write_text(json.dumps({'records': [{
            'image': 'datasets/exports/plate/shared.png', 'dataset_version': 'v4', 'split': 'train'}]}), encoding='utf8')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(media_index.main(['build', '--scan-root', 'runs', '--scan-root',
                                               'datasets/exports', '--no-video-probe']), 0)
        with contextlib.redirect_stdout(output := io.StringIO()):
            media_index.main(['query', '--contains', 'old.png', '--include-missing'])
        self.assertEqual(output.getvalue(), '')
        with contextlib.redirect_stdout(output := io.StringIO()):
            media_index.main(['query', '--contains', 'plot.png'])
        self.assertIn('plot.png', output.getvalue())
        with contextlib.redirect_stdout(output := io.StringIO()):
            media_index.main(['evidence', 'datasets/exports/plate/shared.png', '--limit', '0'])
        self.assertIn('v4', output.getvalue())
        self.assertNotIn('/data/samples.json', output.getvalue())

    def test_incremental_policy_change_removes_old_run_evidence_for_shared_path(self):
        run = self.root / 'runs/plate_detection/plate_detection/v3_20261111_152409_363091'
        (run / 'data').mkdir(parents=True)
        shared = self.root / 'datasets/exports/plate/shared.png'
        shared.parent.mkdir(parents=True)
        shared.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\0' * 8 + (8).to_bytes(4, 'big') * 2)
        (run / 'data/samples.json').write_text(json.dumps([{
            'workspace_image': 'datasets/exports/plate/shared.png', 'split': 'train'}]), encoding='utf8')
        (shared.parent / 'manifest.json').write_text(json.dumps({'records': [{
            'image': 'datasets/exports/plate/shared.png', 'dataset_version': 'v4'}]}), encoding='utf8')
        policy_path = self.root / media_index.LIFECYCLE_POLICY
        policy = json.loads(policy_path.read_text(encoding='utf8'))
        kept = dict(policy)
        kept['rules'] = [r for r in policy['rules'] if r['path'] != (run / 'data').relative_to(self.root).as_posix()]
        policy_path.write_text(json.dumps(kept), encoding='utf8')
        with contextlib.redirect_stdout(io.StringIO()):
            media_index.main(['build', '--scan-root', 'runs', '--scan-root', 'datasets/exports', '--no-video-probe'])
        with contextlib.redirect_stdout(output := io.StringIO()):
            media_index.main(['evidence', 'datasets/exports/plate/shared.png', '--limit', '0'])
        self.assertIn('/data/samples.json', output.getvalue())
        policy_path.write_text(json.dumps(policy), encoding='utf8')
        with contextlib.redirect_stdout(io.StringIO()):
            media_index.main(['build', '--scan-root', 'datasets/exports', '--no-video-probe'])
        with contextlib.redirect_stdout(output := io.StringIO()):
            media_index.main(['evidence', 'datasets/exports/plate/shared.png', '--limit', '0'])
        self.assertNotIn('/data/samples.json', output.getvalue())
        self.assertIn('v4', output.getvalue())


if __name__ == '__main__':
    unittest.main()
