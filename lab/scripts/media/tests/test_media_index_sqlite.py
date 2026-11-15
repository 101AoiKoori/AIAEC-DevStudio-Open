"""Focused snapshot, provenance, lifecycle and interrupted-sync checks."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import media_index as core
import media_index_sqlite as cache
import raw_visual_query as visual


class CacheTests(unittest.TestCase):
    def setUp(self):
        workspace_tmp = core.ROOT / 'tmp'
        workspace_tmp.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=workspace_tmp)
        self.root = Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(workspace_tmp.resolve()))
        self.old_root, self.old_index, self.old_database = core.ROOT, core.INDEX, cache.DATABASE
        core.ROOT = self.root
        core.INDEX = self.root / 'datasets/annotated/media_index'
        cache.DATABASE = core.INDEX / 'query_cache.sqlite3'
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(setattr, core, 'ROOT', self.old_root)
        self.addCleanup(setattr, core, 'INDEX', self.old_index)
        self.addCleanup(setattr, cache, 'DATABASE', self.old_database)
        (self.root / 'lab/scripts/media').mkdir(parents=True)
        policy = {'schema_version': 1, 'policy_revision': 'fixture-1', 'rules': [],
                  'source_rules': [], 'excluded_patterns': []}
        (self.root / 'lab/scripts/media/media_index_lifecycle.json').write_text(json.dumps(policy), encoding='utf8')
        core.load_lifecycle_rules()
        self.revision = dict(core._POLICY_METADATA)
        (core.INDEX / 'shards').mkdir(parents=True)
        self.image = self.root / 'datasets/raw/camera/a.png'
        self.image.parent.mkdir(parents=True)
        self.image.write_bytes(b'fixture')
        item = {'id': 'workspace:datasets/raw/camera/a.png',
                'path': 'datasets/raw/camera/a.png', 'media_type': 'image',
                'bytes': 7, 'modified_ns': 1, 'image_size': [640, 480],
                'duration_seconds': None,
                'path_history': [{'path': 'datasets/raw/old/a.png', 'basis': 'explicit_relocation'}],
                'recorded_claims': {}, 'evidence': []}
        claims = {'source_group': '场景-Group-Alpha', 'generation_method': 'synthetic_render',
                  'split': 'train', 'dataset_version': 'v4', 'evaluation_scope': 'shared-scene',
                  'observed_content': '电动车停车区', 'review_flag': 'pending_label_review',
                  'background_view': 'oblique', 'background_use': 'unreviewed',
                  'mean_luma_0_255': '120.50', 'mean_saturation_0_255': '51.20',
                  'observed_tasks': ['person', 'e_bike'],
                  'regions': [{'role': 'ground', 'task_candidates': ['person'], 'polygon': [[0, 0], [2, 0], [2, 2]],
                               'use_status': 'candidate_not_approved'},
                              {'role': 'wall', 'task_candidates': ['fire_sticker'], 'polygon': [[0, 0], [2, 0], [2, 2]]}]}
        record = {'manifest': 'datasets/exports/derived/manifest.json', 'record_index': 0,
                  'relation': 'describes', 'claims': claims, 'manifest_context': {}}
        reference = {'manifest': 'datasets/exports/derived/manifest.json', 'record_index': 0,
                     'relation': 'referenced_as', 'reference_field': 'parent_image',
                     'claims': dict(claims, split='test', generation_method='other'), 'manifest_context': {}}
        entry = {'claimed_path': 'datasets/raw/camera/a.png', 'evidence': [record]}
        reference_entry = {'claimed_path': 'datasets/raw/camera/a.png', 'evidence': [reference]}
        batch = {'claimed_path': 'datasets/exports/derived/manifest.json', 'evidence': [{
            'manifest': 'datasets/exports/derived/manifest.json', 'record_index': None,
            'relation': 'batch_context', 'claims': {}, 'batch_context': {
                'status': 'candidate', 'evaluation': {'data_scope': 'shared-scene'}}}]}
        self.parts = {
            'datasets/raw/camera': self.shard('raw', [item], []),
            'datasets/exports/derived': self.shard('derived', [], [entry, reference_entry, batch]),
        }
        self.write_catalog()

    def shard(self, name, items, evidence):
        a = core.INDEX / 'shards' / (name + '.jsonl')
        b = core.INDEX / 'shards' / (name + '.evidence.jsonl')
        for p, records in ((a, items), (b, evidence)):
            p.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records), encoding='utf8')
        return {'source': '', 'source_present': True, 'shard': core.relative(a),
                'evidence_shard': core.relative(b), 'updated_utc': '2026-01-01T00:00:00Z'}

    def write_catalog(self):
        catalog = {'schema_version': 1, 'partitions': self.parts,
                   'latest_build_scope': {'state': 'complete', 'lifecycle_policy_revision': self.revision},
                   'last_full_build': {'state': 'complete', 'lifecycle_policy_revision': self.revision}}
        (core.INDEX / 'catalog.json').write_text(json.dumps(catalog), encoding='utf8')

    def call(self, args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cache.main(args)
        return code, out.getvalue(), err.getvalue()

    def test_partial_incremental_and_cross_partition_claims(self):
        self.assertEqual(self.call(['sync', '--partition', 'datasets/raw/camera'])[0], 0)
        self.assertEqual(self.call(['query', '--path', 'datasets/raw/camera/a.png'])[0], 1)
        code, out, err = self.call(['query', '--path', 'datasets/raw/camera/a.png', '--allow-partial'])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)['path'], 'datasets/raw/camera/a.png')
        self.assertIn('partial cache', err)
        self.assertEqual(self.call(['sync', '--partition', 'datasets/exports/derived'])[0], 0)
        code, out, err = self.call(['query', '--source-group', 'group-al', '--generation-method', 'THETIC_',
                                    '--split', 'rain', '--evaluation-scope', 'shared'])
        self.assertEqual(code, 0, err)
        item = json.loads(out)
        self.assertEqual(item['path'], 'datasets/raw/camera/a.png')
        self.assertEqual(item['path_history'][0]['path'], 'datasets/raw/old/a.png')
        self.assertEqual(item['recorded_claims']['v4_declared_splits'], ['train'])
        code, out, err = self.call(['query', '--observed-content', '停车区',
                                    '--review-flag', 'pending_label_review'])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)['path'], 'datasets/raw/camera/a.png')
        code, out, err = self.call(['query', '--observed-content', '楼宇'])
        self.assertEqual(code, 0, err)
        self.assertEqual(out, '')
        code, out, err = self.call(['query', '--background-view', 'oblique',
                                    '--min-luma', '120', '--max-luma', '121',
                                    '--min-saturation', '50', '--max-saturation', '52'])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)['path'], 'datasets/raw/camera/a.png')
        code, out, err = self.call(['query', '--min-luma', '121'])
        self.assertEqual(code, 0, err)
        self.assertEqual(out, '')
        code, out, err = self.call(['query', '--background-use', 'reviewed'])
        self.assertEqual(code, 0, err)
        self.assertEqual(out, '')
        code, out, err = self.call(['query', '--observed-task', 'person', '--region-role', 'ground', '--with-evidence'])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)['evidence'][0]['claims']['regions'][0]['role'], 'ground')
        code, out, err = self.call(['query', '--region-role', 'ground', '--region-task', 'person'])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)['path'], 'datasets/raw/camera/a.png')
        code, out, err = self.call(['query', '--region-role', 'ground', '--region-task', 'fire_sticker'])
        self.assertEqual(code, 0, err)
        self.assertEqual(out, '')
        code, out, err = self.call(['query', '--observed-task', 'pers'])
        self.assertEqual(code, 0, err)
        self.assertEqual(out, '')
        code, out, err = self.call(['evidence', 'datasets/raw/camera/a.png'])
        self.assertEqual(code, 0, err)
        self.assertEqual(len(out.splitlines()), 2)
        self.assertEqual({json.loads(x)['evidence'][0]['relation'] for x in out.splitlines()},
                         {'describes', 'referenced_as'})
        code, out, err = self.call(['query', '--path', 'datasets/raw/camera/a.png', '--split', 'test'])
        self.assertEqual(code, 0, err)
        self.assertEqual(out, '')
        code, out, err = self.call(['batches', '--evaluation-scope', 'shared', '--status', 'date'])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)['relation'], 'batch_context')

    def test_normalized_evidence_preserves_each_declaration(self):
        evidence_path = self.root / self.parts['datasets/exports/derived']['evidence_shard']
        entries = [json.loads(line) for line in evidence_path.read_text(encoding='utf8').splitlines()]
        source = entries[0]['evidence'][0]
        parent = dict(source, relation='referenced_as', reference_field='parent_image',
                      subject_paths=['datasets/raw/camera/a.png'])
        entries.append({'claimed_path': 'datasets/raw/camera/parent.png', 'evidence': [parent]})
        alias = dict(source, manifest='datasets/exports/other/manifest.json')
        entries.append({'claimed_path': 'datasets/raw/camera/a.png', 'evidence': [alias]})
        conflict = dict(source, claims=dict(source['claims'], split='validation'))
        entries.append({'claimed_path': 'datasets/raw/camera/a.png', 'evidence': [conflict]})
        entries.append({'claimed_path': 'datasets/raw/camera/a.png', 'evidence': [source, alias]})
        evidence_path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in entries), encoding='utf8')
        self.assertEqual(self.call(['sync'])[0], 0)
        db = cache.connect(readonly=True)
        try:
            args = types.SimpleNamespace(**{key: None for key in cache.FILTER_FIELDS})
            actual = [json.loads(raw) for _, raw, _ in cache.matching_entries(db, args)]
            self.assertEqual(actual, entries)
            self.assertEqual(db.execute('select count(*) from manifest_rows').fetchone()[0], 4)
            self.assertEqual(db.execute('select count(*) from evidence').fetchone()[0], len(entries))
        finally:
            db.close()

    def test_raw_visual_query_preserves_candidate_and_rejected_separately(self):
        evidence_path = self.root / self.parts['datasets/exports/derived']['evidence_shard']
        entries = [json.loads(line) for line in evidence_path.read_text(encoding='utf8').splitlines()]
        claims = entries[0]['evidence'][0]['claims']
        claims['instances'] = [{'task': 'sign', 'status': 'visible_part_only', 'use_status': 'not_training_box', 'bbox': [1, 2, 3, 4]}]
        claims['rejected_instances'] = [{'task': 'sign', 'reason': 'wrong_location'}]
        claims['background_observations'] = [{'role': 'white_back', 'permitted_as_person_positive': False}]
        claims['visible_ground_patches'] = [{'polygon': [[0, 0], [2, 0], [2, 2]], 'use_status': 'local_reference'}]
        claims['keypoints'] = [{'name': 'T_tip', 'xy': None, 'status': 'unknown'}]
        claims['dial_bbox'] = [2, 3, 20, 21]
        claims['bbox_status'] = 'candidate'
        claims['local_count_observations'] = {'person': {'visible_candidate_lower_bound': 3, 'upper_bound': 'unknown'}}
        claims['task_observations'] = {'person': {'exhaustive': False, 'instances': [{'bbox_xyxy': [1, 2, 3, 4]}]}}
        evidence_path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in entries), encoding='utf8')
        self.assertEqual(self.call(['sync'])[0], 0)
        result = visual.query_image('datasets/raw/camera/a.png')
        self.assertEqual(result['regions'][0]['use_status'], 'candidate_not_approved')
        self.assertEqual(len(result['instances']), 2)
        self.assertEqual(result['instances'][0]['instance']['status'], 'visible_part_only')
        self.assertEqual(result['instances'][0]['use_status'], 'not_training_box')
        self.assertIsNone(result['instances'][0]['bbox_xyxy'])
        self.assertEqual(result['instances'][0]['bbox_untyped'], [1, 2, 3, 4])
        self.assertEqual(result['instances'][1]['bbox_xyxy'], [1, 2, 3, 4])
        self.assertEqual(result['instances'][1]['task_scope']['exhaustive'], False)
        self.assertEqual(result['rejected_instances'][0]['instance']['reason'], 'wrong_location')
        self.assertFalse(result['background_observations'][0]['observation']['permitted_as_person_positive'])
        self.assertEqual(result['visible_ground_patches'][0]['patch']['use_status'], 'local_reference')
        code, out, err = self.call(['query', '--ground-patch-use', 'local_reference'])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)['path'], 'datasets/raw/camera/a.png')
        code, out, err = self.call(['query', '--ground-patch-use', 'reference'])
        self.assertEqual(code, 0, err)
        self.assertEqual(out, '')
        self.assertIsNone(result['keypoints'][0]['keypoints'][0]['xy'])
        self.assertEqual(result['keypoints'][0]['dial_bbox'], [2, 3, 20, 21])
        self.assertEqual(result['local_count_observations'][0]['observation']['upper_bound'], 'unknown')

    def test_possible_duplicate_boxes_are_flagged_not_merged(self):
        items = [{'task': 'person', 'bbox_xyxy': [0, 0, 10, 10]},
                 {'task': 'person', 'bbox_xyxy': [0, 0, 10, 10]},
                 {'task': 'e_bike', 'bbox_xyxy': [0, 0, 10, 10]},
                 {'task': 'person', 'bbox_xyxy': None}]
        self.assertEqual(visual.possible_duplicates(items), [
            {'instance_indices': [0, 1], 'iou': 1.0, 'status': 'possible_duplicate_not_merged'}])
        self.assertEqual(len(items), 4)

    def test_repeat_sync_and_single_partition_refresh_keep_evidence(self):
        self.assertEqual(self.call(['sync'])[0], 0)
        db = cache.connect(readonly=True)
        try:
            counts = tuple(db.execute('select count(*) from ' + table).fetchone()[0]
                           for table in ('evidence', 'manifest_rows'))
            metadata = dict(db.execute('select key,value from meta'))
            retained = list(db.execute('select position,path,entry_json,manifest_row_id from evidence '
                                       'where partition_name=? order by position',
                                       ('datasets/exports/derived',)))
        finally:
            db.close()
        stat_before = cache.DATABASE.stat()
        changes = []
        real_connect = cache.connect
        class TrackedConnection:
            def __init__(self, db):
                self.db = db
            def __getattr__(self, key):
                return getattr(self.db, key)
            def close(self):
                changes.append(self.db.total_changes)
                self.db.close()
        def tracked_connect(readonly=False):
            self.assertTrue(readonly, 'unchanged sync must not open a write connection')
            return TrackedConnection(real_connect(readonly=True))
        with mock.patch.object(cache, 'import_partition', wraps=cache.import_partition) as imported, \
                mock.patch.object(cache, 'connect', side_effect=tracked_connect):
            self.assertEqual(self.call(['sync'])[0], 0)
            imported.assert_not_called()
        self.assertEqual(changes, [0])
        self.assertEqual(cache.DATABASE.stat().st_mtime_ns, stat_before.st_mtime_ns)
        db = cache.connect(readonly=True)
        try:
            self.assertEqual(dict(db.execute('select key,value from meta')), metadata)
            self.assertEqual(counts, tuple(db.execute('select count(*) from ' + table).fetchone()[0]
                                           for table in ('evidence', 'manifest_rows')))
        finally:
            db.close()

        path = self.root / self.parts['datasets/raw/camera']['evidence_shard']
        extra = {'claimed_path': 'datasets/raw/camera/a.png', 'evidence': [{
            'manifest': 'datasets/raw/camera/new_manifest.json', 'record_index': 42,
            'relation': 'describes', 'claims': {'source_group': 'new-source'}}]}
        path.write_text(json.dumps(extra) + '\n', encoding='utf8')
        with mock.patch.object(cache, 'import_partition', wraps=cache.import_partition) as imported:
            self.assertEqual(self.call(['sync'])[0], 0)
            self.assertEqual([call.args[1]['name'] for call in imported.call_args_list],
                             ['datasets/raw/camera'])
        db = cache.connect(readonly=True)
        try:
            self.assertEqual(retained, list(db.execute(
                'select position,path,entry_json,manifest_row_id from evidence '
                'where partition_name=? order by position', ('datasets/exports/derived',))))
            self.assertEqual(tuple(n + 1 for n in counts),
                             tuple(db.execute('select count(*) from ' + table).fetchone()[0]
                                   for table in ('evidence', 'manifest_rows')))
        finally:
            db.close()
        code, output, _ = self.call(['evidence', 'datasets/raw/camera/a.png'])
        self.assertEqual(code, 0)
        identities = {(record['manifest'], record['record_index'])
                      for line in output.splitlines() for record in json.loads(line)['evidence']}
        self.assertIn(('datasets/exports/derived/manifest.json', 0), identities)
        self.assertIn(('datasets/raw/camera/new_manifest.json', 42), identities)

    def test_build_then_sync_reuses_unchanged_partitions(self):
        (self.root / 'datasets/exports/derived').mkdir(parents=True)
        (self.root / 'datasets/exports/derived/b.png').write_bytes(b'derived')
        (core.INDEX / 'catalog.json').unlink()
        with mock.patch.object(core, 'DEFAULT_SOURCES', ('datasets/raw', 'datasets/exports')), \
                mock.patch.object(core, 'ROOT_FILE_SOURCES', ()), \
                mock.patch.object(core, 'PROJECT_ASSET_DIRS', ()):
            def build():
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(core.main(['build', '--no-video-probe']), 0)
                return json.loads((core.INDEX / 'catalog.json').read_text(encoding='utf8'))['partitions']
            before = build()
            self.assertEqual(self.call(['sync'])[0], 0)
            db_mtime = cache.DATABASE.stat().st_mtime_ns
            catalog_path = core.INDEX / 'catalog.json'
            catalog_before = (catalog_path.stat().st_mtime_ns, catalog_path.read_bytes())
            with mock.patch.object(core, 'build_partition', side_effect=AssertionError('unexpected rebuild')):
                same = build()
            self.assertEqual((catalog_path.stat().st_mtime_ns, catalog_path.read_bytes()), catalog_before)
            for name in before:
                self.assertEqual(before[name], same[name])
            with mock.patch.object(cache, 'import_partition', wraps=cache.import_partition) as imported:
                self.assertEqual(self.call(['sync'])[0], 0)
                imported.assert_not_called()
            self.assertEqual(cache.DATABASE.stat().st_mtime_ns, db_mtime)
            self.image.write_bytes(b'changed-camera')
            changed = build()
            changed_names = [name for name in before if before[name]['shard'] != changed[name]['shard']]
            self.assertEqual(changed_names, ['datasets/raw/camera'])
            with mock.patch.object(cache, 'import_partition', wraps=cache.import_partition) as imported:
                self.assertEqual(self.call(['sync'])[0], 0)
                self.assertEqual([c.args[1]['name'] for c in imported.call_args_list], changed_names)

    def test_missing_cascade_indexes_are_repaired_before_unchanged_fast_path(self):
        self.assertEqual(self.call(['sync'])[0], 0)
        db = cache.connect()
        try:
            db.execute('DROP INDEX manifest_rows_partition')
            db.execute('DROP INDEX evidence_manifest_row')
            db.commit()
        finally:
            db.close()
        with mock.patch.object(cache, 'import_partition', wraps=cache.import_partition) as imported:
            self.assertEqual(self.call(['sync'])[0], 0)
            imported.assert_not_called()
        db = cache.connect(readonly=True)
        try:
            queries = [('SELECT rowid FROM manifest_rows WHERE partition_name=?', 'manifest_rows_partition'),
                       ('SELECT rowid FROM evidence WHERE manifest_row_id=?', 'evidence_manifest_row')]
            for query, index in queries:
                details = ' '.join(str(row[-1]) for row in db.execute('EXPLAIN QUERY PLAN ' + query, (1,)))
                self.assertIn('SEARCH', details)
                self.assertIn(index, details)
        finally:
            db.close()

    def test_evidence_persists_across_new_processes(self):
        self.assertEqual(self.call(['sync'])[0], 0)
        code = ('import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); '
                'import media_index as m, media_index_sqlite as c; '
                'm.ROOT=Path(sys.argv[2]); m.INDEX=m.ROOT/"datasets/annotated/media_index"; '
                'c.DATABASE=m.INDEX/"query_cache.sqlite3"; '
                'sys.exit(c.main(["evidence","datasets/raw/camera/a.png"]))')
        outputs = []
        for _ in range(2):
            result = subprocess.run([sys.executable, '-c', code, str(Path(cache.__file__).parent),
                                     str(self.root)], capture_output=True, text=True,
                                    encoding='utf8', timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append([json.loads(line) for line in result.stdout.splitlines()])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual({(record['manifest'], record['record_index'])
                          for entry in outputs[0] for record in entry['evidence']},
                         {('datasets/exports/derived/manifest.json', 0)})

    def test_changed_shard_and_policy_refuse_stale_reads(self):
        self.assertEqual(self.call(['sync'])[0], 0)
        shard = self.root / self.parts['datasets/raw/camera']['shard']
        stat = shard.stat()
        os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        code, out, _ = self.call(['status'])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)['state'], 'stale_or_unavailable')
        self.assertEqual(self.call(['query'])[0], 1)
        self.assertEqual(self.call(['sync'])[0], 0)
        policy = self.root / 'lab/scripts/media/media_index_lifecycle.json'
        changed = json.loads(policy.read_text(encoding='utf8'))
        changed['policy_revision'] = 'fixture-2'
        policy.write_text(json.dumps(changed), encoding='utf8')
        self.assertEqual(self.call(['query'])[0], 1)

    def test_interrupted_partition_cannot_be_queried(self):
        self.assertEqual(self.call(['sync'])[0], 0)
        shard = self.root / self.parts['datasets/raw/camera']['shard']
        stat = shard.stat()
        os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with mock.patch.object(cache, 'import_partition', side_effect=ValueError('simulated interruption')):
            self.assertEqual(self.call(['sync'])[0], 1)
        self.assertEqual(self.call(['query'])[0], 1)
        self.assertEqual(self.call(['sync'])[0], 0)
        self.assertEqual(self.call(['query'])[0], 0)

    def test_lifecycle_change_rejects_then_excludes_old_media(self):
        self.assertEqual(self.call(['sync'])[0], 0)
        path = self.root / 'lab/scripts/media/media_index_lifecycle.json'
        policy = json.loads(path.read_text(encoding='utf8'))
        policy['policy_revision'] = 'fixture-inactive'
        policy['rules'] = [{'path': 'datasets/raw/camera', 'scope': 'media_subtree',
                            'state': 'inactive', 'reason': 'old_v3', 'evidence': []}]
        path.write_text(json.dumps(policy), encoding='utf8')
        self.assertEqual(self.call(['query'])[0], 1)
        core.load_lifecycle_rules()
        self.revision = dict(core._POLICY_METADATA)
        self.write_catalog()
        self.assertEqual(self.call(['sync'])[0], 0)
        code, out, err = self.call(['query'])
        self.assertEqual(code, 0, err)
        self.assertEqual(out, '')
        code, out, err = self.call(['evidence', 'datasets/raw/camera/a.png'])
        self.assertEqual(code, 0, err)
        self.assertEqual(out, '')

    def test_shard_changes_during_sync_refuse_publication(self):
        original = cache.read_shards
        changed = False
        def mutate_after_read(part, policy):
            nonlocal changed
            result = original(part, policy)
            if not changed:
                changed = True
                shard = self.root / part['shard']
                stat = shard.stat()
                os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            return result
        with mock.patch.object(cache, 'read_shards', side_effect=mutate_after_read):
            self.assertEqual(self.call(['sync'])[0], 1)
        self.assertEqual(self.call(['query'])[0], 1)
        self.assertEqual(self.call(['sync'])[0], 0)
        self.assertEqual(self.call(['query'])[0], 0)

    def test_incomplete_json_snapshot_never_serves_full_cache(self):
        self.assertEqual(self.call(['sync'])[0], 0)
        path = core.INDEX / 'catalog.json'
        catalog = json.loads(path.read_text(encoding='utf8'))
        catalog['latest_build_scope']['failures'] = [{'source': 'example', 'error': 'failed'}]
        path.write_text(json.dumps(catalog), encoding='utf8')
        self.assertEqual(self.call(['query'])[0], 1)
        self.assertEqual(self.call(['sync'])[0], 1)
        # An explicit restricted cache can be inspected, with a partial warning.
        self.assertEqual(self.call(['sync', '--partition', 'datasets/raw/camera'])[0], 0)
        self.assertEqual(self.call(['query'])[0], 1)
        self.assertEqual(self.call(['query', '--allow-partial'])[0], 0)
        catalog['last_full_build']['failures'] = [{'source': 'example', 'error': 'failed'}]
        path.write_text(json.dumps(catalog), encoding='utf8')
        self.assertEqual(self.call(['query', '--allow-partial'])[0], 1)

    def test_documented_omission_is_reported_and_parse_failure_blocks_full(self):
        path = core.INDEX / 'catalog.json'
        catalog = json.loads(path.read_text(encoding='utf8'))
        catalog['partitions']['datasets/exports/derived']['warnings'] = [
            {'manifest': 'example.json', 'error': 'manifest exceeds 64 MiB; not parsed'}]
        catalog['partitions']['datasets/exports/derived']['source_present'] = False
        path.write_text(json.dumps(catalog), encoding='utf8')
        self.assertEqual(self.call(['sync'])[0], 0)
        code, out, err = self.call(['status'])
        self.assertEqual(code, 0, err)
        state = json.loads(out)
        self.assertEqual(state['missing_sources'], ['datasets/exports/derived'])
        self.assertEqual(len(state['unparsed_manifests']), 1)
        catalog['partitions']['datasets/exports/derived']['warnings'][0]['error'] = 'JSON parse error'
        path.write_text(json.dumps(catalog), encoding='utf8')
        self.assertEqual(self.call(['query'])[0], 1)
        self.assertEqual(self.call(['sync'])[0], 1)

    @unittest.skipUnless(os.name == 'nt', 'Windows path semantics')
    def test_windows_case_variant_path_and_evidence(self):
        self.assertEqual(self.call(['sync'])[0], 0)
        code, out, err = self.call(['query', '--path', 'DATASETS/RAW/CAMERA/A.PNG', '--source-group', 'alpha'])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)['path'], 'datasets/raw/camera/a.png')
        code, out, err = self.call(['evidence', 'DATASETS/RAW/CAMERA/A.PNG'])
        self.assertEqual(code, 0, err)
        self.assertEqual(len(out.splitlines()), 2)

    def test_concurrent_writer_is_excluded(self):
        module_dir = Path(cache.__file__).parent
        code = ('import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); '
                'import media_index_sqlite as c; c.DATABASE=Path(sys.argv[2]); '
                'g=c.cache_guard(timeout=.2); g.__enter__()')
        with cache.cache_guard():
            result = subprocess.run([sys.executable, '-c', code, str(module_dir), str(cache.DATABASE)],
                                    capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SQLite cache writer is busy', result.stderr)


    def task_declaration_fixture(self):
        folder = self.root / 'datasets/scripts/person/config'
        folder.mkdir(parents=True)
        files = []
        for name in ('source.yaml', 'dataset.yaml', 'production_plan.json'):
            p = folder / name
            p.write_text('{}', encoding='utf8')
            files.append({'path': core.relative(p), 'present': True,
                          'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns})
        files.append({'path': core.relative(folder / 'render_plan.json'), 'present': False})
        config = {'schema_version': 1, 'task': 'person', 'declared_files': files,
                  'source_config': {'current_selection': {'excluded_records': [
                      {'image': 'datasets/raw/camera/excluded.png', 'reason': 'missing_labels'}]}},
                  'dataset_config': {'train_target_images': 3000},
                  'plans': [{'plan_key': 'production_plan', 'path': core.relative(folder/'production_plan.json'),
                             'present': True, 'budget': 1, 'target_split': 'train'},
                            {'plan_key': 'render_plan', 'path': core.relative(folder/'render_plan.json'),
                             'present': False}]}
        state = {'task': 'person', 'plan_key': 'production_plan',
                 'plan_path': core.relative(folder/'production_plan.json'),
                 'target_split': 'train', 'budget': 1,
                 'stratum': {'id': 'sample', 'background': {'image': core.relative(self.image)},
                             'target': {'source': 'datasets/samples/person/future.png'},
                             'labels': {'status': 'candidate'}, 'sources': [core.relative(self.image)]}}
        records = [
            {'claimed_path': core.relative(folder/'source.yaml'), 'evidence': [{
                'manifest': core.relative(folder/'source.yaml'), 'record_index': 0,
                'relation': 'task_config', 'claims': {'task_config': config}}]},
            {'claimed_path': state['plan_path'], 'evidence': [{
                'manifest': state['plan_path'], 'record_index': 0,
                'relation': 'plan_state', 'claims': {'plan_state': state}}]},
        ]
        self.parts['datasets/scripts/person'] = self.shard('person-config', [], records)
        self.write_catalog()
        self.assertEqual(self.call(['sync'])[0], 0)
        return folder

    def test_task_plans_inputs_and_exclusions_are_separate(self):
        self.task_declaration_fixture()
        code, out, _ = self.call(['task-inputs', '--task', 'person'])
        self.assertEqual(code, 0)
        plans = [json.loads(line) for line in out.splitlines()]
        self.assertEqual([(p['status'], p['state_count']) for p in plans],
                         [('declared', 1), ('pending_plan', 0)])
        code, out, _ = self.call(['task-inputs', '--task', 'person', '--kind', 'input',
                                 '--role', 'background', '--split', 'train'])
        self.assertEqual(code, 0)
        inputs = [json.loads(line) for line in out.splitlines()]
        self.assertEqual(len(inputs), 1)
        self.assertTrue(inputs[0]['present'])
        self.assertEqual(inputs[0]['path'], core.relative(self.image))
        self.assertEqual(inputs[0]['plan_key'], 'production_plan')
        code, out, _ = self.call(['task-inputs', '--task', 'person', '--kind', 'input', '--role', 'target'])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(out)['present'])
        code, out, _ = self.call(['task-inputs', '--task', 'person', '--kind', 'config'])
        self.assertEqual(code, 0)
        self.assertIn('missing_labels', out)
        self.assertEqual(self.call(['task-inputs', '--task', 'person', '--kind', 'input',
                                    '--path', 'datasets/raw/camera/excluded.png'])[1], '')
        code, out, _ = self.call(['task-inputs', '--task', 'person', '--kind', 'state'])
        self.assertEqual(json.loads(out)['stratum']['labels']['status'], 'candidate')

    def test_task_query_rejects_changed_declaration_without_rescanning_media(self):
        folder = self.task_declaration_fixture()
        (folder/'source.yaml').write_text('{"changed": true}', encoding='utf8')
        code, out, error = self.call(['task-inputs', '--task', 'person'])
        self.assertEqual(code, 1)
        self.assertEqual(out, '')
        self.assertIn('refresh this partition', error)

    def test_task_query_requires_refresh_when_pending_plan_appears(self):
        folder = self.task_declaration_fixture()
        (folder/'render_plan.json').write_text('{}', encoding='utf8')
        code, out, error = self.call(['task-inputs', '--task', 'person'])
        self.assertEqual(code, 1)
        self.assertEqual(out, '')
        self.assertIn('render_plan.json', error)

    def test_missing_task_partition_is_not_empty_success(self):
        self.assertEqual(self.call(['sync'])[0], 0)
        code, out, error = self.call(['task-inputs', '--task', 'meter'])
        self.assertEqual(code, 1)
        self.assertEqual(out, '')
        self.assertIn('not indexed', error)


if __name__ == '__main__':
    unittest.main()
