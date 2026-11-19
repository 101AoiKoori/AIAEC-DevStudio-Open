"""Machine review writes back to an existing frame without losing prior facts."""

from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
from PIL import Image

from datasets.scripts.common.frame_environment import edit_database, frame_history, query_frames
from datasets.scripts.common.review.placement import GroundPlacement
from datasets.scripts.common.review.placement_record import encode_placement_record
from datasets.scripts.common.background_machine_review import (
    build_machine_review_index, check_machine_review_index, publish_machine_review_index,
)
from datasets.scripts.common.ingest_frame_environment import ingest_frame, ingest_review_batch
from datasets.scripts.common.paths import WORKSPACE


class MachineReviewBatchTests(unittest.TestCase):
    def setUp(self):
        (WORKSPACE / 'tmp').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='machine_review_', dir=WORKSPACE / 'tmp')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.image = root / 'frame.png'
        Image.new('RGB', (80, 60), 'gray').save(self.image)
        self.path = self.image.relative_to(WORKSPACE).as_posix()
        self.digest = sha256(self.image.read_bytes()).hexdigest()
        self.database = root / 'frames.sqlite3'
        with edit_database(self.database) as writer:
            self.frame_id = writer.register(self.path, image_size=[80, 60],
                                            content_sha256=self.digest, parent_group='fixture_parent')
            old_label = writer.observe(
                self.frame_id, task='person', kind='label', scope='frame', evidence_key='legacy',
                reference={'manifest': 'fixture'},
                payload={'presence': 'absent', 'completeness': 'complete', 'scope': {'kind': 'frame'}},
                origin='human',
            )
            writer.decide(self.frame_id, task='person', role='empty_negative', split='unassigned',
                          status='eligible', evidence_ids=[old_label],
                          details={'reason_codes': ['human_complete_negative']}, origin='human')
            fire_label = writer.observe(
                self.frame_id, task='fire', kind='label', scope='frame', evidence_key='legacy',
                reference={'manifest': 'fixture'}, payload={'presence': 'unknown', 'completeness': 'unknown'},
                origin='human',
            )
            writer.decide(self.frame_id, task='fire', role='empty_negative', split='unassigned',
                          status='blocked', evidence_ids=[fire_label],
                          details={'reason_codes': ['human_exclusion']}, origin='human')

    def review(self):
        evidence = {'method': 'machine_visual_review', 'model_version': 'fixture_v1'}
        objects = [
            {'object_id': 'bin_gray', 'kind': 'waste_bin', 'color': 'gray', 'xyxy': [5, 8, 20, 48]},
            {'object_id': 'bin_green', 'kind': 'waste_bin', 'color': 'green', 'xyxy': [23, 8, 38, 48]},
        ]
        return {
            'image': self.path, 'frame_id': self.frame_id, 'content_sha256': self.digest,
            'objects': objects,
            'task_annotations': {
                'waste_bin_state': {
                    'presence': 'present', 'completeness': 'complete', 'scope': {'kind': 'frame'},
                    'instances': [{'object_id': 'bin_gray', 'class_id': 1},
                                  {'object_id': 'bin_green', 'class_id': 0}],
                    'evidence': evidence, 'split': 'unassigned',
                },
                'waste_sticker': {
                    'presence': 'absent', 'completeness': 'complete', 'scope': {'kind': 'frame'},
                    'instances': [], 'evidence': evidence, 'split': 'unassigned',
                },
                'fire': {
                    'presence': 'absent', 'completeness': 'complete', 'scope': {'kind': 'frame'},
                    'instances': [], 'evidence': evidence, 'split': 'unassigned',
                },
            },
            'environment': {'status': 'observed', 'algorithm_version': 'fixture_v1',
                            'evidence': evidence},
            'task_geometry': {task: {'status': 'unknown', 'reason_codes': ['not_measured']}
                              for task in ('waste_bin_state', 'waste_sticker', 'fire')},
            'lineage': {'source_kind': 'archive_camera_video_frame',
                        'capture_identity': 'fixture_parent:frame_1', 'parent_group': 'fixture_parent'},
        }

    def sign_ground_review(self):
        review = self.review()
        review['task_annotations'] = {
            'sign': {
                'presence': 'absent', 'completeness': 'complete', 'scope': {'kind': 'frame'},
                'instances': [], 'evidence': {'method': 'machine_visual_review',
                                              'model_version': 'fixture_v1'}, 'split': 'test',
            },
        }
        candidate = np.zeros((60, 80), dtype=np.bool_)
        candidate[30:55, 10:70] = True
        record = encode_placement_record(GroundPlacement(
            'candidate', 'fixture_floor', candidate, np.zeros_like(candidate),
        ))
        review['task_geometry'] = {'sign': {
            'status': 'candidate', 'ground_polygon': [[10, 30], [70, 30], [70, 55], [10, 55]],
            'placeable_ground': record, 'evidence': {'algorithm_version': 'fixture_floor_v1'},
        }}
        return review

    def add_old_sign_split(self, composite_status):
        with edit_database(self.database) as writer:
            old_geometry = writer.observe(
                self.frame_id, task='sign', kind='geometry', scope='legacy_review',
                evidence_key='background_pool', reference={'manifest': 'fixture'},
                payload={'ground_polygon': [[10, 30], [70, 30], [70, 55], [10, 55]]},
                origin='legacy',
            )
            writer.decide(self.frame_id, task='sign', role='geometry_reference', split='train',
                          status='eligible', evidence_ids=[old_geometry],
                          details={'reason_codes': ['legacy_region_available']}, origin='legacy')
            writer.decide(self.frame_id, task='sign', role='composite_background', split='train',
                          status=composite_status, evidence_ids=[old_geometry],
                          details={'reason_codes': ['legacy_background']}, origin='legacy')

    def test_whole_sticker_bin_uses_contact_without_surface_geometry(self):
        review = self.sign_ground_review()
        review['task_annotations']['waste_sticker'] = review['task_annotations'].pop('sign')
        geometry = review['task_geometry']['waste_sticker'] = review['task_geometry'].pop('sign')
        geometry.pop('ground_polygon')
        ingest_review_batch([review], database=self.database)
        background, = query_frames(task='waste_sticker', role='composite_background', database=self.database)
        self.assertEqual(background['geometry']['placeable_ground'], geometry['placeable_ground'])
        self.assertEqual(query_frames(task='waste_sticker', role='surface', database=self.database), [])

    def test_sticker_surface_survives_unresolved_ground_evidence(self):
        review = self.sign_ground_review()
        review['task_annotations']['waste_sticker'] = review['task_annotations'].pop('sign')
        geometry = review['task_geometry']['waste_sticker'] = review['task_geometry'].pop('sign')
        empty = np.zeros((60, 80), dtype=np.bool_)
        geometry['placeable_ground'] = encode_placement_record(GroundPlacement('unknown', 'no_ground', empty, empty))
        geometry['bins'] = [{'allowed_polygon': [[10, 10], [30, 10], [30, 20], [10, 20]],
                             'exclude_polygons': [], 'occluder_polygons': []}]
        ingest_review_batch([review], database=self.database)
        background, = query_frames(task='waste_sticker', role='composite_background', database=self.database)
        self.assertEqual(background['geometry']['placeable_ground']['status'], 'unknown')
        self.assertEqual(len(query_frames(task='waste_sticker', role='surface', database=self.database)), 1)

    def test_geometry_reference_is_frame_metadata_across_splits(self):
        self.add_old_sign_split('blocked')
        result = ingest_review_batch([self.sign_ground_review()], database=self.database)
        self.assertIn('geometry_reference', result[0]['records']['sign'])
        conn = sqlite3.connect(self.database)
        try:
            decisions = {(role, split): status for role, split, status in conn.execute(
                "SELECT role, split, status FROM decisions WHERE task='sign' AND current=1")}
            self.assertEqual(decisions['geometry_reference', 'train'], 'eligible')
            self.assertEqual(conn.execute(
                "SELECT status,current FROM decisions WHERE task='sign' "
                "AND role='geometry_reference' AND split='unassigned'").fetchone(), ('eligible', 0))
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM observations WHERE task='sign' AND kind='geometry' "
                "AND scope='placement' AND current=1").fetchone()[0], 1)
            self.assertEqual(decisions['composite_background', 'train'], 'blocked')
            self.assertEqual(decisions['composite_background', 'test'], 'eligible')
            self.assertEqual(decisions['empty_negative', 'test'], 'eligible')
        finally:
            conn.close()

    def test_composite_split_reuse_guard_remains_enforced(self):
        self.add_old_sign_split('eligible')
        with self.assertRaisesRegex(ValueError, 'Explicit split reuse needs evidence'):
            ingest_review_batch([self.sign_ground_review()], database=self.database)
        conn = sqlite3.connect(self.database)
        try:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE task='sign' AND split='test'").fetchone()[0], 0)
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM observations WHERE kind='objects'").fetchone()[0], 0)
        finally:
            conn.close()

    def test_primary_subject_blocks_only_its_own_composite_background(self):
        review = self.sign_ground_review()
        review['environment']['primary_subject_task'] = 'waste_bin_state'
        review['task_annotations']['waste_bin_state'] = self.review()['task_annotations']['waste_bin_state']
        review['task_geometry']['waste_bin_state'] = dict(review['task_geometry']['sign'])
        result = ingest_review_batch([review], database=self.database)[0]
        blocked = query_frames(task='waste_bin_state', role='composite_background',
                               record_id=result['records']['waste_bin_state']['composite_background'],
                               database=self.database)[0]
        self.assertEqual(blocked['status'], 'blocked')
        self.assertEqual(blocked['reason_codes'], ['primary_task_target_is_frame_subject'])
        subject = next(item for item in frame_history(self.frame_id, database=self.database)
                       if item['kind'] == 'environment' and item['current'])
        self.assertEqual(subject['payload']['primary_subject_task'], 'waste_bin_state')
        self.assertIn(subject['evidence_id'], {item['evidence_id'] for item in blocked['evidence']})
        for task, role in (
            ('waste_bin_state', 'direct_target'),
            ('sign', 'composite_background'),
            ('sign', 'empty_negative'),
        ):
            with self.subTest(task=task, role=role):
                self.assertEqual(query_frames(task=task, role=role, frame_id=self.frame_id,
                                              database=self.database)[0]['status'], 'eligible')

    def test_optional_subject_keeps_existing_composite_eligibility(self):
        for environment_fields in ({}, {'primary_subject_task': None}, {'primary_subject_task': ''}):
            with self.subTest(environment_fields=environment_fields):
                review = self.sign_ground_review()
                review['environment'].update(environment_fields)
                ingest_review_batch([review], database=self.database)
                self.assertEqual(len(query_frames(task='sign', role='composite_background',
                                                  split='test', database=self.database)), 1)

    def test_invalid_subject_task_cannot_write_decisions(self):
        review = self.sign_ground_review()
        for subject in ('other_task', ['sign']):
            with self.subTest(subject=subject):
                review['environment']['primary_subject_task'] = subject
                with self.assertRaisesRegex(ValueError, 'primary_subject_task must name a V4 task'):
                    ingest_review_batch([review], database=self.database)
        self.assertEqual(query_frames(task='sign', role='composite_background', status=None,
                                      database=self.database), [])

    def test_primary_subject_rejects_absence_but_accepts_unknown_labels(self):
        review = self.sign_ground_review()
        review['environment']['primary_subject_task'] = 'sign'
        with self.assertRaisesRegex(ValueError, 'primary_subject_task cannot have absent target presence'):
            ingest_review_batch([review], database=self.database)
        self.assertEqual(query_frames(task='sign', role='empty_negative', status=None,
                                      database=self.database), [])

        review['task_annotations']['sign'].update(presence='unknown', completeness='unknown')
        ingest_review_batch([review], database=self.database)
        background = query_frames(task='sign', role='composite_background', status=None, database=self.database)[0]
        self.assertEqual(background['label']['presence'], 'unknown')
        self.assertEqual(background['status'], 'blocked')
        self.assertEqual(background['reason_codes'], ['primary_task_target_is_frame_subject'])
        self.assertEqual(query_frames(task='sign', role='empty_negative', database=self.database), [])

    def test_source_metadata_cannot_create_a_complete_negative(self):
        review = self.sign_ground_review()
        review.pop('frame_id')
        review.pop('content_sha256')
        label = review['task_annotations']['sign']
        for evidence in (
            {'source_directory': 'datasets/raw/e_bike'},
            {'source_image': 'datasets/raw/e_bike/frame.jpg'},
            'datasets/raw/e_bike/frame.jpg',
            {'manifest': 'datasets/raw/e_bike/frame.jpg', 'record_index': 0},
            {'manifest': 'datasets/source_copy_manifest.jsonl', 'record_index': 0},
            {'manifest': 'datasets/scripts/e_bike/config/source.yaml', 'record_id': 'fixture'},
        ):
            with self.subTest(evidence=evidence):
                label['evidence'] = evidence
                with self.assertRaisesRegex(ValueError, 'Complete absence requires per-image review'):
                    ingest_frame(**review, database=self.database)

        for method in ('review', 'source_directory_review'):
            with self.subTest(method=method):
                label['evidence'] = {'method': method, 'algorithm_version': 'v1',
                                     'source_directory': 'datasets/raw/e_bike'}
                with self.assertRaisesRegex(ValueError, 'Complete absence requires per-image review'):
                    ingest_review_batch([review], database=self.database)
        self.assertEqual(query_frames(task='sign', role='empty_negative', status=None,
                                      database=self.database), [])

    def test_image_review_or_annotation_manifest_qualifies_a_complete_negative(self):
        review = self.sign_ground_review()
        review.pop('frame_id')
        review.pop('content_sha256')
        labels = Path(self.temp.name) / 'labels.json'
        labels.write_text(json.dumps([{'image': self.path, 'task': 'sign', 'instances': []}]), encoding='utf-8')
        for evidence in (
            {'method': 'original_full_frame_visual_review', 'finding': 'No recognizable sign is visible.',
             'source_directory': 'datasets/raw/e_bike'},
            {'manifest': labels.relative_to(WORKSPACE).as_posix(), 'record_index': 0},
        ):
            with self.subTest(evidence=evidence):
                review['task_annotations']['sign']['evidence'] = evidence
                result = ingest_frame(**review, database=self.database)
                records = query_frames(task='sign', role='empty_negative', frame_id=result['frame_id'],
                                       database=self.database)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['label']['evidence'], evidence)

    def test_existing_frame_adds_objects_and_preserves_omitted_and_human_decisions(self):
        review = self.review()
        result = ingest_review_batch([review], database=self.database)
        self.assertEqual(result[0]['frame_id'], self.frame_id)
        conn = sqlite3.connect(self.database)
        try:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM frames').fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM observations WHERE kind='objects'").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM observations WHERE kind='label' AND origin='human'").fetchone()[0], 2)
            decisions = {(task, role): (status, origin) for task, role, status, origin in conn.execute(
                "SELECT task, role, status, origin FROM decisions WHERE current=1")}
            self.assertEqual(decisions['person', 'empty_negative'], ('eligible', 'human'))
            self.assertEqual(decisions['fire', 'empty_negative'], ('blocked', 'human'))
            self.assertEqual(decisions['waste_bin_state', 'direct_target'], ('eligible', 'machine'))
            self.assertEqual(decisions['waste_sticker', 'empty_negative'], ('eligible', 'machine'))
        finally:
            conn.close()
        ingest_review_batch([review], database=self.database)
        conn = sqlite3.connect(self.database)
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM observations WHERE kind='objects'").fetchone()[0], 1)
        finally:
            conn.close()

    def test_first_intake_qualifies_person_without_measured_scale(self):
        fresh = Path(self.temp.name) / 'fresh.png'
        Image.new('RGB', (80, 60), 'gray').save(fresh)
        candidate = np.zeros((60, 80), dtype=np.bool_)
        candidate[30:55, 10:70] = True
        placement = encode_placement_record(GroundPlacement(
            'candidate', 'fixture_floor', candidate, np.zeros_like(candidate),
        ))
        evidence = {'method': 'fixture_complete_review'}
        result = ingest_frame(
            fresh.relative_to(WORKSPACE).as_posix(), database=self.database,
            objects=[{'object_id': 'bike_1', 'xyxy': [50, 10, 70, 30]}],
            task_annotations={
                'person': {'presence': 'absent', 'completeness': 'complete',
                           'instances': [], 'evidence': evidence, 'split': 'train'},
                'e_bike': {'presence': 'present', 'completeness': 'complete',
                           'instances': [{'object_id': 'bike_1', 'class_id': 0}],
                           'evidence': evidence, 'split': 'train'},
            },
            environment={'status': 'observed', 'evidence': evidence},
            task_geometry={
                'person': {'status': 'candidate', 'placeable_ground': placement,
                           'evidence': evidence},
                'e_bike': {'status': 'unknown', 'reason_codes': ['not_measured']},
            },
            lineage={'source_kind': 'camera_photo', 'parent_group': 'fresh_fixture'},
        )
        self.assertEqual(result['records']['person']['composite_background'],
                         query_frames(task='person', role='composite_background',
                                      frame_id=result['frame_id'], database=self.database)[0]['record_id'])
        person = query_frames(task='person', role='composite_background',
                              frame_id=result['frame_id'], database=self.database)[0]
        self.assertEqual(person['status'], 'eligible')
        self.assertEqual(person['geometry']['placeable_ground']['reference_quad_basis'],
                         'inscribed_contact_mask_rectangle')
        from datasets.scripts.person.planned_sources import environment_background_row, recorded_ground

        dispatch_row = environment_background_row(person)
        self.assertEqual(len(dispatch_row['ground_polygon_640x360']), 4)
        self.assertEqual(recorded_ground(dispatch_row)[1].status, 'candidate')
        self.assertEqual(query_frames(task='e_bike', role='direct_target',
                                      frame_id=result['frame_id'], database=self.database)[0]['status'], 'eligible')

    def test_missing_task_geometry_and_changed_bytes_cannot_write(self):
        review = self.review()
        del review['task_geometry']['waste_sticker']
        with self.assertRaisesRegex(ValueError, 'matching task labels/geometry'):
            ingest_review_batch([review], database=self.database)
        review = self.review()
        review['content_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Image bytes changed'):
            ingest_review_batch([review], database=self.database)
        conn = sqlite3.connect(self.database)
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM observations WHERE kind='objects'").fetchone()[0], 0)
        finally:
            conn.close()

    def test_sidecar_links_only_current_machine_evidence(self):
        review = self.review()
        review_file = Path(self.temp.name) / 'review.jsonl'
        review_file.write_text(json.dumps(review) + '\n', encoding='utf8')
        pool_file = Path(self.temp.name) / 'pool.json'
        pool_file.write_text(json.dumps({'schema_version': 1, 'role': 'shared_background_pool',
                                         'records': [{'image': self.path}]}), encoding='utf8')
        ingest_review_batch([review], database=self.database)
        index = build_machine_review_index(review_file, pool_manifest=pool_file, database=self.database)
        self.assertEqual({row['task'] for row in index['records']},
                         {'waste_bin_state', 'waste_sticker', 'fire'})
        fire = next(row for row in index['records'] if row['task'] == 'fire')
        self.assertNotIn('empty_negative', {item['role'] for item in fire['decisions']})
        self.assertEqual(index['excluded'], [])
        sidecar = Path(self.temp.name) / 'machine_index.json'
        publish_machine_review_index(review_file, pool_manifest=pool_file,
                                     database=self.database, output=sidecar)
        self.assertEqual(check_machine_review_index(sidecar, database=self.database), index)
        review_file.write_text(review_file.read_text(encoding='utf8') + '\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'sidecar is stale'):
            check_machine_review_index(sidecar, database=self.database)

    def test_sidecar_accepts_multiple_review_batches(self):
        review = self.review()
        ingest_review_batch([review], database=self.database)
        first = Path(self.temp.name) / 'first.jsonl'
        second = Path(self.temp.name) / 'second.jsonl'
        for path in (first, second):
            path.write_text(json.dumps(review) + '\n', encoding='utf8')
        pool = Path(self.temp.name) / 'pool.json'
        pool.write_text(json.dumps({'schema_version': 1, 'role': 'shared_background_pool',
                                    'records': [{'image': self.path}]}), encoding='utf8')
        index = build_machine_review_index([first, second], pool_manifest=pool, database=self.database)
        self.assertEqual(len(index['review_manifests']), 2)
        self.assertEqual(len(index['records']), 3)
        self.assertEqual(len(index['excluded']), 3)
        self.assertEqual({row['reason'] for row in index['excluded']}, {'superseded_review_row'})


if __name__ == '__main__':
    unittest.main()
