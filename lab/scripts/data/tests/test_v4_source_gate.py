"""Focused checks for source leakage and task review gates."""
import json
import tempfile
import unittest
import sys
from unittest import mock
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from lab.scripts.data.v4_source_gate import audit, development_duplicate_pairs, development_exposure, provenance
from lab.scripts.data import v4_source_gate as gate


class SourceGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        Image.new('RGB', (640, 480), 'blue').save(self.root / 'image.jpg')
        (self.root / 'positive.txt').write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf8')
        (self.root / 'negative.txt').write_text('', encoding='utf8')
        (self.root / 'person.yaml').write_text('names: [community, non_community]\n', encoding='utf8')
        (self.root / 'meter.yaml').write_text('names: [meter]\n', encoding='utf8')
        (self.root / 'review.json').write_text('{}', encoding='utf8')
        policy = self.root / 'lab/scripts/media/media_index_lifecycle.json'
        policy.parent.mkdir(parents=True)
        policy.write_text(json.dumps({'schema_version': 1, 'rules': []}), encoding='utf8')
        self.row = {'id': 'sample', 'image': 'image.jpg', 'source_group': 'video',
                    'source_kind': 'photo', 'generation_method': 'camera_capture',
                    'episode_group': 'episode_1', 'parent_ids': [], 'split': 'train',
                    'lineage_review': 'confirmed', 'near_duplicate_review': 'confirmed',
                    'review_evidence': 'review.json', 'selected_tasks': ['person'],
                    'tasks': {'person': {'state': 'complete_positive', 'label': 'positive.txt',
                                         'review_evidence': 'review.json'}}}
        self.data = {'samples': [self.row], 'near_duplicate_pairs': [], 'exposed_images': [],
                     'models': {'person': {'data_yaml': 'person.yaml'}}}

    def tearDown(self):
        self.temp.cleanup()

    def test_reviewed_sample_and_invalid_label(self):
        self.assertEqual(audit(self.root, self.data)['status'], 'passed')
        (self.root / 'positive.txt').write_text('2 0.5 0.5 0.2 0.2\n', encoding='utf8')
        self.assertEqual(audit(self.root, self.data)['status'], 'failed')

    def test_partial_and_negative_cannot_hide_positive(self):
        self.row['tasks']['person']['state'] = 'partial'
        self.assertEqual(audit(self.root, self.data)['status'], 'failed')
        self.row['tasks']['person']['state'] = 'complete_negative'
        self.assertEqual(audit(self.root, self.data)['status'], 'failed')
        self.row['tasks']['person']['label'] = 'negative.txt'
        self.assertEqual(audit(self.root, self.data)['status'], 'passed')

    def test_known_exposure_propagates_even_if_parent_omitted(self):
        known = {'samples': [{'id': 'image.jpg', 'parent_ids': ['ancestor']},
                             {'id': 'ancestor', 'parent_ids': []}],
                 'near_duplicate_pairs': [['ancestor', 'exposed']],
                 'exposed_images': ['exposed'],
                 'canonical_models': {'person': {'names': ['community', 'non_community']}}}
        (self.root / 'exposed').write_bytes(b'other')
        (self.root / 'ancestor').write_bytes(b'ancestor')
        self.assertEqual(audit(self.root, self.data, known)['status'], 'failed')

    def test_shared_parent_composites_cross_splits_with_recorded_reuse(self):
        Image.new('RGB', (32, 32), 'yellow').save(self.root / 'parent.png')
        (self.root / 'source.mp4').write_bytes(b'fixture video')
        self.row.update(source_kind='derived_image', generation_method='composite',
                        parent_ids=['parent'], parent_video='source.mp4', segment_id='segment_1',
                        physical_assets=['standee_1'], source_assets=['texture_1'],
                        image_processing={'resize': '640x480', 'compression': 'jpeg90', 'blur': 'none'})
        self.data['samples'].append({'id': 'parent', 'image': 'parent.png',
                                     'parent_ids': [], 'split': 'unassigned'})
        for split, color in (('val', 'red'), ('test', 'green')):
            Image.new('RGB', (640, 480), color).save(self.root / (split + '.jpg'))
            (self.root / (split + '.txt')).write_text('', encoding='utf8')
            self.data['samples'].append(dict(self.row, id=split, image=split + '.jpg', split=split,
                tasks={'person': {'state': 'complete_negative', 'label': split + '.txt',
                                  'review_evidence': 'review.json'}}))
        result = audit(self.root, self.data)
        self.assertEqual(result['status'], 'passed', result['errors'])
        reuse = {item['relation']: item for item in result['source_reuse']}
        for field in ('lineage_group', 'parent_ids', 'parent_video', 'episode_group',
                      'segment_id', 'physical_assets', 'source_assets'):
            self.assertEqual(reuse[field]['splits'], ['test', 'train', 'val'])
        self.assertEqual(reuse['parent_ids']['source'], 'parent')
        self.assertEqual(reuse['parent_ids']['sample_ids'],
                         {'test': ['test'], 'train': ['sample'], 'val': ['val']})
        self.assertEqual(provenance(self.data, 'person', result)['source_reuse'], result['source_reuse'])

    def test_diagnostic_test_rejected_but_regression_retained(self):
        self.row['split'] = 'test'
        self.data['exposed_images'] = ['image.jpg']
        result = audit(self.root, self.data)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any('已接触诊断图' in error for error in result['errors']))
        self.row['split'] = 'regression'
        result = audit(self.root, self.data)
        self.assertEqual(result['status'], 'pending')
        self.assertEqual(result['counts'], {'regression': 1})
        self.assertEqual(result['errors'], [])

    def test_unknown_review_and_empty_selection(self):
        self.row['near_duplicate_review'] = 'pending'
        self.assertEqual(audit(self.root, self.data)['status'], 'failed')
        self.row['split'] = 'unassigned'
        self.assertEqual(audit(self.root, self.data)['status'], 'pending')

    def test_development_exposure_requires_evidence_and_propagates_video_time(self):
        path = self.root / 'datasets/annotated/site_capture_20261112/v4_development_exposure.json'
        path.parent.mkdir(parents=True)
        (self.root / 'source.mp4').write_bytes(b'fixture')
        (self.root / 'artwork.png').write_bytes(b'fixture artwork')
        ledger = {'schema_version': 1, 'items': [{'image': 'image.jpg', 'roles': ['direct_background'],
                   'reasons': ['used in regression'], 'evidence': ['review.json'],
                   'parent_video': 'source.mp4', 'source_time_seconds': 1.25},
                   {'image': 'artwork.png', 'roles': ['direct_foreground_source'],
                    'reasons': ['shared foreground'], 'evidence': ['review.json'],
                    'parent_video': None, 'source_time_seconds': None}]}
        path.write_text(json.dumps(ledger), encoding='utf8')
        images, videos, evidence = development_exposure(self.root)
        self.assertEqual(images, {'image.jpg'})
        self.assertEqual(videos[0]['seconds'], 1.25)
        self.assertEqual(evidence, ['review.json'])
        ledger['items'][0]['evidence'] = ['missing.json']
        path.write_text(json.dumps(ledger), encoding='utf8')
        with self.assertRaises(ValueError):
            development_exposure(self.root)
        ledger['items'][0]['evidence'] = ['review.json']
        ledger['items'][0]['roles'] = ['unknown_role']
        path.write_text(json.dumps(ledger), encoding='utf8')
        with self.assertRaises(ValueError):
            development_exposure(self.root)

    def test_missing_history_retains_claims_and_audit_fails_closed(self):
        path = self.root / gate.DEVELOPMENT_EXPOSURE
        path.parent.mkdir(parents=True)
        ledger = {'schema_version': 1, 'items': [{'image': 'missing-image.png',
                  'roles': ['direct_background'], 'reasons': ['historical use'],
                  'evidence': ['missing-evidence.json'], 'parent_video': 'missing-video.mp4',
                  'source_time_seconds': 1.25}],
                  'near_duplicate_pairs': [{'a': 'missing-image.png', 'b': 'image.jpg',
                  'review_status': 'confirmed', 'evidence': 'missing-pair-evidence.json'}]}
        path.write_text(json.dumps(ledger), encoding='utf8')
        with self.assertRaises(ValueError):
            development_exposure(self.root)
        missing = []
        images, frames, evidence = development_exposure(self.root, missing_paths=missing)
        pairs, pair_evidence = development_duplicate_pairs(self.root, missing_paths=missing)
        self.assertEqual(images, {'missing-image.png'})
        self.assertEqual(frames[0]['video'], 'missing-video.mp4')
        self.assertEqual(pairs, [['image.jpg', 'missing-image.png']])
        self.assertEqual((evidence, pair_evidence), ([], []))
        self.assertEqual({item['kind'] for item in missing},
                         {'exposure_image', 'exposure_evidence', 'exposure_parent_video',
                          'duplicate_image', 'duplicate_evidence'})
        known = {'samples': [], 'near_duplicate_pairs': pairs, 'exposed_images': list(images),
                 'canonical_models': {'person': {'names': ['community', 'non_community']}},
                 'missing_exposure_paths': missing}
        result = audit(self.root, self.data, known)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any('历史文件/证据缺失' in error for error in result['errors']))

    def test_missing_canonical_header_cannot_pass_source_audit(self):
        known = {'samples': [], 'near_duplicate_pairs': [], 'exposed_images': [],
                 'canonical_models': {'person': None}}
        result = audit(self.root, self.data, known)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any('当前类别契约来源缺失' in error for error in result['errors']))

    def test_queue_keeps_missing_canonical_headers_unknown_until_available(self):
        script = self.root / 'lab/features/meter/training/scripts/generate_meter_synthetic.py'
        script.parent.mkdir(parents=True)
        script.write_text('KEYPOINTS = ' + repr([str(i) for i in range(8)]), encoding='utf8')
        values = {gate.INVENTORY: {'items': []}, gate.SEGMENTS: {'segments': []},
                  gate.SELECTION: {'items': []}, gate.EXPOSURE: {'items': []},
                  gate.DUPLICATES: {'candidate_pairs': []},
                  gate.REVIEW_SUMMARY: {'human_confirmations': []},
                  'config/perception_tasks.json': {'tasks': [{'id': 'waste_bin', 'waste_items': ['object']}]}}
        with mock.patch.object(gate, 'read_json', side_effect=lambda root, path: values[path]), \
                mock.patch.object(gate, 'read_yaml', return_value={'data': 'datasets/old.yaml'}), \
                mock.patch.object(gate, 'SITE_SELECTIONS', ()), mock.patch.object(gate, 'ANNOTATIONS', ()), \
                mock.patch.object(gate, 'unavailable_history', return_value=([], [])), \
                mock.patch.object(gate, 'development_exposure', return_value=(set(), [], [])), \
                mock.patch.object(gate, 'development_duplicate_pairs', return_value=([], [])):
            queue = gate._build_queue(self.root)
            self.assertEqual(set(queue['canonical_models_unavailable']), set(gate.MODELS))
            self.assertTrue(all(value is None for value in queue['canonical_models'].values()))
            header = self.root / 'datasets/old.yaml'
            header.parent.mkdir(parents=True, exist_ok=True)
            header.write_text('names: [object]\n', encoding='utf8')
            queue = gate._build_queue(self.root)
            self.assertEqual(queue['canonical_models_unavailable'], {})
            self.assertEqual(queue['canonical_models']['e_bike']['names'], ['object'])
            self.assertEqual(queue['canonical_models']['meter']['keypoint_names'], [str(i) for i in range(8)])

    def test_queue_inputs_adds_canonical_header_when_it_reappears(self):
        inputs = (gate.INVENTORY, gate.SEGMENTS, gate.SELECTION, *gate.ANNOTATIONS,
                  gate.REVIEW_SUMMARY, gate.EXPOSURE, gate.DUPLICATES, *gate.SITE_SELECTIONS,
                  gate.DEVELOPMENT_EXPOSURE, 'config/perception_tasks.json')
        for value in inputs:
            path = self.root / value
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('image\n' if path.suffix == '.csv' else '{}', encoding='utf8')
        with mock.patch.object(gate, 'read_yaml', return_value={'data': 'datasets/old.yaml'}):
            _, before, _ = gate.queue_inputs(self.root)
            self.assertNotIn(self.root / 'datasets/old.yaml', before)
            header = self.root / 'datasets/old.yaml'
            header.write_text('names: [object]\n', encoding='utf8')
            _, after, _ = gate.queue_inputs(self.root)
            self.assertIn(header, after)

    def test_lifecycle_excluded_history_is_retained_but_not_compared(self):
        from lab.training_common.camera_tone import source_role
        inputs = (gate.INVENTORY, gate.SEGMENTS, gate.SELECTION, *gate.ANNOTATIONS,
                  gate.REVIEW_SUMMARY, gate.EXPOSURE, gate.DUPLICATES, *gate.SITE_SELECTIONS,
                  gate.DEVELOPMENT_EXPOSURE, 'config/perception_tasks.json')
        for value in inputs:
            path = self.root / value
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('image\n' if path.suffix == '.csv' else '{}', encoding='utf8')
        old = 'Archive/technical/datasets/old.png'
        path = self.root / old
        path.parent.mkdir(parents=True)
        path.write_bytes((self.root / 'image.jpg').read_bytes())
        policy = {'schema_version': 1, 'rules': [{'path': 'Archive/technical/datasets',
                  'scope': 'media_subtree', 'state': 'inactive', 'reason': 'technical historical datasets'}]}
        (self.root / 'lab/scripts/media/media_index_lifecycle.json').write_text(json.dumps(policy), encoding='utf8')
        ledger = {'schema_version': 1, 'items': [{'image': old, 'roles': ['regression_image'],
                   'reasons': ['historical'], 'evidence': ['review.json']}],
                  'near_duplicate_pairs': [{'a': old, 'b': 'image.jpg', 'evidence': 'review.json',
                                            'review_status': 'confirmed'}]}
        (self.root / gate.DEVELOPMENT_EXPOSURE).write_text(json.dumps(ledger), encoding='utf8')
        with mock.patch.object(gate, 'read_yaml', return_value={'data': 'datasets/old.yaml'}):
            media, procedural, _ = gate.queue_inputs(self.root)
        self.assertNotIn(path, media)
        self.assertIn(self.root / 'image.jpg', media)
        self.assertIn(self.root / 'lab/scripts/media/media_index_lifecycle.json', procedural)
        unavailable, _ = gate.unavailable_history(self.root)
        self.assertEqual({entry['role'] for entry in unavailable}, {'regression_image', 'near_duplicate_a'})
        self.assertTrue(all(entry['source'] == gate.DEVELOPMENT_EXPOSURE and entry['record_index'] == 0
                            for entry in unavailable))
        known = {'exposed_images': [old], 'exposed_video_frames': [],
                 'unavailable_exposure_paths': unavailable, 'exposure_status': 'incomplete'}
        self.assertEqual(source_role(self.root, self.root / 'image.jpg', known)[0], 'unknown')
        self.assertEqual(source_role(self.root, path, known)[0], 'regression')
        known.update(samples=[], near_duplicate_pairs=[], canonical_models={'person': {'names': ['community', 'non_community']}})
        self.assertEqual(audit(self.root, self.data, known)['status'], 'failed')
    def test_development_near_duplicate_evidence_links_selected_splits(self):
        path = self.root / 'datasets/annotated/site_capture_20261112/v4_development_exposure.json'
        path.parent.mkdir(parents=True)
        Image.new('RGB', (640, 480), 'red').save(self.root / 'other.jpg')
        (self.root / 'other.txt').write_text('', encoding='utf8')
        ledger = {'near_duplicate_pairs': [{'a': 'image.jpg', 'b': 'other.jpg',
                   'review_status': 'confirmed', 'evidence': 'review.json'}]}
        path.write_text(json.dumps(ledger), encoding='utf8')
        pairs, evidence = development_duplicate_pairs(self.root)
        self.assertEqual(evidence, ['review.json'])
        other = dict(self.row, id='second', image='other.jpg', split='test', episode_group='other',
                     tasks={'person': {'state': 'complete_negative', 'label': 'other.txt',
                                       'review_evidence': 'review.json'}})
        self.data['samples'].append(other)
        known = {'samples': [], 'near_duplicate_pairs': pairs, 'exposed_images': [],
                 'canonical_models': {'person': {'names': ['community', 'non_community']}}}
        result = audit(self.root, self.data, known)
        self.assertEqual(result['status'], 'passed', result['errors'])
        self.assertTrue(any(item['relation'] == 'lineage_group' for item in result['source_reuse']))
        known['exposed_images'] = ['image.jpg']
        self.assertTrue(any('已接触诊断图' in e for e in audit(self.root, self.data, known)['errors']))

    def test_missing_pose_shape_rejected(self):
        self.row['tasks'] = {'meter': {'state': 'complete_negative', 'label': 'negative.txt',
                                        'review_evidence': 'review.json'}}
        self.row['selected_tasks'] = ['meter']
        self.data['models']['meter'] = {'data_yaml': 'meter.yaml'}
        self.assertEqual(audit(self.root, self.data)['status'], 'failed')

    def test_missing_evidence_and_unknown_generation_rejected(self):
        self.row['review_evidence'] = 'missing.json'
        self.assertEqual(audit(self.root, self.data)['status'], 'failed')
        self.row['review_evidence'] = 'review.json'
        self.row['generation_method'] = 'invented'
        self.assertEqual(audit(self.root, self.data)['status'], 'failed')

    def test_renamed_byte_copy_of_exposed_rejected(self):
        (self.root / 'diagnostic.jpg').write_bytes((self.root / 'image.jpg').read_bytes())
        known = {'samples': [], 'near_duplicate_pairs': [], 'exposed_images': ['diagnostic.jpg'],
                 'canonical_models': {'person': {'names': ['community', 'non_community']}}}
        result = audit(self.root, self.data, known)
        self.assertTrue(any('字节相同' in error for error in result['errors']))

    def test_unassigned_renamed_ancestor_of_diagnostic_is_rejected(self):
        Image.new('RGB', (32, 32), 'red').save(self.root / 'diagnostic.png')
        (self.root / 'renamed.png').write_bytes((self.root / 'diagnostic.png').read_bytes())
        Image.new('RGB', (32, 32), 'green').save(self.root / 'intermediate.png')
        self.data['samples'].extend([
            {'id': 'parent', 'image': 'renamed.png', 'parent_ids': [], 'split': 'unassigned'},
            {'id': 'intermediate', 'image': 'intermediate.png', 'parent_ids': ['parent'], 'split': 'unassigned'},
        ])
        self.row['parent_ids'] = ['intermediate']
        known = {'samples': [], 'near_duplicate_pairs': [], 'exposed_images': ['diagnostic.png'],
                 'canonical_models': {'person': {'names': ['community', 'non_community']}}}
        result = audit(self.root, self.data, known)
        self.assertTrue(any('已接触诊断图或其派生' in error for error in result['errors']))

    def test_unassigned_copy_of_known_diagnostic_near_duplicate_is_rejected(self):
        Image.new('RGB', (32, 32), 'red').save(self.root / 'diagnostic.png')
        Image.new('RGB', (32, 32), 'green').save(self.root / 'near.png')
        (self.root / 'alias.png').write_bytes((self.root / 'near.png').read_bytes())
        self.data['samples'].append({'id': 'parent', 'image': 'alias.png', 'parent_ids': [],
                                     'split': 'unassigned'})
        self.row['parent_ids'] = ['parent']
        known = {'samples': [], 'near_duplicate_pairs': [['diagnostic.png', 'near.png']],
                 'exposed_images': ['diagnostic.png'],
                 'canonical_models': {'person': {'names': ['community', 'non_community']}}}
        self.assertTrue(any('已接触诊断图或其派生' in e for e in audit(self.root, self.data, known)['errors']))

    def test_unassigned_duplicate_parents_allow_distinct_children_across_splits(self):
        Image.new('RGB', (32, 32), 'red').save(self.root / 'parent.png')
        (self.root / 'parent_copy.png').write_bytes((self.root / 'parent.png').read_bytes())
        Image.new('RGB', (640, 480), 'green').save(self.root / 'other.jpg')
        (self.root / 'other.txt').write_text('', encoding='utf8')
        self.row['parent_ids'] = ['parent_a']
        other = dict(self.row, id='other', image='other.jpg', split='test', episode_group='other',
                     parent_ids=['parent_b'], tasks={'person': {'state': 'complete_negative',
                     'label': 'other.txt', 'review_evidence': 'review.json'}})
        self.data['samples'].extend([other,
            {'id': 'parent_a', 'image': 'parent.png', 'parent_ids': [], 'split': 'unassigned'},
            {'id': 'parent_b', 'image': 'parent_copy.png', 'parent_ids': [], 'split': 'unassigned'}])
        known = {'samples': [], 'near_duplicate_pairs': [], 'exposed_images': [],
                 'canonical_models': {'person': {'names': ['community', 'non_community']}}}
        self.assertEqual(audit(self.root, self.data, known)['status'], 'passed')
        self.assertEqual(audit(self.root, self.data)['status'], 'passed')
        result = audit(self.root, self.data, known)
        self.assertEqual(result['status'], 'passed')
        exported = provenance(self.data, 'person', result)['records']
        self.assertEqual(exported[0]['near_duplicate_group'], exported[1]['near_duplicate_group'])

    def test_alias_closure_reaches_unselected_ancestor(self):
        Image.new('RGB', (32, 32), 'red').save(self.root / 'parent.png')
        (self.root / 'alias.png').write_bytes((self.root / 'parent.png').read_bytes())
        Image.new('RGB', (32, 32), 'green').save(self.root / 'diagnostic.png')
        self.row['parent_ids'] = ['parent_a']
        self.data['samples'].extend([
            {'id': 'parent_a', 'image': 'parent.png', 'parent_ids': [], 'split': 'unassigned'},
            {'id': 'parent_b', 'image': 'alias.png', 'parent_ids': ['diagnostic'], 'split': 'unassigned'},
            {'id': 'diagnostic', 'image': 'diagnostic.png', 'parent_ids': [], 'split': 'unassigned'}])
        known = {'samples': [], 'near_duplicate_pairs': [], 'exposed_images': ['diagnostic.png'],
                 'canonical_models': {'person': {'names': ['community', 'non_community']}}}
        self.assertTrue(any('已接触诊断图或其派生' in e for e in audit(self.root, self.data, known)['errors']))

    def test_alias_of_known_near_duplicate_parent_records_cross_split_reuse(self):
        Image.new('RGB', (32, 32), 'red').save(self.root / 'a.png')
        Image.new('RGB', (32, 32), 'green').save(self.root / 'b.png')
        (self.root / 'alias.png').write_bytes((self.root / 'a.png').read_bytes())
        Image.new('RGB', (640, 480), 'green').save(self.root / 'other.jpg')
        (self.root / 'other.txt').write_text('', encoding='utf8')
        self.row['parent_ids'] = ['parent_a']
        other = dict(self.row, id='other', image='other.jpg', split='test', episode_group='other',
                     parent_ids=['parent_b'], tasks={'person': {'state': 'complete_negative',
                     'label': 'other.txt', 'review_evidence': 'review.json'}})
        self.data['samples'].extend([other,
            {'id': 'parent_a', 'image': 'alias.png', 'parent_ids': [], 'split': 'unassigned'},
            {'id': 'parent_b', 'image': 'b.png', 'parent_ids': [], 'split': 'unassigned'}])
        known = {'samples': [], 'near_duplicate_pairs': [['a.png', 'b.png']], 'exposed_images': [],
                 'canonical_models': {'person': {'names': ['community', 'non_community']}}}
        result = audit(self.root, self.data, known)
        self.assertEqual(result['status'], 'passed', result['errors'])
        self.assertTrue(any(item['relation'] == 'lineage_group' and item['splits'] == ['test', 'train']
                            for item in result['source_reuse']))

    def test_untouched_parent_and_unrelated_unassigned_do_not_fail(self):
        Image.new('RGB', (32, 32), 'red').save(self.root / 'parent.png')
        Image.new('RGB', (32, 32), 'green').save(self.root / 'diagnostic.png')
        self.data['samples'].extend([
            {'id': 'parent', 'image': 'parent.png', 'parent_ids': [], 'split': 'unassigned'},
            {'id': 'unrelated', 'image': 'not-created.png', 'parent_ids': [], 'split': 'unassigned'}])
        self.row['parent_ids'] = ['parent']
        known = {'samples': [], 'near_duplicate_pairs': [], 'exposed_images': ['diagnostic.png'],
                 'canonical_models': {'person': {'names': ['community', 'non_community']}}}
        self.assertEqual(audit(self.root, self.data, known)['status'], 'passed')
        self.data['samples'][1]['image'] = 'missing-parent.png'
        self.assertTrue(any('无法核查' in error for error in audit(self.root, self.data, known)['errors']))

    def test_unassigned_video_parent_enforces_contact_time_for_alias(self):
        Image.new('RGB', (32, 32), 'red').save(self.root / 'source_frame.png')
        (self.root / 'alias.png').write_bytes((self.root / 'source_frame.png').read_bytes())
        (self.root / 'video.mp4').write_bytes(b'fixture')
        self.data['samples'].append({'id': 'parent', 'image': 'alias.png', 'split': 'unassigned',
                                     'parent_ids': [], 'source_kind': 'photo'})
        self.row['parent_ids'] = ['parent']
        known = {'samples': [{'id': 'source', 'image': 'source_frame.png', 'parent_ids': [],
                              'source_kind': 'video_frame', 'parent_video': 'video.mp4',
                              'source_time_seconds': 10.1}], 'near_duplicate_pairs': [],
                 'exposed_images': [], 'exposed_video_frames': [
                     {'video': 'video.mp4', 'seconds': 10.0, 'image': 'contact.png'}],
                 'canonical_models': {'person': {'names': ['community', 'non_community']}}}
        self.assertTrue(any('已接触诊断图或其派生' in e for e in audit(self.root, self.data, known)['errors']))
        known['samples'][0]['source_time_seconds'] = 11.0
        self.assertEqual(audit(self.root, self.data, known)['status'], 'passed')
        del known['samples'][0]['source_time_seconds']
        self.assertTrue(any('须记录源时间' in e for e in audit(self.root, self.data, known)['errors']))

    def test_selected_known_video_alias_cannot_falsify_timestamp(self):
        (self.root / 'video.mp4').write_bytes(b'fixture')
        (self.root / 'original.jpg').write_bytes((self.root / 'image.jpg').read_bytes())
        self.row.update(source_kind='video_frame', generation_method='provided_video_frame',
                        parent_video='video.mp4', segment_id='s', episode_group='e', source_time_seconds=30.0)
        known = {'samples': [{'id': 'original.jpg', 'image': 'original.jpg', 'parent_ids': [],
                              'source_kind': 'video_frame', 'generation_method': 'provided_video_frame',
                              'source_group': 'video', 'parent_video': 'video.mp4', 'segment_id': 's',
                              'episode_group': 'e', 'source_time_seconds': 10.1}], 'near_duplicate_pairs': [],
                 'exposed_images': [], 'exposed_video_frames': [
                     {'video': 'video.mp4', 'seconds': 10.0, 'image': 'contact.jpg'}],
                 'canonical_models': {'person': {'names': ['community', 'non_community']}}}
        result = audit(self.root, self.data, known)
        self.assertTrue(any('已登记原图不符' in e for e in result['errors']))
        self.assertTrue(any('已接触诊断图或其派生' in e for e in result['errors']))

    def test_renamed_identical_bytes_cross_split_rejected(self):
        (self.root / 'copy.jpg').write_bytes((self.root / 'image.jpg').read_bytes())
        (self.root / 'other.txt').write_text('', encoding='utf8')
        other = dict(self.row, id='other', image='copy.jpg', split='test', episode_group='other',
                     tasks={'person': {'state': 'complete_negative', 'label': 'other.txt',
                                       'review_evidence': 'review.json'}})
        self.data['samples'].append(other)
        self.assertTrue(any('字节相同' in error for error in audit(self.root, self.data)['errors']))

    def test_selected_class_order_matches_canonical(self):
        known = {'samples': [], 'near_duplicate_pairs': [], 'exposed_images': [],
                 'canonical_models': {'person': {'names': ['non_community', 'community']}}}
        self.assertEqual(audit(self.root, self.data, known)['status'], 'failed')

    def test_meter_point_order_matches_canonical(self):
        (self.root / 'meter.yaml').write_text('names: [meter]\nkpt_shape: [8, 3]\n', encoding='utf8')
        self.row['tasks'] = {'meter': {'state': 'complete_negative', 'label': 'negative.txt',
                                      'review_evidence': 'review.json'}}
        self.row['selected_tasks'] = ['meter']
        self.data['models']['meter'] = {'data_yaml': 'meter.yaml', 'keypoint_names': ['wrong'] * 8}
        known = {'samples': [], 'near_duplicate_pairs': [], 'exposed_images': [],
                 'canonical_models': {'meter': {'names': ['meter'], 'keypoint_names': [str(i) for i in range(8)]}}}
        result = audit(self.root, self.data, known)
        self.assertTrue(any('八点名称或顺序' in error for error in result['errors']))

    def test_provenance_has_packager_fields_and_retains_candidate_for_other_task(self):
        self.row['tasks']['e_bike'] = {'state': 'candidate'}
        self.row['source_time_seconds'] = 7.213
        self.assertEqual(audit(self.root, self.data)['status'], 'passed')
        known = {'samples': [], 'near_duplicate_pairs': []}
        record = provenance(self.data, 'person', audit(self.root, self.data))['records'][0]
        self.assertEqual(record['label'], 'positive.txt')
        self.assertEqual(record['source_id'], 'video')
        self.assertEqual(record['source_time_seconds'], 7.213)
        self.assertEqual(record['label_status'], 'reviewed')
        self.assertEqual(record['review_evidence'], 'review.json')
        self.assertIsNone(record['parent_sample'])
        self.assertTrue(record['near_duplicate_group'])

    def test_reviewed_strata_round_trip_and_invalid_fields(self):
        task = self.row['tasks']['person']
        task.update(strata_review='confirmed', strata={
            'difficulty': 'hard', 'conditions': ['glare', 'occlusion'], 'target_states': []})
        self.assertEqual(audit(self.root, self.data)['status'], 'passed')
        record = provenance(self.data, 'person', audit(self.root, self.data))['records'][0]
        self.assertEqual(record['strata_review'], 'confirmed')
        self.assertEqual(record['strata'], task['strata'])
        task['strata']['conditions'] = ['glare', 'glare']
        self.assertEqual(audit(self.root, self.data)['status'], 'failed')
        task['strata']['conditions'] = ['glare']
        task['strata']['difficulty'] = ['hard']
        self.assertEqual(audit(self.root, self.data)['status'], 'failed')
        task['strata']['difficulty'] = 'hard'
        task['strata']['conditions'] = [f'condition_{i}' for i in range(65)]
        self.assertEqual(audit(self.root, self.data)['status'], 'failed')

    def test_reviewed_resolution_must_match_decoded_image(self):
        self.row['image_processing'] = {'resolution_review': 'confirmed', 'source_size': [640, 480],
                                        'output_size': [640, 480], 'resampling': 'none', 'clarity': 'original'}
        self.assertEqual(audit(self.root, self.data)['status'], 'passed')
        self.row['image_processing']['output_size'] = [640, 360]
        self.assertTrue(any('分辨率或清晰度' in e for e in audit(self.root, self.data)['errors']))
        self.row['image_processing'].update(output_size=[640, 480], source_size=[1920, 1080],
                                            resampling='resampled', clarity='original')
        self.assertTrue(any('分辨率或清晰度' in e for e in audit(self.root, self.data)['errors']))
        self.row['image_processing']['clarity'] = 'modified'
        self.assertEqual(audit(self.root, self.data)['status'], 'passed')
        self.row['image_processing'].update(origin_review='confirmed', parent_origin='unverified')
        self.row.update(source_kind='derived_image', generation_method='augmentation')
        self.assertTrue(any('parent_origin' in e for e in audit(self.root, self.data)['errors']))

    def test_source_kind_method_and_known_frame_cannot_be_disguised(self):
        self.row['source_kind'] = 'invented'
        self.assertTrue(any('来源种类' in e for e in audit(self.root, self.data)['errors']))
        self.row['source_kind'] = 'video_frame'
        self.assertTrue(any('来源种类' in e for e in audit(self.root, self.data)['errors']))
        self.row['source_kind'] = 'photo'
        known = {'samples': [{'id': 'original', 'image': 'image.jpg', 'source_kind': 'frame_sequence',
                              'generation_method': 'provided_video_frame', 'source_group': 'video',
                              'parent_video': 'source.mp4', 'segment_id': 's1', 'episode_group': 'e1',
                              'parent_ids': []}], 'near_duplicate_pairs': [], 'exposed_images': [],
                 'canonical_models': {'person': {'names': ['community', 'non_community']}}}
        self.assertTrue(any('声明来源种类' in e for e in audit(self.root, self.data, known)['errors']))

    def test_frame_requires_existing_video_event_and_segment(self):
        self.row.update(source_kind='video_frame', generation_method='provided_video_frame',
                        parent_video='missing.mp4', segment_id='s', episode_group='e')
        self.assertTrue(any('序列帧缺少' in e for e in audit(self.root, self.data)['errors']))
        (self.root / 'source.mp4').write_bytes(b'video fixture')
        self.row['parent_video'] = 'source.mp4'
        self.assertEqual(audit(self.root, self.data)['status'], 'passed')

    def test_site_frame_uses_logged_pts_not_fixed_fps_estimate(self):
        sequence = self.root / 'datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene'
        sequence.mkdir(parents=True)
        image = sequence / 'frame_000002.png'
        Image.new('RGB', (640, 360), 'blue').save(image)
        (sequence / 'ffmpeg.log').write_text(
            '[showinfo] n: 0 pts: 0 pts_time:0\n'
            '[showinfo] n: 1 pts: 19170 pts_time:0.213\n', encoding='utf8')
        video = self.root / 'datasets/raw/site_capture_20261112/videos/full_scene.mp4'
        video.parent.mkdir(parents=True)
        video.write_bytes(b'fixture')
        self.row.update(image=image.relative_to(self.root).as_posix(), source_kind='video_frame',
                        generation_method='provided_video_frame', parent_video=video.relative_to(self.root).as_posix(),
                        segment_id='s', episode_group='e', source_time_seconds=0.213)
        self.assertEqual(audit(self.root, self.data)['status'], 'passed')
        self.row['source_time_seconds'] = 1/30
        self.assertTrue(any('解码日志实际帧时间' in e for e in audit(self.root, self.data)['errors']))
        self.row['source_time_seconds'] = 0.213
        self.row['image'] = './' + image.relative_to(self.root).as_posix()
        self.assertTrue(any('规范工作空间相对路径' in e for e in audit(self.root, self.data)['errors']))

    def test_person_frame_uses_probe_display_order(self):
        sequence = self.root / 'datasets/annotated/site_capture_20261112/sequence_v2/person_standees'
        sequence.mkdir(parents=True)
        image = sequence / 'frame_000002.png'
        Image.new('RGB', (640, 360), 'blue').save(image)
        (sequence / 'frames_probe.json').write_text(
            '{"frames":[{"best_effort_timestamp_time":"0.0"},'
            '{"best_effort_timestamp_time":"0.213"}]}', encoding='utf8')
        video = self.root / 'datasets/raw/site_capture_20261112/videos/person_standees.mp4'
        video.parent.mkdir(parents=True)
        video.write_bytes(b'fixture')
        self.row.update(image=image.relative_to(self.root).as_posix(), source_kind='frame_sequence',
                        generation_method='provided_video_frame', parent_video=video.relative_to(self.root).as_posix(),
                        segment_id='s', episode_group='e', source_time_seconds=0.213)
        self.assertEqual(audit(self.root, self.data)['status'], 'passed')
        self.row['source_time_seconds'] = 1/30
        self.assertTrue(any('人物序列探测时间' in e for e in audit(self.root, self.data)['errors']))

    def test_unassigned_sequence_parent_checks_actual_pts_and_source(self):
        for which in ('site', 'person'):
            with self.subTest(sequence=which):
                if which == 'site':
                    sequence = self.root / 'datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene'
                    video = self.root / 'datasets/raw/site_capture_20261112/videos/full_scene.mp4'
                else:
                    sequence = self.root / 'datasets/annotated/site_capture_20261112/sequence_v2/person_standees'
                    video = self.root / 'datasets/raw/site_capture_20261112/videos/person_standees.mp4'
                sequence.mkdir(parents=True, exist_ok=True)
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b'fixture')
                image = sequence / 'frame_000002.png'
                Image.new('RGB', (640, 360), 'red').save(image)
                if which == 'site':
                    (sequence / 'ffmpeg.log').write_text('[showinfo] n: 1 pts: 19170 pts_time:0.213\n', encoding='utf8')
                else:
                    (sequence / 'frames_probe.json').write_text(
                        '{"frames":[{"best_effort_timestamp_time":"0.0"},'
                        '{"best_effort_timestamp_time":"0.213"}]}', encoding='utf8')
                parent = {'id': 'parent', 'image': image.relative_to(self.root).as_posix(),
                          'parent_ids': [], 'split': 'unassigned', 'source_kind': 'video_frame',
                          'generation_method': 'provided_video_frame',
                          'parent_video': video.relative_to(self.root).as_posix(), 'source_time_seconds': 0.213}
                self.data['samples'] = [self.row, parent]
                self.row['parent_ids'] = ['parent']
                known = {'samples': [], 'near_duplicate_pairs': [], 'exposed_images': [],
                         'exposed_video_frames': [{'video': parent['parent_video'], 'seconds': 2.0}],
                         'canonical_models': {'person': {'names': ['community', 'non_community']}}}
                self.assertEqual(audit(self.root, self.data, known)['status'], 'passed')
                parent['source_time_seconds'] = 1.0
                self.assertEqual(audit(self.root, self.data, known)['status'], 'failed')
                parent['source_time_seconds'] = 0.213
                parent['source_kind'] = 'photo'
                self.assertEqual(audit(self.root, self.data, known)['status'], 'failed')
                parent['source_kind'] = 'video_frame'
                known['exposed_video_frames'][0]['seconds'] = 0.3
                self.assertEqual(audit(self.root, self.data, known)['status'], 'failed')

    def test_site_diagnostic_frame_is_blocked_across_encodings(self):
        (self.root / 'source.mp4').write_bytes(b'video fixture')
        self.row.update(source_kind='video_frame', generation_method='provided_video_frame',
                        parent_video='source.mp4', segment_id='s', episode_group='e')
        known = {'samples': [], 'near_duplicate_pairs': [], 'exposed_images': [],
                 'exposed_video_frames': [{'video': 'source.mp4', 'seconds': 80.0, 'image': 'old.jpg'}],
                 'canonical_models': {'person': {'names': ['community', 'non_community']}}}
        self.assertTrue(any('须记录源时间' in e for e in audit(self.root, self.data, known)['errors']))
        self.row['source_time_seconds'] = 80.005
        self.assertTrue(any('0.25 秒缓冲内' in e for e in audit(self.root, self.data, known)['errors']))
        self.row['source_time_seconds'] = 80.20
        self.assertTrue(any('0.25 秒缓冲内' in e for e in audit(self.root, self.data, known)['errors']))
        self.row['source_time_seconds'] = 81.0
        self.assertEqual(audit(self.root, self.data, known)['status'], 'passed')

    def test_720p_rotation_limit_and_decode_failure(self):
        for size, expected in [((1280, 720), 'passed'), ((720, 1280), 'passed'),
                               ((1280, 721), 'failed'), ((721, 1280), 'failed'),
                               ((1281, 720), 'failed')]:
            with self.subTest(size=size):
                Image.new('RGB', size).save(self.root / 'image.jpg')
                self.assertEqual(audit(self.root, self.data)['status'], expected)
        (self.root / 'image.jpg').write_bytes(b'not an image')
        self.assertTrue(any('解码失败' in e for e in audit(self.root, self.data)['errors']))

    def test_render_provenance_records_scene_assets_and_processing(self):
        self.row.update(source_kind='blender_scene', generation_method='blender_render',
                        mother_scene='scene_1', source_assets=['model_A', 'texture_A'],
                        physical_assets=['object_A'],
                        image_processing={'resize': '640x480', 'compression': 'jpeg quality 90', 'blur': 'none'})
        self.assertEqual(audit(self.root, self.data)['status'], 'passed')
        record = provenance(self.data, 'person', audit(self.root, self.data))['records'][0]
        self.assertEqual(record['mother_scene'], 'scene_1')
        self.assertEqual(record['source_assets'], ['model_A', 'texture_A'])
        self.assertEqual(record['physical_assets'], ['object_A'])
        self.assertEqual(record['image_processing']['blur'], 'none')
        self.row['mother_scene'] = None
        self.assertTrue(any('母场景' in e for e in audit(self.root, self.data)['errors']))
        self.row['mother_scene'] = 'scene_1'
        self.row['image_processing'].pop('blur')
        self.assertTrue(any('模糊参数' in e for e in audit(self.root, self.data)['errors']))


class SourceReuseEntryTest(unittest.TestCase):
    """Exercise real SQLite guards around queue and source-export entrypoints."""

    def setUp(self):
        from lab.scripts.media.tests.test_media_index_sqlite import CacheTests
        fixture = CacheTests(methodName='runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.parts, self.image, self.call = fixture.root, fixture.parts, fixture.image, fixture.call
        shard = self.root / self.parts['datasets/raw/camera']['shard']
        item = json.loads(shard.read_text(encoding='utf8'))
        item.update(bytes=self.image.stat().st_size, modified_ns=self.image.stat().st_mtime_ns)
        shard.write_text(json.dumps(item) + '\n', encoding='utf8')
        fixtures = self.root / 'fixtures'
        fixtures.mkdir()
        self.source_config = fixtures / 'request.json'
        self.source_config.write_text('{"revision":1}', encoding='utf8')
        self.source_code = fixtures / 'generator.py'
        self.source_code.write_text('# fixture generator\n', encoding='utf8')
        self.assertEqual(self.call(['sync'])[0], 0)
        self.deps = mock.patch.object(gate, 'queue_inputs', return_value=(
            [self.image], [self.source_config], [self.source_code]))
        self.deps.start()
        self.addCleanup(self.deps.stop)

    def test_queue_reuses_persistent_result_and_only_rebuilds_changed_inputs(self):
        with mock.patch.object(gate, '_build_queue', return_value={'samples': [{'id': 'declared'}]}) as build:
            first = gate.build_queue(self.root)
            self.assertEqual(gate.build_queue(self.root), first)
            self.assertEqual(build.call_count, 1)
            self.source_config.write_text('{"revision":2,"new":true}', encoding='utf8')
            self.assertEqual(gate.build_queue(self.root), first)
            self.assertEqual(build.call_count, 2)

    def test_source_bootstrap_does_not_rewrite_and_missing_database_writes_nothing(self):
        out = self.root / 'datasets/annotated/source.json'
        with mock.patch.object(gate, 'ROOT', self.root), \
                mock.patch.object(gate, '_build_queue', return_value={'samples': []}) as build:
            self.assertEqual(gate.main(['bootstrap', '--output', 'datasets/annotated/source.json']), 0)
            before = (out.read_bytes(), out.stat().st_mtime_ns)
            self.assertEqual(gate.main(['bootstrap', '--output', 'datasets/annotated/source.json']), 0)
            self.assertEqual(before, (out.read_bytes(), out.stat().st_mtime_ns))
            self.assertEqual(build.call_count, 1)
            # The shared fixture binds this same real database; no mock admission.
            database = self.root / 'datasets/annotated/media_index/query_cache.sqlite3'
            database.rename(database.with_suffix('.disabled'))
            self.assertEqual(gate.main(['bootstrap', '--output', 'datasets/annotated/no_write.json']), 2)
            self.assertFalse((out.parent / 'no_write.json').exists())

    def test_provenance_export_reuses_output_and_rechecks_changed_selection(self):
        manifest = self.root / 'datasets/annotated/selection.json'
        timing_log = self.root / gate.SITE_SEQUENCE / 'ffmpeg.log'
        timing_log.parent.mkdir(parents=True)
        timing_log.write_text('fixture timestamps 1', encoding='utf8')
        selection = {'samples': [{'image': gate.SITE_SEQUENCE + '/frame_000001.png'}]}
        manifest.write_text(json.dumps(selection), encoding='utf8')
        output = self.root / 'datasets/annotated/provenance.json'
        argv = ['export-provenance', '--manifest', 'datasets/annotated/selection.json',
                '--model', 'person', '--output', 'datasets/annotated/provenance.json']
        with mock.patch.object(gate, 'ROOT', self.root), \
                mock.patch.object(gate, '_build_queue', return_value={'samples': []}) as build, \
                mock.patch.object(gate, 'audit', return_value={'status': 'passed', 'assigned': 1}) as audit_call, \
                mock.patch.object(gate, 'provenance', return_value={'records': [{'image': 'existing'}]}):
            self.assertEqual(gate.main(argv), 0)
            before = (output.read_bytes(), output.stat().st_mtime_ns)
            self.assertEqual(gate.main(argv), 0)
            self.assertEqual(before, (output.read_bytes(), output.stat().st_mtime_ns))
            self.assertEqual((build.call_count, audit_call.call_count), (1, 1))
            timing_log.write_text('fixture timestamps 2 changed', encoding='utf8')
            self.assertEqual(gate.main(argv), 0)
            self.assertEqual((build.call_count, audit_call.call_count), (1, 2))
            selection['revision'] = 2
            manifest.write_text(json.dumps(selection), encoding='utf8')
            self.assertEqual(gate.main(argv), 0)
            self.assertEqual((build.call_count, audit_call.call_count), (1, 3))


if __name__ == '__main__':
    unittest.main()
