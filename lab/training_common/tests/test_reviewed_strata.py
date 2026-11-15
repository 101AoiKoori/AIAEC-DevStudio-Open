"""Review-gated fixed-threshold strata and staged provenance contracts."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.data import records, reviewed_image_domain, stage_dataset
from engine.freeze_gate import _same_dataset
from engine.reporting import (class_detection_counts, diagnostics, joint_domain_condition_metrics, match_detections,
                              stratify_class_metrics, stratify_image_metrics)
from engine.runtime import run_context


def item(name, supervision='complete', state='complete_positive', **attributes):
    review = {'supervision': supervision, 'label_state': state, **attributes}
    return {'run_image': 'data/images/val/' + name, 'reviewed_strata': review}


def counts(name, truth, tp, fp, fn):
    return {'image': name, 'ground_truth': truth, 'predictions': tp+fp,
            'true_positives': tp, 'false_positives': fp, 'false_negatives': fn}


class ReviewedStrata(unittest.TestCase):
    def test_overlapping_conditions_unknown_and_excluded_never_become_negative(self):
        rows = [counts('a.png', 2, 1, 1, 1), counts('b.png', 0, 0, 1, 0),
                counts('c.png', 0, 0, 3, 0)]
        records = [item('a.png', difficulty='hard', conditions=['glare', 'occlusion']),
                   item('b.png', state='complete_negative', conditions=[]),
                   item('c.png', supervision='unreviewed')]
        groups, excluded = stratify_image_metrics(rows, records)
        lookup = {(g['dimension'], g['value'], g['value_status']): g for g in groups}
        self.assertEqual(lookup['conditions', 'glare', 'confirmed']['tp'], 1)
        self.assertEqual(lookup['conditions', 'occlusion', 'confirmed']['fn'], 1)
        self.assertEqual(lookup['conditions', 'none', 'confirmed_empty']['negative_images_with_false_positives'], 1)
        self.assertEqual(lookup['difficulty', 'unknown', 'missing']['negative_images'], 1)
        self.assertEqual(excluded, [{'image': 'c.png', 'reason': 'supervision_not_confirmed'}])
        self.assertIsNone(lookup['label_state', 'complete_negative', 'confirmed']['recall'])

    def test_duplicate_mismatch_and_state_disagreement(self):
        a = counts('a.png', 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            stratify_image_metrics([a, a], [item('a.png'), item('a.png')])
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            stratify_image_metrics([a], [item('other.png')])
        groups, excluded = stratify_image_metrics([a], [item('a.png')])
        self.assertEqual(groups, [])
        self.assertEqual(excluded[0]['reason'], 'label_state_disagrees_with_staged_label')
        with self.assertRaisesRegex(ValueError, 'duplicated reviewed stratum'):
            stratify_image_metrics([counts('a.png', 1, 1, 0, 0)],
                                   [item('a.png', conditions=['glare', 'glare'])])

    def test_empty_groups(self):
        self.assertEqual(stratify_image_metrics([], []), ([], []))

    def test_per_class_mixed_images_wrong_class_duplicates_and_missing_class(self):
        truth = [{'class_id': 0, 'xyxy': [0, 0, 10, 10]}, {'class_id': 1, 'xyxy': [20, 0, 30, 10]}]
        pred = [{'class_id': 1, 'xyxy': [0, 0, 10, 10]}, {'class_id': 1, 'xyxy': [20, 0, 30, 10]},
                {'class_id': 1, 'xyxy': [20, 0, 30, 10]}]
        matched, missed, false = match_detections(truth, pred, .5)
        first = class_detection_counts(truth, pred, matched, missed, false, 3)
        self.assertEqual(first[0], {'ground_truth': 1, 'predictions': 0, 'tp': 0, 'fp': 0, 'fn': 1})
        self.assertEqual(first[1], {'ground_truth': 1, 'predictions': 3, 'tp': 1, 'fp': 2, 'fn': 0})
        negative_pred = [{'class_id': 0, 'xyxy': [0, 0, 10, 10]}]
        negative = class_detection_counts([], negative_pred, [], [], [0], 3)
        rows = [counts('a.png', 2, 1, 2, 1), counts('b.png', 0, 0, 1, 0), counts('unknown.png', 0, 0, 1, 0)]
        records = [item('a.png', difficulty='hard', conditions=['glare', 'occlusion']),
                   item('b.png', state='complete_negative', difficulty='hard', conditions=[]),
                   item('unknown.png', supervision='unreviewed')]
        groups = stratify_class_metrics(rows, records, [first, negative, negative], ['one', 'two', 'absent'])
        group = {(g['dimension'], g['value'], g['class_id']): g for g in groups}
        cls0, cls1, cls2 = [group['difficulty', 'hard', i] for i in range(3)]
        self.assertEqual((cls0['images'], cls0['ground_truth'], cls0['tp'], cls0['fp'], cls0['fn']), (2, 1, 0, 1, 1))
        self.assertEqual((cls0['images_with_class'], cls0['images_without_class_with_false_positives']), (1, 1))
        self.assertEqual((cls1['ground_truth'], cls1['tp'], cls1['fp'], cls1['fn']), (1, 1, 2, 0))
        self.assertEqual((cls1['precision_denominator'], cls1['recall_denominator']), (3, 1))
        self.assertIsNone(cls2['recall']); self.assertIsNone(cls2['precision'])
        self.assertEqual(group['conditions', 'glare', 1]['tp'], group['conditions', 'occlusion', 1]['tp'])
        self.assertEqual(group['source_group', 'unknown', 0]['images'], 2)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            stratify_class_metrics([rows[0], rows[0]], [records[0], records[0]], [first, first], ['one', 'two', 'absent'])

    def test_class_match_partition_and_class_range_are_checked(self):
        truth = [{'class_id': 0, 'xyxy': [0, 0, 10, 10]}]
        with self.assertRaisesRegex(ValueError, 'partition'):
            class_detection_counts(truth, [], [], [], [], 1)
        with self.assertRaisesRegex(ValueError, 'outside'):
            class_detection_counts(truth, [{'class_id': 1}], [], [0], [0], 1)
        two = truth * 2
        with self.assertRaisesRegex(ValueError, 'partition'):
            class_detection_counts(two, two, [(0, 0, 1.0), (0, 0, 1.0)], [], [], 1)

    def test_joint_dark_low_saturation_by_domain_and_class(self):
        names = ['one', 'absent']
        domains = ['direct_real_native_clarity', 'direct_real_resampled', 'real_scene_composite',
                   'blender_render', 'other_synthetic', None]
        rows, recs, classes = [], [], []
        for i, domain in enumerate(domains):
            filename = f'{i}.png'
            rows.append(counts(filename, 1, 1 if i == 0 else 0, 0, 0 if i == 0 else 1))
            recs.append(item(filename, image_domain=domain, conditions=['dim', 'low_saturation', 'blur']))
            classes.append([{'ground_truth': 1, 'predictions': 1 if i == 0 else 0,
                             'tp': 1 if i == 0 else 0, 'fp': 0, 'fn': 0 if i == 0 else 1},
                            {'ground_truth': 0, 'predictions': 0, 'tp': 0, 'fp': 0, 'fn': 0}])
        rows.append(counts('unreviewed.png', 0, 0, 1, 0))
        recs.append(item('unreviewed.png', supervision='unreviewed', conditions=['dim', 'low_saturation']))
        classes.append([{'ground_truth': 0, 'predictions': 1, 'tp': 0, 'fp': 1, 'fn': 0}, classes[0][1]])
        output = joint_domain_condition_metrics(rows, recs, classes, names)
        lookup = {(g['image_domain'], g['conditions_all_of'], g['class_id']): g for g in output}
        key = '["dim", "low_saturation"]'
        for domain in domains:
            row = lookup[domain or 'unknown', key, 0]
            self.assertEqual((row['images'], row['ground_truth'], row['recall_denominator']), (1, 1, 1))
            self.assertEqual(row['image_domain_status'], 'confirmed' if domain else 'missing')
            self.assertIsNone(lookup[domain or 'unknown', key, 1]['recall'])
        self.assertEqual(lookup['direct_real_native_clarity', key, 0]['tp'], 1)
        self.assertEqual(lookup['direct_real_resampled', key, 0]['fn'], 1)
        self.assertEqual(lookup['direct_real_resampled', '["dim"]', 0]['images'], 1)
        self.assertEqual(lookup['direct_real_resampled', '["blur", "dim"]', 0]['images'], 1)

    def test_unrecorded_conditions_do_not_become_negative_condition(self):
        rows = [counts('a.png', 0, 0, 0, 0), counts('b.png', 0, 0, 0, 0)]
        recs = [item('a.png', state='complete_negative'), item('b.png', state='complete_negative', conditions=[])]
        zero = [{'ground_truth': 0, 'predictions': 0, 'tp': 0, 'fp': 0, 'fn': 0}]
        output = joint_domain_condition_metrics(rows, recs, [zero, zero], ['one'])
        self.assertEqual({row['condition_status'] for row in output}, {'missing', 'confirmed_empty'})
        self.assertTrue(all(row['conditions_all_of'] == '[]' for row in output))

    def test_invalid_empty_condition_types_are_not_confirmed_empty(self):
        rows = [counts('a.png', 0, 0, 0, 0)]
        zero = [{'ground_truth': 0, 'predictions': 0, 'tp': 0, 'fp': 0, 'fn': 0}]
        for value in ('', (), {}, False, 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                joint_domain_condition_metrics(rows, [item('a.png', state='complete_negative', conditions=value)],
                                               [zero], ['one'])
        with self.assertRaisesRegex(ValueError, 'at most 64'):
            joint_domain_condition_metrics(rows, [item('a.png', state='complete_negative',
                                                       conditions=[str(i) for i in range(65)])], [zero], ['one'])


class ImageDomain(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / 'image.png'
        Image.new('RGB', (640, 360)).save(self.image)

    def test_direct_video_requires_explicit_review_and_actual_dimensions(self):
        source = {'source_kind': 'video_frame', 'generation_method': 'provided_video_frame'}
        self.assertEqual(reviewed_image_domain(source, self.image), (None, None))
        p = {'resolution_review': 'confirmed', 'source_size': [3840, 2160], 'output_size': [640, 360],
             'resampling': 'resampled', 'clarity': 'modified'}
        self.assertEqual(reviewed_image_domain({**source, 'image_processing': p}, self.image)[0], 'direct_real_resampled')
        self.assertEqual(reviewed_image_domain({**source, 'image_processing': {**p, 'resampling': 'none', 'clarity': 'original'}}, self.image), (None, None))
        native = {**p, 'source_size': [640, 360], 'resampling': 'none', 'clarity': 'original'}
        self.assertEqual(reviewed_image_domain({**source, 'image_processing': native}, self.image)[0], 'direct_real_native_clarity')
        self.assertEqual(reviewed_image_domain({**source, 'image_processing': {**native, 'output_size': [640, 480]}}, self.image), (None, None))
        self.assertEqual(reviewed_image_domain({**source, 'image_processing': {**native, 'resolution_review': 'pending'}}, self.image), (None, None))

    def test_synthetic_composite_and_augmentation_require_reviewed_origins(self):
        self.assertEqual(reviewed_image_domain({'source_kind': 'blender_scene', 'generation_method': 'blender_render'}, self.image)[0], 'blender_render')
        self.assertEqual(reviewed_image_domain({'source_kind': 'synthetic_scene', 'generation_method': 'synthetic_render'}, self.image)[0], 'other_synthetic')
        composite = {'source_kind': 'derived_image', 'generation_method': 'composite',
                     'image_processing': {'background_origin': 'direct_camera'}}
        self.assertEqual(reviewed_image_domain(composite, self.image), (None, None))
        composite['image_processing']['origin_review'] = 'confirmed'
        self.assertEqual(reviewed_image_domain(composite, self.image)[0], 'real_scene_composite')
        augmentation = {'source_kind': 'derived_image', 'generation_method': 'augmentation',
                        'image_processing': {'origin_review': 'confirmed', 'parent_origin': 'direct_camera'}}
        self.assertEqual(reviewed_image_domain(augmentation, self.image)[0], 'real_augmentation')
        augmentation['image_processing']['parent_origin'] = 'unknown'
        self.assertEqual(reviewed_image_domain(augmentation, self.image), (None, None))


class EmptyBoxes:
    def __len__(self): return 0


class EmptyResult:
    boxes = EmptyBoxes()
    keypoints = None
    orig_shape = (64, 64)

    def __init__(self, path): self.path = str(path)
    def save_txt(self, *args, **kwargs): pass


class EmptyModel:
    def predict(self, **options):
        return (EmptyResult(path) for path in sorted(Path(options['source']).glob('*.png')))


class BrokenModel:
    def predict(self, **options):
        raise RuntimeError('fixture prediction failure')


class StagedProvenance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {'model_key': 'person', 'task': 'detect', 'version': 'v4',
                       'data': 'dataset/data.yaml', 'seed': 1, 'imgsz': 64,
                       'device': 'cpu', 'preview_limit': 0}
        (self.root / 'review').mkdir()
        (self.root / 'review/task.json').write_text('{}', encoding='utf8')
        (self.root / 'review/source.json').write_text('{}', encoding='utf8')
        rows = []
        for split in ('train', 'val', 'test'):
            for folder in ('images', 'labels'):
                (self.root / 'dataset' / folder / split).mkdir(parents=True)
            for index in range(3):
                image = f'dataset/images/{split}/{index}.png'
                label = f'dataset/labels/{split}/{index}.txt'
                Image.new('RGB', (64, 64)).save(self.root / image)
                (self.root / label).write_text('0 0.5 0.5 0.2 0.2\n' if index == 0 else '', encoding='utf8')
                task = {'state': 'complete_positive' if index == 0 else 'complete_negative',
                        'review_evidence': 'review/task.json', 'strata_review': 'confirmed',
                        'strata': {'difficulty': 'hard', 'conditions': ['glare', 'motion']}}
                rows.append({'image': image, 'label': label, 'split': split, 'selected_tasks': ['person'],
                             'source_group': 'camera_1', 'source_kind': 'video_frame',
                             'generation_method': 'provided_video_frame',
                             'lineage_review': 'confirmed',
                             'near_duplicate_review': 'confirmed', 'review_evidence': 'review/source.json',
                             'tasks': {'person': task}})
        (self.root / 'dataset/data.yaml').write_text(yaml.safe_dump(
            {'names': ['person'], **{s: 'images/' + s for s in ('train', 'val', 'test')}}), encoding='utf8')
        (self.root / 'dataset/workspace_sources.json').write_text(json.dumps({'records': rows}), encoding='utf8')

    def test_snapshot_and_diagnostic_use_only_selected_split(self):
        source_path = self.root / 'dataset/workspace_sources.json'
        data = json.loads(source_path.read_text(encoding='utf8'))
        for row in data['records']:
            if row['split'] != 'val':
                continue
            index = int(Path(row['image']).stem)
            row['tasks']['person']['strata']['conditions'] = ['dim', 'low_saturation']
            if index in (0, 1):
                row['image_processing'] = {'resolution_review': 'confirmed',
                    'source_size': [64, 64] if index == 0 else [3840, 2160], 'output_size': [64, 64],
                    'resampling': 'none' if index == 0 else 'resampled',
                    'clarity': 'original' if index == 0 else 'modified'}
        source_path.write_text(json.dumps(data), encoding='utf8')
        run = self.root / 'runs/person/fixture'; run.mkdir(parents=True)
        data_path, _ = stage_dataset(self.root, self.config, run)
        samples = json.loads((run / 'data/samples.json').read_text(encoding='utf8'))
        val = [row for row in samples if row['split'] == 'val']
        self.assertEqual(val[0]['reviewed_strata']['conditions'], ['dim', 'low_saturation'])
        self.assertEqual(val[0]['reviewed_strata']['image_domain'], 'direct_real_native_clarity')
        self.assertEqual(val[1]['reviewed_strata']['image_domain'], 'direct_real_resampled')
        self.assertEqual(val[1]['reviewed_strata']['image_domain_basis']['source_size'], [3840, 2160])
        self.assertNotIn('image_domain', val[2]['reviewed_strata'])
        condition_evidence = val[0]['reviewed_strata']['conditions_review_evidence_snapshot']
        self.assertEqual((run / condition_evidence).read_text(encoding='utf8'), '{}')
        self.assertEqual(val[2]['reviewed_strata']['label_state'], 'complete_negative')
        # Changing the source manifest after staging cannot change the run snapshot.
        (self.root / 'dataset/workspace_sources.json').write_text('{"records": []}', encoding='utf8')
        destination = run / 'evaluation/val'; destination.mkdir(parents=True)
        result = diagnostics(EmptyModel(), data_path, 'val', self.config, destination)
        self.assertEqual(result['images'], 3)
        self.assertEqual(result['reviewed_strata']['included_images'], 3)
        self.assertEqual(len(result['reviewed_strata']['excluded_images']), 0)
        with (destination / 'by_reviewed_stratum.csv').open(encoding='utf-8-sig', newline='') as stream:
            grouped = list(csv.DictReader(stream))
        group = next(g for g in grouped if g['dimension'] == 'generation_method' and g['value'] == 'provided_video_frame')
        self.assertEqual((group['images'], group['ground_truth'], group['fn'], group['negative_images']), ('3', '1', '1', '2'))
        with (destination / 'by_reviewed_stratum_class.csv').open(encoding='utf-8-sig', newline='') as stream:
            grouped_class = list(csv.DictReader(stream))
        group = next(g for g in grouped_class if g['dimension'] == 'difficulty' and g['value'] == 'hard')
        self.assertEqual((group['class_id'], group['images'], group['ground_truth'], group['fn'], group['images_without_class']),
                         ('0', '3', '1', '1', '2'))
        with (destination / 'by_domain_conditions_class.csv').open(encoding='utf-8-sig', newline='') as stream:
            joint = list(csv.DictReader(stream))
        group = next(g for g in joint if g['image_domain'] == 'direct_real_resampled' and
                     g['conditions_all_of'] == '["dim", "low_saturation"]')
        self.assertEqual((group['images'], group['ground_truth'], group['recall_denominator']), ('1', '0', '0'))
        self.assertEqual(group['recall'], '')

    def test_v3_diagnostics_do_not_create_v4_stratum_files(self):
        run = self.root / 'runs/person/v3'; run.mkdir(parents=True)
        config = {**self.config, 'version': 'v3'}
        data_path, _ = stage_dataset(self.root, config, run)
        destination = run / 'evaluation/val'; destination.mkdir(parents=True)
        diagnostics(EmptyModel(), data_path, 'val', config, destination)
        self.assertFalse((destination / 'by_reviewed_stratum_class.csv').exists())
        self.assertFalse((destination / 'by_reviewed_stratum.csv').exists())

    def test_v3_unchanged_and_wrong_label_not_audited(self):
        source = self.root / 'dataset/workspace_sources.json'
        data = json.loads(source.read_text(encoding='utf8'))
        data['records'][0]['label'] = 'dataset/labels/train/other.txt'
        source.write_text(json.dumps(data), encoding='utf8')
        self.assertEqual(records(self.root, self.config, 'train')[0]['reviewed_strata'], {'supervision': 'unreviewed'})
        self.assertNotIn('reviewed_strata', records(self.root, {**self.config, 'version': 'v3'}, 'train')[0])

    def test_unreviewed_conditions_remain_unknown_and_not_snapshotted(self):
        path = self.root / 'dataset/workspace_sources.json'
        data = json.loads(path.read_text(encoding='utf8'))
        task = data['records'][0]['tasks']['person']
        task['strata_review'] = 'pending'
        task['strata']['conditions'] = ['dim', 'low_saturation']
        path.write_text(json.dumps(data), encoding='utf8')
        run = self.root / 'runs/person/unreviewed_conditions'; run.mkdir(parents=True)
        stage_dataset(self.root, self.config, run)
        sample = json.loads((run / 'data/samples.json').read_text(encoding='utf8'))[0]['reviewed_strata']
        self.assertNotIn('conditions', sample)
        self.assertNotIn('conditions_review_evidence_snapshot', sample)

    def test_prediction_failure_is_recorded_without_claiming_stratum_results(self):
        run = self.root / 'runs/person/failed'; run.mkdir(parents=True)
        data_path, _ = stage_dataset(self.root, self.config, run)
        destination = run / 'evaluation/val'; destination.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, 'fixture prediction failure'):
            with run_context(self.root, self.config, run, 'evaluate_val'):
                diagnostics(BrokenModel(), data_path, 'val', self.config, destination)
        status = json.loads((run / 'metadata/evaluate_val_status.json').read_text(encoding='utf8'))
        self.assertEqual(status['status'], 'failed')
        self.assertIn('fixture prediction failure', status['error'])
        self.assertFalse((destination / 'by_reviewed_stratum.csv').exists())

    def test_frozen_dataset_comparison_includes_strata_snapshot(self):
        source_path = self.root / 'dataset/workspace_sources.json'
        source = json.loads(source_path.read_text(encoding='utf8'))
        source['records'][0]['image_processing'] = {'resolution_review': 'confirmed', 'source_size': [3840, 2160],
            'output_size': [64, 64], 'resampling': 'resampled', 'clarity': 'modified'}
        source_path.write_text(json.dumps(source), encoding='utf8')
        run = self.root / 'runs/person/input'; run.mkdir(parents=True)
        stage_dataset(self.root, self.config, run)
        frozen = self.root / 'runs/person/frozen'; frozen.mkdir(parents=True)
        shutil.copytree(run / 'data', frozen / 'data')
        self.assertTrue(_same_dataset(run / 'data', frozen / 'data'))
        path = frozen / 'data/samples.json'
        samples = json.loads(path.read_text(encoding='utf8'))
        self.assertEqual(samples[0]['reviewed_strata']['image_domain_basis']['source_size'], [3840, 2160])
        evidence_path = samples[0]['reviewed_strata']['image_domain_basis']['source_review_evidence_snapshot']
        self.assertEqual((frozen / evidence_path).read_text(encoding='utf8'), '{}')
        conditions_path = samples[0]['reviewed_strata']['conditions_review_evidence_snapshot']
        self.assertEqual((frozen / conditions_path).read_text(encoding='utf8'), '{}')
        (frozen / evidence_path).write_text('changed evidence', encoding='utf8')
        self.assertFalse(_same_dataset(run / 'data', frozen / 'data'))
        (frozen / evidence_path).write_text('{}', encoding='utf8')
        self.assertTrue(_same_dataset(run / 'data', frozen / 'data'))
        samples[0]['reviewed_strata']['image_domain_basis']['source_size'] = [64, 64]
        path.write_text(json.dumps(samples), encoding='utf8')
        self.assertFalse(_same_dataset(run / 'data', frozen / 'data'))

    def test_v4_staging_rejects_missing_or_inconsistent_supervision(self):
        source = self.root / 'dataset/workspace_sources.json'
        original = source.read_text(encoding='utf8')
        source.write_text('{"records": []}', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'complete reviewed supervision'):
            stage_dataset(self.root, self.config, self.root / 'runs/without_source')
        data = json.loads(original)
        data['records'][0]['tasks']['person']['state'] = 'complete_negative'
        source.write_text(json.dumps(data), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'matching provenance'):
            stage_dataset(self.root, self.config, self.root / 'runs/inconsistent')

    def test_flat_gate_export_can_stage_with_reviewed_strata(self):
        path = self.root / 'dataset/workspace_sources.json'
        data = json.loads(path.read_text(encoding='utf8'))
        output = []
        for row in data['records']:
            task = row['tasks']['person']
            output.append({'image': row['image'], 'label': row['label'], 'split': row['split'],
                           'source_id': row['source_group'], 'source_kind': row['source_kind'],
                           'generation_method': row['generation_method'], 'label_status': 'reviewed',
                           'review_state': task['state'], 'review_evidence': task['review_evidence'],
                           'source_review_evidence': row['review_evidence'],
                           'strata': task['strata'], 'strata_review': task['strata_review'],
                           'image_processing': {'resolution_review': 'confirmed', 'source_size': [64, 64],
                                                'output_size': [64, 64], 'resampling': 'none', 'clarity': 'original'}})
        path.write_text(json.dumps({'schema_version': 1, 'model_key': 'person', 'records': output}), encoding='utf8')
        run = self.root / 'runs/person/flat'; run.mkdir(parents=True)
        stage_dataset(self.root, self.config, run)
        sample = json.loads((run / 'data/samples.json').read_text(encoding='utf8'))[0]
        self.assertEqual(sample['source_group'], 'camera_1')
        self.assertEqual(sample['reviewed_strata']['difficulty'], 'hard')
        self.assertEqual(sample['reviewed_strata']['image_domain'], 'direct_real_native_clarity')

    def test_actual_built_package_uses_its_evidence_map(self):
        spec = importlib.util.spec_from_file_location('v4_package_fixture', Path(__file__).with_name('test_package_builder.py'))
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        root = self.root / 'workspace'; root.mkdir()
        builder_fixture = fixture.PackageBuilderTest('runTest')
        builder_fixture.setUp()
        self.addCleanup(builder_fixture.doCleanups)
        builder_fixture.v4_fixture(root)
        queue = {'samples': [], 'near_duplicate_pairs': [], 'exposed_images': [], 'inputs': [],
                 'canonical_models': {'person': {'names': ['community', 'noncommunity']}}}
        with mock.patch.object(fixture.v4_source_gate, 'build_queue', return_value=queue), \
                mock.patch.object(fixture.builder, '_verify_v4_staged', return_value={'status': 'fixture_mock'}), \
                mock.patch.object(fixture.builder, '_verify_v4_launcher', return_value={'status': 'fixture_mock'}), \
                mock.patch.object(fixture.builder, 'ROOT', root), \
                mock.patch.object(fixture.builder, 'DEST', root / 'packages'):
            fixture.builder.build_v4('person', 'input/manifest.json')
        package = root / 'packages/person'
        self.assertFalse((package / 'input/review.txt').exists())
        config = yaml.safe_load((package / 'config/train.yaml').read_text(encoding='utf8'))
        run = package / 'runs/person/evidence'; run.mkdir(parents=True)
        stage_dataset(package, config, run)
        samples = json.loads((run / 'data/samples.json').read_text(encoding='utf8'))
        self.assertTrue(all(row['reviewed_strata']['supervision'] == 'complete' for row in samples))
        evidence = package / 'dataset/source_records/evidence_map.json'
        mapping = json.loads(evidence.read_text(encoding='utf8'))
        mapping['input/review.txt'] = '../outside.txt'
        evidence.write_text(json.dumps(mapping), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'complete reviewed supervision'):
            stage_dataset(package, config, package / 'runs/person/bad_evidence')


if __name__ == '__main__':
    unittest.main()
