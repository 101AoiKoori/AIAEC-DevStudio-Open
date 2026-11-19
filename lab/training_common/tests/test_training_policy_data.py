"""Exercise training split policies with temporary datasets and no models."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.data import preflight_dataset, stage_dataset


class TrainingPolicyData(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / 'dataset'
        self.spec = {'names': ['community', 'noncommunity'],
                     'train': 'images/train', 'val': 'images/val', 'test': 'images/test'}
        self.evidence = self.root / 'review.json'
        self.evidence.write_text('{}', encoding='utf-8')
        self.rows = [self.sample('train', 0, 'red'), self.sample('val', 0, 'green'), self.sample('train', 1, 'yellow'),
                     self.sample('val', 1, 'blue')]
        self.source = self.data / 'workspace_sources.json'
        self.put_manifest()
        self.write_spec()
        self.config = {'task': 'detect', 'version': 'v4', 'model_key': 'person',
                       'data': 'dataset/data.yaml', 'training_policy': {'mode': 'fixed'}}
        self.run_dir = self.root / 'runs'
        self.run_dir.mkdir()

    def sample(self, split, class_id, color):
        stem = 'negative' if class_id is None else str(class_id)
        image = self.data / f'images/{split}/{stem}.png'
        label = self.data / f'labels/{split}/{stem}.txt'
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (20, 16), color=color).save(image)
        label.write_text('' if class_id is None else f'{class_id} 0.5 0.5 0.5 0.5\n', encoding='utf-8')
        state = 'complete_negative' if class_id is None else 'complete_positive'
        return {'image': image.relative_to(self.root).as_posix(), 'label': label.relative_to(self.root).as_posix(),
                'split': split, 'source_group': 'session_' + split, 'selected_tasks': ['person'],
                'lineage_review': 'confirmed', 'near_duplicate_review': 'confirmed', 'review_evidence': 'review.json',
                'tasks': {'person': {'state': state, 'review_evidence': 'review.json'}}}

    def put_manifest(self, **fields):
        self.source.write_text(json.dumps({'records': self.rows, **fields}), encoding='utf-8')

    def write_spec(self):
        (self.data / 'data.yaml').write_text(yaml.safe_dump(self.spec), encoding='utf-8')

    def early(self):
        return {**self.config, 'training_policy': {'mode': 'early_stopping'}}

    def test_fixed_validation_stages_reviewed_train_and_val_without_test(self):
        config = {**self.config, 'training_policy': {'mode': 'fixed_validation'}}
        path, report = stage_dataset(self.root, config, self.run_dir)
        spec = yaml.safe_load(path.read_text(encoding='utf-8'))
        self.assertEqual(report['selected_images'], {'train': 2, 'val': 2})
        self.assertEqual(report['training_mode'], 'fixed_validation')
        self.assertEqual(spec['val'], 'images/val')
        self.assertNotIn('test', spec)

    def test_fixed_skips_validation_files_and_states(self):
        (self.root / self.rows[1]['label']).unlink()
        self.rows.append(self.rows[1])
        self.put_manifest()
        _, splits, report = preflight_dataset(self.root, self.config)
        self.assertEqual(list(splits), ['train'])
        self.assertEqual(report['test_status'], 'not_used_in_training')
        path, staged = stage_dataset(self.root, self.config, self.run_dir)
        self.assertEqual(staged['selected_images'], {'train': 2})
        spec = yaml.safe_load(path.read_text(encoding='utf-8'))
        self.assertIsNone(spec['val'])
        self.assertNotIn('test', spec)
        self.assertEqual([child.name for child in (path.parent / 'images').iterdir()], ['train'])

    def test_fixed_missing_review_manifest_fails_in_read_only_preflight(self):
        self.source.unlink()
        with self.assertRaisesRegex(ValueError, r'missing source evidence manifest: .*workspace_sources.json$'):
            preflight_dataset(self.root, self.config)

    def test_missing_configured_data_is_reported_before_missing_source_index(self):
        config = {**self.config, 'data': 'old_export/data.yaml'}
        with self.assertRaisesRegex(ValueError, 'Training data YAML missing.*old_export/data.yaml'):
            preflight_dataset(self.root, config)

    def test_model_selection_explains_all_negative_validation_before_training(self):
        for row in self.rows:
            if row['split'] == 'val':
                (self.root / row['label']).write_text('', encoding='utf-8')
                row['tasks']['person']['state'] = 'complete_negative'
        self.put_manifest()
        config = {**self.config, 'training_policy': {'mode': 'fixed_validation'}}
        with self.assertRaisesRegex(ValueError, 'requires validation ground-truth.*mode=fixed'):
            preflight_dataset(self.root, config)
        preflight_dataset(self.root, self.config)

    def test_lineage_and_near_duplicate_review_status_do_not_gate_supervision(self):
        self.rows[0]['lineage_review'] = 'unknown'
        self.rows[0].pop('near_duplicate_review')
        self.put_manifest()
        _, splits, _ = preflight_dataset(self.root, self.config)
        self.assertTrue(all(row['reviewed_strata']['supervision'] == 'complete' for row in splits['train']))

    def test_policy_rejects_swapped_dataset_class_names(self):
        config = {**self.config, 'class_names': list(reversed(self.spec['names']))}
        with self.assertRaisesRegex(ValueError, 'must match the dataset class names and order'):
            preflight_dataset(self.root, config)

    def test_fixed_warns_of_class_coverage_after_train_limit(self):
        config = {**self.config, 'train_limit': 1}
        _, _, report = preflight_dataset(self.root, config)
        self.assertEqual(report['class_coverage_warnings'][0]['split'], 'train')
        self.assertEqual(len(report['class_coverage_warnings'][0]['missing_classes']), 1)
        path, staged = stage_dataset(self.root, config, self.run_dir)
        stored = json.loads((path.parent / 'selection.json').read_text(encoding='utf-8'))
        self.assertEqual(stored['class_coverage_warnings'], report['class_coverage_warnings'])
        self.assertEqual(staged['class_coverage_warnings'], report['class_coverage_warnings'])

    def test_missing_training_class_in_source_warns_for_all_policies(self):
        row = self.rows[2]
        row['tasks']['person']['state'] = 'complete_negative'
        (self.root / row['label']).write_text('', encoding='utf-8')
        self.put_manifest()
        for mode in ('fixed', 'fixed_validation', 'early_stopping'):
            with self.subTest(mode=mode):
                config = {**self.config, 'training_policy': {'mode': mode}}
                _, _, report = preflight_dataset(self.root, config)
                self.assertEqual(report['class_coverage_warnings'][0]['split'], 'train')
                self.assertEqual(report['class_coverage_warnings'][0]['missing_classes'], ['noncommunity'])

    def test_fixed_reports_available_and_selected_counts_and_staged_label_hash(self):
        self.rows.append(self.sample('train', None, 'gray'))
        self.put_manifest()
        label = self.root / self.rows[0]['label']
        with label.open('a', encoding='utf-8') as target:
            target.write('1 0.3 0.3 0.2 0.2\n1 0.7 0.7 0.2 0.2\n')
        config = {**self.config, 'train_limit': 1}
        _, _, report = preflight_dataset(self.root, config)
        available = report['available_label_counts']['train']
        selected = report['selected_label_counts']['train']
        self.assertEqual(available['negative_images'], 1)
        self.assertEqual(selected['negative_images'], 0)
        self.assertEqual(available['classes'][1], {'class_id': 1, 'class_name': 'noncommunity', 'images': 2, 'boxes': 3})
        self.assertEqual(selected['classes'][1], {'class_id': 1, 'class_name': 'noncommunity', 'images': 1, 'boxes': 2})
        path, staged = stage_dataset(self.root, config, self.run_dir)
        self.assertEqual(staged['selected_label_counts'], report['selected_label_counts'])
        sample, = json.loads((path.parent / 'samples.json').read_text(encoding='utf-8'))
        expected = hashlib.sha256((self.run_dir / sample['run_label']).read_bytes()).hexdigest()
        self.assertEqual(sample['label_sha256'], expected)

    def test_fixed_rejects_label_replaced_after_preflight(self):
        original = self.root / self.rows[0]['label']
        copy_file = shutil.copy2

        def copy_with_update(source, destination):
            if source == original:
                source.write_text('0 0.5 0.5 0.3 0.3\n', encoding='utf-8')
            return copy_file(source, destination)

        with patch('engine.data.shutil.copy2', side_effect=copy_with_update):
            with self.assertRaisesRegex(ValueError, 'Training label changed between preflight and staging'):
                stage_dataset(self.root, self.config, self.run_dir)
        self.assertFalse((self.run_dir / 'data/data.yaml').exists())

    def test_early_stages_train_and_full_validation_without_test(self):
        path, report = stage_dataset(self.root, self.early(), self.run_dir)
        self.assertEqual(report['selected_images'], {'train': 2, 'val': 2})
        self.assertEqual(report['source_reuse'], [])
        spec = yaml.safe_load(path.read_text(encoding='utf-8'))
        self.assertEqual(spec['val'], 'images/val')
        self.assertNotIn('test', spec)

    def test_early_rejects_byte_copy_or_identical_decoded_pixels(self):
        first = self.root / self.rows[0]['image']
        second = self.root / self.rows[1]['image']
        second.write_bytes(first.read_bytes())
        with self.assertRaisesRegex(ValueError, r'repeats an image \(file\)'):
            preflight_dataset(self.root, self.early())
        Image.new('RGB', (20, 16), color='red').save(second, compress_level=0)
        self.assertNotEqual(first.read_bytes(), second.read_bytes())
        with self.assertRaisesRegex(ValueError, r'repeats an image \(pixels\)'):
            preflight_dataset(self.root, self.early())

    def test_early_reports_observed_and_declared_reuse_as_diagnostic(self):
        for row in self.rows:
            row['parent_video'] = 'camera.mp4'
        self.put_manifest()
        _, _, report = preflight_dataset(self.root, self.early())
        self.assertEqual(report['evaluation_scope'], 'same_source_diagnostic')
        self.assertFalse(report['source_reuse'][0]['declared_in_manifest'])
        reuse = {'relation': 'parent_video', 'source': 'camera.mp4', 'splits': ['train', 'val']}
        self.put_manifest(source_reuse=[reuse])
        _, _, report = preflight_dataset(self.root, self.early())
        self.assertEqual(report['source_reuse'][0]['counts'], {'train': 2, 'val': 2})
        self.assertTrue(report['source_reuse'][0]['declared_in_manifest'])

    def test_unknown_source_group_is_reported_without_losing_known_reuse(self):
        self.evidence.write_text('{"source":"machine"}', encoding='utf-8')
        for row in self.rows:
            row['source_group'] = None
            row['parent_video'] = 'camera.mp4'
            row['episode_group'] = 'local_id'
        self.put_manifest()
        path, report = stage_dataset(self.root, self.early(), self.run_dir)
        self.assertEqual(report['source_group_unknown_counts'], {'train': 2, 'val': 2})
        self.assertEqual(report['evaluation_scope'], 'same_source_diagnostic')
        self.assertEqual([row['relation'] for row in report['source_reuse']], ['parent_video'])
        self.assertEqual(json.loads((path.parent / 'selection.json').read_text(encoding='utf-8'))
                         ['source_group_unknown_counts'], {'train': 2, 'val': 2})

    def test_unknown_source_group_alone_never_claims_independence(self):
        self.rows[0]['source_group'] = None
        self.rows[1]['source_group'] = None
        self.put_manifest()
        _, _, report = preflight_dataset(self.root, self.early())
        self.assertEqual(report['source_group_unknown_counts'], {'train': 1, 'val': 1})
        self.assertEqual(report['evaluation_scope'], 'independence_not_established')
        self.assertEqual(report['source_reuse'], [])
        (self.root / self.rows[1]['image']).write_bytes((self.root / self.rows[0]['image']).read_bytes())
        with self.assertRaisesRegex(ValueError, r'repeats an image \(file\)'):
            preflight_dataset(self.root, self.early())

    def test_staging_preserves_prior_diagnostic_contact(self):
        self.rows[1]['development_contact'] = ['near_diagnostic_frame']
        self.put_manifest()
        path, report = stage_dataset(self.root, self.early(), self.run_dir)
        self.assertEqual(report['evaluation_scope'], 'same_source_diagnostic')
        self.assertEqual(report['development_contact_counts'], {'train': 0, 'val': 1})
        samples = json.loads((path.parent / 'samples.json').read_text(encoding='utf-8'))
        self.assertEqual([row['development_contact'] for row in samples if row.get('development_contact')],
                         [['near_diagnostic_frame']])

    def test_early_rejects_unreviewed_and_warns_missing_validation_classes(self):
        truth = self.rows[-1]['tasks']['person']
        truth['review_evidence'] = 'missing.json'
        self.put_manifest()
        with self.assertRaisesRegex(ValueError, 'complete reviewed supervision'):
            preflight_dataset(self.root, self.early())
        truth.update(review_evidence='review.json', state='complete_negative')
        (self.root / self.rows[-1]['label']).write_text('', encoding='utf-8')
        self.put_manifest()
        _, _, report = preflight_dataset(self.root, self.early())
        self.assertEqual(report['class_coverage_warnings'][0]['split'], 'val')
        self.assertEqual(report['class_coverage_warnings'][0]['missing_classes'], ['noncommunity'])

    def test_missing_label_and_out_of_range_class_still_fail(self):
        label = self.root / self.rows[0]['label']
        label.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing label'):
            preflight_dataset(self.root, self.config)
        label.write_text('2 0.5 0.5 0.5 0.5\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            preflight_dataset(self.root, self.config)

    def test_empty_split_still_fails(self):
        for row in self.rows:
            if row['split'] == 'train':
                (self.root / row['image']).unlink()
                (self.root / row['label']).unlink()
        with self.assertRaisesRegex(ValueError, 'Empty dataset split: train'):
            preflight_dataset(self.root, self.config)

    def test_early_pose_requires_labeled_keypoints_in_validation(self):
        self.spec['kpt_shape'] = [1, 3]
        self.write_spec()
        for row in self.rows:
            label = self.root / row['label']
            label.write_text(label.read_text(encoding='utf-8').strip() + ' 0 0 0\n', encoding='utf-8')
        config = {**self.early(), 'task': 'pose', 'keypoint_names': ['pivot']}
        with self.assertRaisesRegex(ValueError, 'needs labeled visible keypoints'):
            preflight_dataset(self.root, config)
        (self.root / self.rows[-1]['label']).write_text('1 0.5 0.5 0.5 0.5 0.5 0.5 2\n', encoding='utf-8')
        _, splits, _ = preflight_dataset(self.root, config)
        self.assertEqual(sum(row['visible_keypoints'] for row in splits['val']), 1)
        with self.assertRaisesRegex(ValueError, 'keypoint_names requires kpt_shape'):
            preflight_dataset(self.root, {**config, 'keypoint_names': ['pivot', 'tip']})

    def test_staging_outside_training_uses_legacy_split_scope(self):
        self.rows.append(self.sample('test', 1, 'purple'))
        self.put_manifest()
        _, report = stage_dataset(self.root, self.config, self.run_dir, training=False)
        self.assertEqual(report['selected_images'], {'train': 2, 'val': 2, 'test': 1})
        self.assertNotIn('selected_label_counts', report)


if __name__ == '__main__':
    unittest.main()
