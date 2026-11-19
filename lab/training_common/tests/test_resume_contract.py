"""Verify the relocated training-data contract without loading a checkpoint."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.data import relocate_staged_data, stage_dataset, staged_dataset_signature, validate_staged_resume_data


class StagedResumeContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.root = self.temp / 'first workspace'
        self.run = self.root / 'runs' / 'started'
        self.run.mkdir(parents=True)

    def stage(self, mode=None, *, test_unavailable=False, pose=False):
        staged_splits = ('train', 'val') if test_unavailable else ('train', 'val', 'test')
        spec = {'names': ['target'], **{split: 'images/' + split for split in staged_splits}}
        config = {'version': 'v4' if mode or test_unavailable else 'v3', 'model_key': 'person',
                  'task': 'pose' if pose else 'detect', 'data': 'dataset/data.yaml',
                  'class_names': ['target'], 'model': 'weights/original.pt'}
        if mode:
            config['training_policy'] = {'mode': mode}
        if test_unavailable:
            config['test_unavailable'] = True
        if pose:
            spec['kpt_shape'] = [1, 3]
            config['keypoint_names'] = ['point']
        evidence = self.root / 'dataset/review.json'
        evidence.parent.mkdir()
        evidence.write_text('{}', encoding='utf-8')
        rows = []
        for split, color in zip(staged_splits, ('red', 'green', 'blue')):
            image = self.root / 'dataset/images' / split / 'item.png'
            label = self.root / 'dataset/labels' / split / 'item.txt'
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            Image.new('RGB', (20, 16), color=color).save(image)
            label.write_text('0 0.5 0.5 0.3 0.3' + (' 0.5 0.5 2' if pose else '') + '\n', encoding='utf-8')
            rows.append({'image': image.relative_to(self.root).as_posix(),
                         'label': label.relative_to(self.root).as_posix(), 'split': split,
                         'source_group': 'session_' + split, 'selected_tasks': ['person'],
                         'lineage_review': 'confirmed', 'near_duplicate_review': 'confirmed',
                         'review_evidence': 'dataset/review.json',
                         'tasks': {'person': {'state': 'complete_positive', 'review_evidence': 'dataset/review.json'}}})
        (self.root / 'dataset/data.yaml').write_text(yaml.safe_dump(spec), encoding='utf-8')
        (self.root / 'dataset/workspace_sources.json').write_text(json.dumps({'records': rows}), encoding='utf-8')
        stage_dataset(self.root, config, self.run)
        return config

    def samples(self):
        return json.loads((self.run / 'data/samples.json').read_text(encoding='utf-8'))

    def write_samples(self, samples):
        (self.run / 'data/samples.json').write_text(json.dumps(samples), encoding='utf-8')

    def test_copied_fixed_run_needs_no_original_sources_and_keeps_signature(self):
        config = self.stage('fixed')
        self.assertFalse((self.root / config['model']).exists())
        original_signature = staged_dataset_signature(self.run)
        target = self.temp / 'second workspace' / 'runs' / 'copied'
        target.parent.mkdir(parents=True)
        shutil.copytree(self.run, target)
        for record in self.samples():
            for key in ('image', 'label'):
                (self.root / record[key]).unlink()
        (self.root / config['data']).unlink()
        self.assertEqual(validate_staged_resume_data(target, config), original_signature)
        path = relocate_staged_data(target)
        self.assertEqual(yaml.safe_load(path.read_text(encoding='utf-8'))['path'], str(target / 'data'))
        self.assertEqual(staged_dataset_signature(target), original_signature)

    def test_early_pose_checks_train_val_and_keypoints(self):
        config = self.stage('early_stopping', pose=True)
        self.assertEqual({row['split'] for row in self.samples()}, {'train', 'val'})
        self.assertEqual(validate_staged_resume_data(self.run, config), staged_dataset_signature(self.run))
        with self.assertRaisesRegex(ValueError, 'keypoints differ'):
            validate_staged_resume_data(self.run, {**config, 'keypoint_names': ['first', 'second']})

    def test_legacy_missing_hashes_still_uses_structural_checks(self):
        config = self.stage()
        samples = self.samples()
        self.assertEqual({row['split'] for row in samples}, {'train', 'val', 'test'})
        for row in samples:
            row.pop('image_sha256')
            row.pop('label_sha256')
        self.write_samples(samples)
        self.assertEqual(validate_staged_resume_data(self.run, config), staged_dataset_signature(self.run))
        with self.assertRaisesRegex(ValueError, 'class order differs'):
            validate_staged_resume_data(self.run, {**config, 'class_names': ['different']})

    def test_legacy_explicit_test_unavailable_accepts_only_train_val(self):
        config = self.stage(test_unavailable=True)
        self.assertEqual({row['split'] for row in self.samples()}, {'train', 'val'})
        self.assertEqual(validate_staged_resume_data(self.run, config), staged_dataset_signature(self.run))

    def test_tampered_image_or_label_is_rejected(self):
        config = self.stage()
        sample = self.samples()[0]
        for kind in ('image', 'label'):
            with self.subTest(kind=kind):
                path = self.run / sample['run_' + kind]
                original = path.read_bytes()
                self.assertEqual(sample[kind + '_sha256'], hashlib.sha256(original).hexdigest())
                path.write_bytes(original + b'tampered')
                with self.assertRaisesRegex(ValueError, kind + '_sha256 mismatch'):
                    validate_staged_resume_data(self.run, config)
                path.write_bytes(original)

    def test_files_missing_or_unlisted_are_rejected(self):
        config = self.stage()
        image = self.run / self.samples()[0]['run_image']
        extra = image.with_name('unlisted.png')
        shutil.copy2(image, extra)
        with self.assertRaisesRegex(ValueError, 'disagree with the sample manifest'):
            validate_staged_resume_data(self.run, config)
        extra.unlink()
        image.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing, duplicated, misplaced or unpaired'):
            validate_staged_resume_data(self.run, config)

    def test_external_or_cross_split_sample_paths_are_rejected(self):
        config = self.stage()
        samples = self.samples()
        manifest = samples[0]
        label = manifest['run_label']
        manifest['run_label'] = '../' + label
        self.write_samples(samples)
        with self.assertRaisesRegex(ValueError, 'Path escapes'):
            validate_staged_resume_data(self.run, config)
        manifest['run_label'] = samples[1]['run_label']
        self.write_samples(samples)
        with self.assertRaisesRegex(ValueError, 'misplaced or unpaired'):
            validate_staged_resume_data(self.run, config)


if __name__ == '__main__':
    unittest.main()
