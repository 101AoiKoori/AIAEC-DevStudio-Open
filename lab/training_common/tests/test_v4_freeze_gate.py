"""Test selection, fixed test input, evidence and relocation with synthetic files."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'lab/training_common'))
from engine.freeze_gate import freeze_selection, require_frozen_selection, frozen_dataset_path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class FreezeGateTests(unittest.TestCase):
    def setUp(self):
        temp_base = ROOT / 'tmp'
        temp_base.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temp_base)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'package'
        self.run = self.root / 'runs/person/r1'
        self.config = {'version': 'v4', 'task': 'detect', 'model_key': 'person'}
        self.weights = self.run / 'training/weights/best.pt'
        self.weights.parent.mkdir(parents=True)
        self.weights.write_bytes(b'synthetic selected checkpoint')
        snapshot = self.run / 'metadata/evaluation_inputs/val/checkpoint.pt'
        snapshot.parent.mkdir(parents=True)
        shutil.copyfile(self.weights, snapshot)
        self.summary = self.run / 'evaluation/val/summary.json'
        save(self.summary, {'version': 'v4', 'task': 'detect', 'model_key': 'person', 'split': 'val',
                            'images': 1, 'metrics': {'example': 0.5},
                            'checkpoint': self.weights.relative_to(self.root).as_posix(),
                            'checkpoint_snapshot': snapshot.relative_to(self.root).as_posix()})
        self.status = self.run / 'metadata/evaluate_val_status.json'
        save(self.status, {'phase': 'evaluate_val', 'status': 'complete'})
        save(self.run / 'config/evaluate_val.json', self.config)
        data = self.run / 'data'
        records = []
        for split in ('train', 'val', 'test'):
            image, label = data / 'images' / split / 'a.png', data / 'labels' / split / 'a.txt'
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(('synthetic ' + split).encode())
            label.write_text('0 0.5 0.5 0.1 0.1\n', encoding='utf-8')
            records.append({'split': split, 'run_image': image.relative_to(self.run).as_posix(),
                            'run_label': label.relative_to(self.run).as_posix()})
        save(data / 'samples.json', records)
        (data / 'data.yaml').write_text(yaml.safe_dump({'path': str(data), 'names': ['person'],
            **{split: 'images/' + split for split in ('train', 'val', 'test')}}), encoding='utf-8')
        (data / 'labels/train.cache').write_text('regenerable cache', encoding='utf-8')
        snapshot_data = self.run / 'metadata/evaluation_inputs/val/data'
        shutil.copytree(data, snapshot_data)
        snapshot_yaml = snapshot_data / 'data.yaml'
        spec = yaml.safe_load(snapshot_yaml.read_text(encoding='utf-8'))
        spec['path'] = str(snapshot_data)
        snapshot_yaml.write_text(yaml.safe_dump(spec), encoding='utf-8')
        summary = json.loads(self.summary.read_text(encoding='utf-8'))
        summary['dataset_snapshot'] = snapshot_yaml.relative_to(self.root).as_posix()
        save(self.summary, summary)

    def freeze(self):
        return freeze_selection(self.root, self.config, self.run, self.weights, 'val', 'fixture validation review')

    def test_freeze_records_not_executed_and_only_selected_checkpoint_allowed(self):
        selected = self.freeze()
        record = json.loads((self.run / 'frozen_selection/record.json').read_text(encoding='utf-8'))
        self.assertEqual(record['selection_reason'], 'fixture validation review')
        state = json.loads((self.run / 'metadata/final_test_status.json').read_text(encoding='utf-8'))
        self.assertEqual(state['status'], 'not_executed')
        self.assertEqual(require_frozen_selection(self.root, self.config, self.run, selected)['status'], 'frozen')
        with self.assertRaisesRegex(ValueError, 'frozen selected checkpoint'):
            require_frozen_selection(self.root, self.config, self.run, self.weights)
        with self.assertRaises(FileExistsError):
            self.freeze()

    def test_missing_selection_reason_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'selection reason'):
            freeze_selection(self.root, self.config, self.run, self.weights, 'val', '  ')

    def test_val_or_config_change_rejects_test(self):
        selected = self.freeze()
        self.summary.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'evidence'):
            require_frozen_selection(self.root, self.config, self.run, selected)

    def test_unverified_and_changed_checkpoint_fail_before_freeze(self):
        save(self.status, {'phase': 'evaluate_val', 'status': 'failed'})
        with self.assertRaisesRegex(ValueError, 'did not complete'):
            self.freeze()
        save(self.status, {'phase': 'evaluate_val', 'status': 'complete'})
        self.weights.write_bytes(b'changed candidate')
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            self.freeze()
        self.assertFalse((self.run / 'frozen_selection').exists())

    def test_wrong_run_and_cross_run_evidence_rejected(self):
        other = self.root / 'runs/person/r2'
        other.mkdir(parents=True)
        with self.assertRaises((ValueError, FileNotFoundError)):
            freeze_selection(self.root, self.config, other, self.weights, 'val', 'fixture validation review')
        selected = self.freeze()
        with self.assertRaises((ValueError, FileNotFoundError)):
            require_frozen_selection(self.root, self.config, other, selected)

    def test_dataset_is_fixed_for_test_and_relocates(self):
        selected = self.freeze()
        source = self.run / 'data/images/test/a.png'
        source.write_bytes(b'changed source after freeze')
        frozen = self.run / 'frozen_selection/data/images/test/a.png'
        self.assertEqual(frozen.read_bytes(), b'synthetic test')
        self.assertFalse((self.run / 'frozen_selection/data/labels/train.cache').exists())
        moved = self.root.parent / 'copied package with spaces'
        shutil.copytree(self.root, moved)
        moved_run = moved / self.run.relative_to(self.root)
        yaml_path = frozen_dataset_path(moved, self.config, moved_run, selected)
        self.assertEqual(Path(yaml.safe_load(yaml_path.read_text(encoding='utf-8'))['path']),
                         moved_run / 'frozen_selection/data')
        self.assertEqual((yaml_path.parent / 'images/test/a.png').read_bytes(), b'synthetic test')

    def test_wrong_split_and_candidate_are_rejected(self):
        summary = json.loads(self.summary.read_text(encoding='utf-8'))
        summary['split'] = 'test'
        save(self.summary, summary)
        with self.assertRaisesRegex(ValueError, 'split'):
            self.freeze()
        summary['split'] = 'val'
        summary['checkpoint'] = 'runs/person/other.pt'
        save(self.summary, summary)
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.freeze()

    def test_dataset_cannot_point_outside_frozen_split(self):
        data_yaml = self.run / 'metadata/evaluation_inputs/val/data/data.yaml'
        data = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
        data['test'] = '../external'
        data_yaml.write_text(yaml.safe_dump(data), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'local train/val/test'):
            self.freeze()

    def test_changed_frozen_data_is_rejected(self):
        selected = self.freeze()
        (self.run / 'frozen_selection/data/images/test/a.png').write_bytes(b'changed frozen input')
        with self.assertRaisesRegex(ValueError, 'differs from validation dataset'):
            require_frozen_selection(self.root, self.config, self.run, selected)

    def test_validation_snapshot_must_belong_to_same_evaluation(self):
        summary = json.loads(self.summary.read_text(encoding='utf-8'))
        summary['dataset_snapshot'] = 'runs/person/r2/data/data.yaml'
        save(self.summary, summary)
        with self.assertRaisesRegex(ValueError, 'outside this evaluation'):
            self.freeze()

    def test_frozen_class_schema_change_is_rejected(self):
        selected = self.freeze()
        data_yaml = self.run / 'frozen_selection/data/data.yaml'
        data = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
        data['names'] = ['wrong_class']
        data_yaml.write_text(yaml.safe_dump(data), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'differs from validation dataset'):
            require_frozen_selection(self.root, self.config, self.run, selected)

    def test_existing_not_executed_status_is_preserved_but_started_test_rejected(self):
        state = self.run / 'metadata/final_test_status.json'
        save(state, {'status': 'running'})
        with self.assertRaisesRegex(ValueError, 'already started'):
            self.freeze()
        save(state, {'status': 'not_executed', 'source': 'training'})
        self.freeze()
        after = json.loads(state.read_text(encoding='utf-8'))
        self.assertEqual(after['status'], 'not_executed')
        self.assertEqual(after['source'], 'training')
        self.assertEqual(after['selection'], 'frozen_selection/record.json')


if __name__ == '__main__':
    unittest.main()
