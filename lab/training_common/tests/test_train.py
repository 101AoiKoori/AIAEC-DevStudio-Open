import importlib.util
import csv
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml

PATH = Path(__file__).resolve().parents[1] / 'train.py'
SPEC = importlib.util.spec_from_file_location('portable_train', PATH)
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


class PortableTrainingChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        for directory in ('config', 'dataset', 'weights'):
            (self.root / directory).mkdir()
        (self.root / 'weights/model.pt').write_bytes(b'fixture-not-a-real-model')
        self.config = dict(task='detect', imgsz=640, epochs=100, data='dataset/data.yaml',
                           model='weights/model.pt')
        self.data = dict(train='images/train', val='images/val', test='images/test',
                         names={0: 'upright', 1: 'fallen'})
        for split in train.SPLITS:
            image_dir = self.root / 'dataset/images' / split
            label_dir = self.root / 'dataset/labels' / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            (image_dir / '001.jpg').write_bytes(b'fixture-image')
            (label_dir / '001.txt').write_text('0 0.5 0.5 0.2 0.4\n', encoding='utf-8')
        self.write_config()

    def write_config(self):
        for file, data in (('config/train.yaml', self.config),
                           ('dataset/data.yaml', self.data)):
            (self.root / file).write_text(yaml.safe_dump(data), encoding='utf-8')

    def test_valid_datasets_and_empty_negative(self):
        (self.root / 'dataset/labels/test/001.txt').write_text('', encoding='utf-8')
        _, _, _, counts = train.preflight(self.root)
        self.assertEqual(counts, {'train': 1, 'val': 1, 'test': 1})

    def test_same_images_across_splits_need_no_source_manifest(self):
        self.assertFalse((self.root / 'dataset/sources.yaml').exists())
        self.assertFalse((self.root / 'dataset/assignments.jsonl').exists())
        _, _, _, counts = train.preflight(self.root)
        self.assertEqual(counts, {'train': 1, 'val': 1, 'test': 1})

    def test_missing_labels_and_out_of_bounds_are_rejected(self):
        label = self.root / 'dataset/labels/train/001.txt'
        label.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing label'):
            train.preflight(self.root)
        label.write_text('0 0.99 0.5 0.5 0.5', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'exceeds image'):
            train.preflight(self.root)

    def test_local_model_and_640_are_mandatory(self):
        self.config['model'] = '../model.pt'
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'escapes package'):
            train.preflight(self.root)
        self.config['model'] = 'weights/missing.pt'
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'missing'):
            train.preflight(self.root)
        self.config['imgsz'] = 320
        self.write_config()
        with self.assertRaisesRegex(ValueError, '640'):
            train.preflight(self.root)

    def test_pose_keypoints(self):
        self.config['task'] = 'pose'
        self.data['kpt_shape'] = [8, 3]
        self.write_config()
        for split in train.SPLITS:
            label = self.root / 'dataset/labels' / split / '001.txt'
            label.write_text('0 0.5 0.5 0.2 0.4' + ' 0.5 0.5 2' * 8, encoding='utf-8')
        train.preflight(self.root)
        (self.root / 'dataset/labels/val/001.txt').write_text('0 0.5 0.5 0.2 0.4', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'bad field count'):
            train.preflight(self.root)

    def test_arguments_and_rejected_training_use_fake_model(self):
        calls, model_paths = [], []
        run_dir = self.root / 'runs/train'
        test_dir = run_dir / 'test'

        class FakeModel:
            def __init__(self, path):
                self.path = Path(path)
                self.trainer = types.SimpleNamespace(save_dir=run_dir)

            def train(self, **kwargs):
                calls.append(('train', kwargs))
                (run_dir / 'weights').mkdir(parents=True)
                (run_dir / 'weights/best.pt').write_bytes(b'fake-checkpoint')

            def add_callback(self, name, callback):
                self.callback = callback

            def val(self, **kwargs):
                calls.append(('test', kwargs))
                (test_dir / 'labels').mkdir(parents=True)
                self.callback(types.SimpleNamespace(seen=1, dataloader=types.SimpleNamespace(dataset=[1])))
                box = types.SimpleNamespace(ap_class_index=[0], p=[.8], r=[.7], f1=[.74], ap50=[.75], ap=[.6])
                return types.SimpleNamespace(save_dir=test_dir, box=box,
                    results_dict={'metrics/precision(B)': .8}, speed={'inference': 1.2})

        fake_ultralytics = types.ModuleType('ultralytics')
        def fake_yolo(path, **kwargs):
            model_paths.append(path)
            return FakeModel(path)
        fake_ultralytics.YOLO = fake_yolo
        fake_callbacks = types.ModuleType('ultralytics.utils.callbacks')
        class FakeWriter:
            def __init__(self, log_dir): pass
            def add_scalar(self, *args): pass
            def flush(self): pass
            def close(self): pass
        fake_callbacks.tensorboard = types.SimpleNamespace(SummaryWriter=FakeWriter)
        modules = {'ultralytics': fake_ultralytics,
                   'ultralytics.utils': types.ModuleType('ultralytics.utils'),
                   'ultralytics.utils.callbacks': fake_callbacks}
        with patch.dict('sys.modules', modules), patch.object(train, '_save_class_plot', return_value=True):
            train.train(self.root)
            self.assertEqual(calls[0][1]['imgsz'], 640)
            self.assertEqual(calls[0][1]['epochs'], 100)
            self.assertFalse(calls[0][1]['exist_ok'])
            self.assertEqual(calls[1][0], 'test')
            self.assertEqual(calls[1][1]['split'], 'test')
            self.assertEqual(calls[1][1]['conf'], .001)
            self.assertTrue(calls[1][1]['plots'])
            self.assertTrue(calls[1][1]['save_json'])
            self.assertTrue(calls[1][1]['save_txt'])
            self.assertEqual(Path(model_paths[1]), run_dir / 'weights/best.pt')
            summary = json.loads((test_dir / 'test_summary.json').read_text(encoding='utf8'))
            self.assertEqual(summary['split'], 'test')
            self.assertTrue(summary['negative_false_positives']['coverage_confirmed'])
            self.assertIn('not fixed-threshold 0.25', summary['metric_definitions']['precision_recall_f1'])
            self.config['imgsz'] = 320
            self.write_config()
            with self.assertRaisesRegex(ValueError, '640'):
                train.train(self.root)
            self.assertEqual(len(calls), 2)

    def test_pose_report_class_indices_and_unavailable_metrics(self):
        self.config['task'] = 'pose'
        self.data['kpt_shape'] = [8, 3]
        self.write_config()
        for split in train.SPLITS:
            (self.root / 'dataset/labels' / split / '001.txt').write_text(
                '0 0.5 0.5 0.2 0.4' + ' 0.5 0.5 2' * 8, encoding='utf-8')
        destination = self.root / 'test-output'
        (destination / 'labels').mkdir(parents=True)
        box = types.SimpleNamespace(ap_class_index=[0], p=[.8], r=[.7], f1=[.74], ap50=[.75], ap=[.6])
        pose = types.SimpleNamespace(ap_class_index=[], p=[], r=[], f1=[], ap50=[], ap=[])
        metrics = types.SimpleNamespace(box=box, pose=pose, speed={}, results_dict={
            'metrics/precision(B)': .8, 'metrics/precision(P)': 0.0})
        with patch.object(train, '_save_class_plot', return_value=True):
            summary = train.write_test_report(metrics, self.root/'dataset/data.yaml', 'pose', destination,
                                              {'seen': 1, 'dataset_size': 1})
        self.assertEqual(summary['metrics']['metrics/precision(B)'], .8)
        self.assertIsNone(summary['metrics']['metrics/precision(P)'])
        with (destination/'per_class_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['box_ap50'], '0.75')
        self.assertEqual(rows[0]['pose_ap50'], '')
        self.assertEqual(rows[1]['box_ap50'], '')
        self.assertTrue(any('not individual keypoints' in s for s in summary['notes']))

    def test_negative_false_positive_coverage_and_threshold(self):
        labels = self.root / 'dataset/labels/test'
        images = self.root / 'dataset/images/test'
        (labels/'001.txt').write_text('', encoding='utf8')
        (labels/'002.txt').write_text('', encoding='utf8')
        (images/'002.jpg').write_bytes(b'fixture')
        destination = self.root / 'test-output'
        (destination/'labels').mkdir(parents=True)
        (destination/'labels/001.txt').write_text('0 .5 .5 .2 .2 .9\n1 .5 .5 .2 .2 .1\n', encoding='utf8')
        result = train._negative_false_positives(self.root/'dataset/data.yaml', destination,
                                                 {'seen': 2, 'dataset_size': 2})
        self.assertEqual(result['negative_images'], 2)
        self.assertEqual(result['images_with_false_positives'], 1)
        self.assertEqual(result['false_positive_boxes'], 1)
        with (destination/'negative_false_positives.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual([r['predicted_boxes'] for r in rows], ['1', '0'])
        incomplete = train._negative_false_positives(self.root/'dataset/data.yaml', destination,
                                                     {'seen': 1, 'dataset_size': 2})
        self.assertFalse(incomplete['coverage_confirmed'])
        self.assertIsNone(incomplete['false_positive_boxes'])
        with (destination/'negative_false_positives.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual([r['predicted_boxes'] for r in rows], ['1', ''])

    def test_negative_predictions_invalid_confidence_fails(self):
        (self.root/'dataset/labels/test/001.txt').write_text('', encoding='utf8')
        destination = self.root / 'test-output'
        (destination/'labels').mkdir(parents=True)
        (destination/'labels/001.txt').write_text('0 .5 .5 .2 .2 nan\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'Invalid prediction confidence'):
            train._negative_false_positives(self.root/'dataset/data.yaml', destination,
                                            {'seen': 1, 'dataset_size': 1})
        (destination/'labels/001.txt').write_text('0.9\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'Invalid prediction confidence'):
            train._negative_false_positives(self.root/'dataset/data.yaml', destination,
                                            {'seen': 1, 'dataset_size': 1})

    def test_tensorboard_writes_only_available_metrics_and_closes(self):
        written=[]
        class FakeWriter:
            def __init__(self, log_dir): written.append(('open', log_dir))
            def add_scalar(self, *args): written.append(args)
            def flush(self): written.append(('flush',))
            def close(self): written.append(('close',))
        summary={'metrics': {'metrics/mAP50(B)': .8, 'metrics/mAP50(P)': None},
                 'negative_false_positives': {'negative_images': 3, 'evaluated_negative_images': 2,
                    'images_with_false_positives': None, 'false_positive_boxes': None}}
        rows=[{'class_id': 1, 'box_ap50': .7, 'box_recall': None, 'pose_ap50': float('nan')}]
        train._write_test_tensorboard(summary, rows, self.root, FakeWriter)
        tags=[entry[0] for entry in written]
        self.assertIn('test/overall/metrics/mAP50(B)', tags)
        self.assertIn('test/class_1/box_ap50', tags)
        self.assertNotIn('test/overall/metrics/mAP50(P)', tags)
        self.assertNotIn('test/class_1/box_recall', tags)
        self.assertNotIn('test/class_1/pose_ap50', tags)
        self.assertNotIn('test/negative_fp_025/false_positive_boxes', tags)
        self.assertEqual(tags[-1], 'close')

    def test_tensorboard_event_file_contains_simulated_test_metrics(self):
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        summary={'metrics': {'metrics/mAP50(B)': .8}, 'negative_false_positives': {
            'negative_images': 3, 'evaluated_negative_images': 3,
            'images_with_false_positives': 1, 'false_positive_boxes': 2}}
        train._write_test_tensorboard(summary, [{'class_id': 0, 'box_ap50': .7}], self.root, SummaryWriter)
        events=EventAccumulator(str(self.root));events.Reload()
        tags=events.Tags()['scalars']
        self.assertIn('test/overall/metrics/mAP50(B)', tags)
        self.assertIn('test/class_0/box_ap50', tags)
        self.assertIn('test/negative_fp_025/false_positive_boxes', tags)
        self.assertAlmostEqual(events.Scalars('test/class_0/box_ap50')[0].value, .7, places=5)

    def test_per_class_chart_uses_missing_metrics_without_faking_zeros(self):
        rows = [{'name': 'upright', 'instances': 3, 'box_precision': .8,
                 'box_recall': .6, 'box_ap50': .7, 'box_map50_95': .5},
                {'name': 'fallen', 'instances': 1, 'box_precision': None,
                 'box_recall': None, 'box_ap50': None, 'box_map50_95': None}]
        plot = self.root / 'per_class.png'
        self.assertTrue(train._save_class_plot(rows, 'box', plot))
        self.assertTrue(plot.is_file())
        self.assertGreater(plot.stat().st_size, 1000)
        self.assertFalse(train._save_class_plot(rows, 'pose', self.root/'missing.png'))

    def test_missing_best_checkpoint_prevents_test_and_reports_failure(self):
        calls=[]
        class FakeModel:
            def __init__(self):
                self.trainer=types.SimpleNamespace(save_dir=self_root/'runs/train')
            def train(self, **kwargs):
                calls.append('train')
        self_root=self.root
        fake=types.ModuleType('ultralytics')
        fake.YOLO=lambda *args, **kwargs: FakeModel()
        callbacks=types.ModuleType('ultralytics.utils.callbacks')
        callbacks.tensorboard=types.SimpleNamespace(SummaryWriter=object())
        modules={'ultralytics':fake, 'ultralytics.utils':types.ModuleType('ultralytics.utils'),
                 'ultralytics.utils.callbacks':callbacks}
        with patch.dict('sys.modules', modules):
            with self.assertRaisesRegex(RuntimeError, 'without a best.pt'):
                train.train(self.root)
        self.assertEqual(calls, ['train'])

    def test_false_test_flag_skips_test_and_rejects_non_boolean(self):
        self.config['test_after_training']=False
        self.write_config()
        calls=[]
        class FakeModel:
            def train(self, **kwargs): calls.append('train')
        fake=types.ModuleType('ultralytics')
        fake.YOLO=lambda *args, **kwargs: FakeModel()
        callbacks=types.ModuleType('ultralytics.utils.callbacks')
        callbacks.tensorboard=types.SimpleNamespace(SummaryWriter=object())
        modules={'ultralytics':fake, 'ultralytics.utils':types.ModuleType('ultralytics.utils'),
                 'ultralytics.utils.callbacks':callbacks}
        with patch.dict('sys.modules', modules): train.train(self.root)
        self.assertEqual(calls, ['train'])
        self.config['test_after_training']='false'
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'must be true or false'): train.preflight(self.root)


if __name__ == '__main__':
    unittest.main()
