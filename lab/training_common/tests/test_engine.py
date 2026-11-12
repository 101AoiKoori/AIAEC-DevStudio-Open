"""Checks for split provenance, output ownership and diagnostic matching."""
from pathlib import Path
import json
import os
import sys
import tempfile
import types
import time
import unittest
from unittest.mock import patch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.config import validate
from engine.data import records, select_records, stage_dataset, relocate_staged_data
from engine.reporting import match_detections, artifact_manifest
from engine.runtime import new_run, run_context
from engine.runner import train_run, assert_model_contract, compare_runs, resume_configuration
from engine.bounded import BudgetExhausted, run_bounded


class EngineContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {'schema_version': 3, 'task_id': 'person', 'model_key': 'person',
                       'version': 'v1', 'task': 'detect', 'data': 'dataset/data.yaml',
                       'model': 'model.pt', 'output_root': 'runs/person', 'imgsz': 320,
                       'epochs': 1, 'batch': 2, 'workers': 0, 'seed': 42, 'train_limit': 2}
        (self.root / 'model.pt').write_bytes(b'fixture-only')
        for split in ('train', 'val', 'test'):
            for directory in ('images', 'labels'):
                (self.root / 'dataset' / directory / split).mkdir(parents=True)
            for index in range(4):
                (self.root / f'dataset/images/{split}/{index}.jpg').write_bytes(b'image fixture')
                (self.root / f'dataset/labels/{split}/{index}.txt').write_text(
                    f'{index % 2} 0.5 0.5 0.2 0.2\n' if index != 3 else '', encoding='utf-8')
        spec = {'names': ['first', 'second'], **{s: 'images/' + s for s in ('train', 'val', 'test')}}
        (self.root / 'dataset/data.yaml').write_text(yaml.safe_dump(spec), encoding='utf-8')

    def test_staging_limits_only_train_and_preserves_source(self):
        source = self.root / 'dataset/labels/train/0.txt'
        before = source.read_bytes()
        run = new_run(self.root, self.config, 'one')
        path, report = stage_dataset(self.root, self.config, run)
        self.assertEqual(report['selected_images'], {'train': 2, 'val': 4, 'test': 4})
        manifest = json.loads((run / 'data/samples.json').read_text(encoding='utf-8'))
        sample = next(row for row in manifest if row['split'] == 'train')
        (run / sample['run_label']).write_text('', encoding='utf-8')
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(yaml.safe_load(path.read_text())['names'], ['first', 'second'])

    def test_seeded_subset_preserves_rare_class(self):
        items = records(self.root, self.config, 'train')
        first = select_records(items, 2, 42)
        self.assertEqual(first, select_records(items, 2, 42))
        self.assertEqual({cls for row in first for cls in row['classes']}, {0, 1})

    def test_copied_package_retains_original_source_group(self):
        manifest = {'records': [{'image': 'dataset/images/train/0.jpg',
                                'source_group': 'original-session', 'workspace_image': 'datasets/exports/original.jpg'}]}
        (self.root / 'dataset/workspace_sources.json').write_text(json.dumps(manifest), encoding='utf-8')
        row = records(self.root, self.config, 'train')[0]
        self.assertEqual(row['source_group'], 'original-session')
        self.assertEqual(row['workspace_image'], 'datasets/exports/original.jpg')

    def test_unsupported_option_and_overwrite_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            validate({**self.config, 'learn_rate_typo': .01})
        new_run(self.root, self.config, 'existing')
        with self.assertRaises(FileExistsError): new_run(self.root, self.config, 'existing')

    def test_output_cannot_escape_runs_by_traversal_or_resolution(self):
        with self.assertRaisesRegex(ValueError, 'cannot contain'):
            validate({**self.config, 'output_root': 'runs/../docs'})
        with patch('engine.config.resolve', return_value=self.root / 'docs'):
            with self.assertRaisesRegex(ValueError, 'Resolved output_root'):
                new_run(self.root, self.config, 'escape')
        self.assertFalse((self.root / 'docs').exists())

    def test_display_alias_does_not_allow_reversed_source_classes(self):
        source = yaml.safe_load((self.root / 'dataset/data.yaml').read_text())
        source['names'] = ['second', 'first']
        (self.root / 'dataset/reversed.yaml').write_text(yaml.safe_dump(source), encoding='utf-8')
        config = {**self.config, 'class_names': ['display first', 'display second'],
                  'data_overrides': {'train': [{'data':'dataset/reversed.yaml','split':'train'}]}}
        with self.assertRaisesRegex(ValueError, 'Sources disagree'):
            records(self.root, config, 'train')

    def test_orphan_and_missing_labels_rejected(self):
        (self.root / 'dataset/labels/train/absent.txt').write_text('', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Labels without images'):
            records(self.root, self.config, 'train')
        (self.root / 'dataset/labels/train/absent.txt').unlink()
        (self.root / 'dataset/labels/train/0.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing label'):
            records(self.root, self.config, 'train')

    def test_relocation_refreshes_yaml_and_run_cache_only(self):
        run = new_run(self.root, self.config, 'relocate')
        stage_dataset(self.root, self.config, run)
        cache = run / 'data/labels/train.cache'; cache.write_bytes(b'cache')
        moved = run.with_name('moved'); run.rename(moved)
        path = relocate_staged_data(moved)
        self.assertEqual(yaml.safe_load(path.read_text())['path'], str(moved / 'data'))
        self.assertFalse((moved / 'data/labels/train.cache').exists())

    def test_duplicate_prediction_is_false_positive(self):
        truth = [{'class_id': 0, 'xyxy': [0, 0, 10, 10]}]
        preds = [{'class_id': 0, 'xyxy': [0, 0, 10, 10]}, {'class_id': 0, 'xyxy': [0, 0, 10, 10]},
                 {'class_id': 1, 'xyxy': [0, 0, 10, 10]}]
        matches, missed, false = match_detections(truth, preds, .5)
        self.assertEqual((len(matches), len(missed), len(false)), (1, 0, 2))

    def test_resume_rejects_changed_training_semantics(self):
        run = new_run(self.root, self.config, 'resume')
        (run / 'training/weights').mkdir(parents=True)
        (run / 'training/weights/last.pt').write_bytes(b'fixture-checkpoint')
        (run / 'config/effective.json').write_text(json.dumps(self.config), encoding='utf-8')
        changed = {**self.config, 'optimizer': 'SGD'}
        with self.assertRaisesRegex(ValueError, 'Resume config differs'):
            train_run(self.root, changed, resume='runs/person/resume/training/weights/last.pt')

    def test_package_run_resumes_using_frozen_inputs_after_copy_to_workspace(self):
        run = new_run(self.root, self.config, 'copied_package_run')
        stage_dataset(self.root, self.config, run)
        (run / 'training/weights').mkdir(parents=True)
        checkpoint = run / 'training/weights/last.pt'; checkpoint.write_bytes(b'fixture-checkpoint')
        package_config = {**self.config, 'model':'weights/pretrained.pt',
                          'source_notes':['Copied package source records']}
        (run / 'config/effective.json').write_text(json.dumps(package_config), encoding='utf-8')
        workspace_config = {**self.config, 'data':'datasets/exports/unavailable/data.yaml',
                            'model':'models/weights/unavailable.pt', 'data_overrides':{'train':[{'data':'unused.yaml'}]},
                            'source_notes':['Workspace source records'], 'batch':4, 'device':'cpu'}
        effective, mapping = resume_configuration(self.root, workspace_config, run, checkpoint)
        self.assertEqual(effective['data'], 'runs/person/copied_package_run/data/data.yaml')
        self.assertEqual(effective['model'], 'runs/person/copied_package_run/training/weights/last.pt')
        self.assertEqual(effective['batch'], 4)
        self.assertNotIn('data_overrides', effective)
        self.assertEqual(effective['source_notes'], ['Copied package source records'])
        self.assertIn('data_overrides', mapping['ignored_source_fields'])
        with self.assertRaisesRegex(ValueError, 'training semantics'):
            resume_configuration(self.root, {**workspace_config,'epochs':2}, run, checkpoint)

    def test_wall_budget_stops_owned_child(self):
        started = time.monotonic()
        with self.assertRaises(BudgetExhausted):
            run_bounded([sys.executable,'-B','-c','import time; time.sleep(20)'],
                        time.monotonic()+.15, self.root / 'runs/child.log', self.root)
        self.assertLess(time.monotonic()-started, 10)

    def test_failure_preserves_status_and_restores_process_environment(self):
        run = new_run(self.root, self.config, 'failure')
        before_cwd, before_temp = Path.cwd(), tempfile.tempdir
        before_config = os.environ.get('YOLO_CONFIG_DIR')
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            with run_context(self.root, self.config, run, 'check_failure'):
                self.assertEqual(Path.cwd(), run)
                raise RuntimeError('fixture failure')
        state = json.loads((run / 'metadata/status.json').read_text(encoding='utf-8'))
        self.assertEqual(state['status'], 'failed')
        self.assertIn('fixture failure', (run / 'logs/check_failure.log').read_text(encoding='utf-8'))
        self.assertEqual((Path.cwd(), tempfile.tempdir, os.environ.get('YOLO_CONFIG_DIR')),
                         (before_cwd, before_temp, before_config))

    def test_offline_context_disables_proxy_and_restores_it(self):
        run = new_run(self.root, self.config, 'offline')
        with patch.dict(os.environ, {'HTTPS_PROXY':'http://127.0.0.1:1234', 'NO_PROXY':'localhost'}):
            with run_context(self.root, self.config, run, 'offline_check'):
                self.assertNotIn('HTTPS_PROXY', os.environ)
                self.assertEqual(os.environ['YOLO_OFFLINE'], 'true')
                self.assertEqual(os.environ['NO_PROXY'], '*')
            self.assertEqual(os.environ['HTTPS_PROXY'], 'http://127.0.0.1:1234')
            self.assertEqual(os.environ['NO_PROXY'], 'localhost')

    def test_evaluation_rejects_wrong_class_order_task_and_keypoints(self):
        path = self.root / 'dataset/data.yaml'
        model = types.SimpleNamespace(names={0:'first',1:'second'}, task='detect')
        assert_model_contract(model, path, self.config)
        model.names = {0:'second',1:'first'}
        with self.assertRaisesRegex(ValueError, 'class count/order'):
            assert_model_contract(model, path, self.config)
        model.names = {0:'first',1:'second'}; model.task = 'pose'
        with self.assertRaisesRegex(ValueError, 'task differs'):
            assert_model_contract(model, path, self.config)
        spec = yaml.safe_load(path.read_text()); spec['kpt_shape'] = [8,3]
        path.write_text(yaml.safe_dump(spec), encoding='utf-8')
        model.model = types.SimpleNamespace(model=[types.SimpleNamespace(kpt_shape=[7,3])])
        with self.assertRaisesRegex(ValueError, 'keypoint shape'):
            assert_model_contract(model, path, {**self.config,'task':'pose'})

    def test_compare_marks_different_sources_incomparable(self):
        for index in (1, 2):
            run = new_run(self.root, self.config, f'comparison_input_{index}')
            (run / 'metadata/environment.json').write_text(json.dumps({'packages': {'ultralytics':'8.2.103'}}), encoding='utf-8')
            directory = run / 'evaluation/val'; directory.mkdir(parents=True)
            (directory / 'summary.json').write_text(json.dumps({'model_key':'person', 'version':'v'+str(index),
                'split':'val','images':1,'metrics':{'metrics/mAP50-95(B)':0.2},
                'evaluation_sources':[f'dataset/image_{index}.jpg'], 'evaluation_config':{'class_names':['first','second']}}), encoding='utf-8')
        result = compare_runs(self.root, self.config,
                              ['runs/person/comparison_input_1','runs/person/comparison_input_2'], 'comparison_result')
        content = json.loads((result / 'comparison.json').read_text())
        self.assertFalse(content['groups']['person/val']['comparable'])
        self.assertFalse(list(result.glob('comparison_*.png')))

    def test_artifact_inventory_explains_absence_without_rewriting_results(self):
        from PIL import Image
        evaluation = self.root / 'runs/frozen/evaluation/test'
        evaluation.mkdir(parents=True)
        original_summary = b'{"task":"detect","split":"test","metrics":{"ap50":0}}\n'
        (evaluation / 'summary.json').write_bytes(original_summary)
        original_curve = b'original native curve fixture'
        (evaluation / 'PR_curve.png').write_bytes(original_curve)
        path = artifact_manifest(evaluation, postprocess_name='availability_test')
        content = json.loads(path.read_text(encoding='utf-8'))
        entries = {entry['id']: entry for entry in content['artifacts']}
        self.assertEqual(entries['box_pr_curve']['status'], 'present')
        self.assertEqual(entries['box_pr_curve']['files'], ['PR_curve.png'])
        self.assertEqual(entries['box_f1_curve']['status'], 'missing')
        self.assertEqual(entries['box_per_class']['status'], 'missing')
        self.assertNotIn('explanation_file', entries['box_pr_curve'])
        with Image.open(path.parent / content['explanation_file']) as picture:
            self.assertEqual(picture.format, 'PNG')
            self.assertGreater(picture.width, 100)
        self.assertEqual((evaluation / 'summary.json').read_bytes(), original_summary)
        self.assertEqual((evaluation / 'PR_curve.png').read_bytes(), original_curve)
        self.assertFalse((evaluation / 'F1_curve.png').exists())
        self.assertEqual(content['tool']['revision'], 'artifact_availability_v1')
        with self.assertRaises(FileExistsError):
            artifact_manifest(evaluation, postprocess_name='availability_test')

    def test_artifact_inventory_recognizes_pose_prefixes_and_preserves_all_plots(self):
        evaluation = self.root / 'runs/pose/evaluation/val'
        evaluation.mkdir(parents=True)
        (evaluation / 'summary.json').write_text('{"task":"pose","split":"val"}', encoding='utf-8')
        files = [prefix + curve + '_curve.png' for prefix in ('Box', 'Pose') for curve in ('PR','F1','P','R')]
        files += ['box_per_class.png','pose_per_class.png','confusion_matrix.png',
                  'confusion_matrix_normalized.png','diagnostic_counts.png']
        for filename in files:
            (evaluation / filename).write_bytes(filename.encode())
        path = artifact_manifest(evaluation, postprocess_name='availability_pose')
        content = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(content['missing_count'], 0)
        self.assertIsNone(content['explanation_file'])
        self.assertTrue(all(entry['status'] == 'present' for entry in content['artifacts']))
        self.assertFalse(list(path.parent.glob('*.png')))
        self.assertTrue(all((evaluation / name).read_bytes() == name.encode() for name in files))


if __name__ == '__main__':
    unittest.main()
