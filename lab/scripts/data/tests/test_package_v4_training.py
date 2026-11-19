"""Exercise V4 package creation with temporary audited files, without training."""
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKSPACE / 'lab/scripts/data'))
import package_v4_training as builder


class V4TrainingPackage(unittest.TestCase):
    def setUp(self):
        scratch = WORKSPACE / 'tmp'
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix='v4-package-test-', dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.common = self.root / builder.COMMON
        shutil.copytree(WORKSPACE / builder.COMMON / 'engine', self.common / 'engine',
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(WORKSPACE / builder.COMMON / 'train.py', self.common / 'train.py')
        shutil.copy2(WORKSPACE / 'lab/train.ps1', self.root / 'lab/train.ps1')
        shutil.copy2(WORKSPACE / 'lab/tensorboard.ps1', self.root / 'lab/tensorboard.ps1')
        (self.common / 'launcher.yaml').write_text(yaml.safe_dump({
            'wall_seconds': 30, 'tasks': {'person': {'train_config': 'workspace_only.yaml', 'wall_seconds': 99}},
        }), encoding='utf-8')
        for path in builder.SCRIPTS:
            if path != 'train.py':
                shutil.copy2(WORKSPACE / builder.COMMON / path, self.common / path)
        self.portable = self.root / builder.PORTABLE
        self.portable.mkdir(parents=True)
        self.launcher_source = self.common / 'portable_env/run.ps1'
        self.launcher_source.parent.mkdir(parents=True, exist_ok=True)
        self.launcher_source.write_bytes(b'current launcher')
        (self.portable / 'site-packages').mkdir()
        (self.portable / 'site-packages' / 'fixture.py').write_text('fixture = True\n', encoding='utf-8')
        (self.portable / 'cp314' / 'site-packages').mkdir(parents=True)
        (self.portable / 'cp314' / 'site-packages' / 'fixture.py').write_text('fixture = True\n', encoding='utf-8')
        for name in ('run.ps1', 'bootstrap.py', 'manifest.json', 'requirements.lock'):
            (self.portable / name).touch()
        self.input = self.root / 'finished_dataset'
        self.input.mkdir()
        (self.root / 'initial.pt').write_bytes(b'local-file-only fixture')
        (self.input / 'source_review.json').write_text('"source checked"\n', encoding='utf-8')
        (self.input / 'label_review.json').write_text('"label checked"\n', encoding='utf-8')
        self.rows = []
        for split, color in (('train', 'black'), ('val', 'white')):
            images, labels = (self.input / component / split for component in ('images', 'labels'))
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            image, label = images / 'photo.png', labels / 'photo.txt'
            Image.new('RGB', (20, 20), color).save(image)
            label.write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf-8')
            self.rows.append({'image': image.relative_to(self.root).as_posix(),
                              'label': label.relative_to(self.root).as_posix(), 'split': split,
                              'source_id': split + '-session', 'label_status': 'reviewed',
                              'review_state': 'complete_positive',
                              'source_review_evidence': 'finished_dataset/source_review.json',
                              'review_evidence': 'finished_dataset/label_review.json'})
        self.write_rows()
        self.spec = {'names': ['community'], 'train': 'images/train', 'val': 'images/val',
                     'test': 'images/unassigned'}
        (self.input / 'data.yaml').write_text(yaml.safe_dump(self.spec), encoding='utf-8')
        self.config = {'schema_version': 3, 'task_id': 'person', 'model_key': 'person', 'version': 'v4',
                       'data': 'obsolete/data.yaml', 'model': 'obsolete.pt', 'output_root': 'runs/person',
                       'task': 'detect', 'epochs': None, 'imgsz': 640, 'batch': 4,
                       'training_policy': {'mode': 'fixed', 'save_period': 1},
                       'data_overrides': {'train': [{'data': 'irrelevant.yaml'}]}}
        self.write_config()

    def write_config(self):
        (self.root / 'train.yaml').write_text(yaml.safe_dump(self.config), encoding='utf-8')

    def write_rows(self):
        (self.input / 'workspace_sources.json').write_text(
            json.dumps({'model_key': 'person', 'records': self.rows}), encoding='utf-8')

    def package(self, **kwargs):
        return builder.package('person', 'finished_dataset/data.yaml', 'train.yaml', 'initial.pt',
                               root=self.root, **kwargs)

    def test_plan_reports_missing_files_and_writes_no_package(self):
        (self.input / 'workspace_sources.json').unlink()
        (self.root / 'initial.pt').unlink()
        report = self.package(plan_only=True)
        self.assertEqual(report['status'], 'blocked')
        self.assertEqual(report['missing_assets'], ['finished_dataset/workspace_sources.json', 'initial.pt'])
        self.assertEqual(list(self.portable.parent.iterdir()), [self.portable])

    def test_meter_package_keeps_pose_entry_reports_and_test_evaluator(self):
        from engine.meter_report import POINTS
        task_scripts = self.root / 'lab/features/meter/training/scripts'
        task_scripts.mkdir(parents=True)
        for name in ('train_meter_v4.py', 'evaluate_meter.ps1'):
            shutil.copy2(WORKSPACE / 'lab/features/meter/training/scripts' / name, task_scripts / name)
        self.spec.update(names=['meter_dial'], kpt_shape=[8, 3], flip_idx=list(range(8)))
        self.spec['test'] = 'images/test'
        (self.input/'data.yaml').write_text(yaml.safe_dump(self.spec), encoding='utf-8')
        (self.input/'images/test').mkdir(); (self.input/'labels/test').mkdir()
        Image.new('RGB', (20, 20), color='green').save(self.input/'images/test/photo.png')
        self.rows.append({**self.rows[0], 'split': 'test', 'source_id': 'test-session',
                          'image': 'finished_dataset/images/test/photo.png', 'label': 'finished_dataset/labels/test/photo.txt'})
        label = '0 .5 .5 .6 .6 ' + ' '.join(['.5 .5 1']*8) + '\n'
        for split in ('train', 'val', 'test'):
            (self.input/f'labels/{split}/photo.txt').write_text(label)
        (self.input/'workspace_sources.json').write_text(json.dumps({'model_key': 'meter', 'records': self.rows}))
        self.config.update(task_id='meter', model_key='meter', task='pose', epochs=150,
                           keypoint_names=list(POINTS), output_root='runs/meter',
                           training_policy={'mode': 'early_stopping', 'save_period': 5,
                                            'min_epochs': 100, 'patience': 20, 'min_delta': .001})
        self.write_config()
        result = builder.package('meter', 'finished_dataset/data.yaml', 'train.yaml', 'initial.pt',
                                 root=self.root, include_test=True, wall_seconds=36000)
        self.assertEqual(result['status'], 'built', result)
        package = self.root/'training_packages/meter'
        self.assertIn('from train_meter_v4 import main', (package/'scripts/train.py').read_text())
        self.assertTrue((package/'scripts/common/engine/meter_report.py').is_file())
        self.assertTrue((package/'evaluate.ps1').is_file())
        self.assertEqual(result['dataset_counts'], dict(train=1, val=1, test=1))
        command = subprocess.run([sys.executable, '-B', str(package/'scripts/train.py'),
                                  'meter-report', '--evaluation', 'runs/absent'],
                                 cwd=WORKSPACE, capture_output=True, text=True, encoding='utf-8', errors='replace')
        self.assertEqual(command.returncode, 1)
        self.assertIn(str(package), command.stderr.replace('\\\\', '\\'))
        self.assertNotIn('invalid choice', command.stderr)

    def test_launcher_status_is_reported_without_modifying_shared_environment(self):
        launcher = self.portable / 'run.ps1'
        launcher.write_bytes(b'old launcher')
        plan = self.package(plan_only=True)
        self.assertEqual(plan['status'], 'ready')
        self.assertEqual(plan['portable_launcher']['status'], 'outdated')
        self.assertEqual(launcher.read_bytes(), b'old launcher')
        built = self.package(output_name='launcher_fixture')
        self.assertEqual(built['status'], 'built')
        self.assertEqual(built['portable_launcher']['status'], 'outdated')
        self.assertEqual(launcher.read_bytes(), b'old launcher')
        manifest = json.loads((self.portable.parent / 'launcher_fixture/package_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['portable_launcher'], built['portable_launcher'])
        launcher.write_bytes(self.launcher_source.read_bytes())
        self.assertEqual(self.package(plan_only=True)['portable_launcher']['status'], 'current')
        launcher.unlink()
        missing = self.package(plan_only=True)
        self.assertEqual(missing['portable_launcher']['status'], 'missing')
        self.assertIn('training_packages/portable_env/run.ps1', missing['missing_assets'])

    def test_unavailable_source_comparison_does_not_block_packaging(self):
        self.launcher_source.unlink()
        report = self.package(plan_only=True)
        self.assertEqual(report['status'], 'ready')
        self.assertEqual(report['portable_launcher']['status'], 'unavailable')

    def test_build_reports_launcher_state_at_build_not_old_plan(self):
        launcher = self.portable / 'run.ps1'
        launcher.write_bytes(b'old launcher')
        report, plan = builder._prepare(self.root, 'person', 'finished_dataset/data.yaml',
                                        'train.yaml', 'initial.pt', 'fresh_status_fixture')
        self.assertEqual(report['portable_launcher']['status'], 'outdated')
        launcher.write_bytes(self.launcher_source.read_bytes())
        destination = self.root / 'fresh_status_stage'
        destination.mkdir()
        built = builder._populate(self.root, destination, report, plan)
        self.assertEqual(built['portable_launcher']['status'], 'current')

    def test_existing_package_and_runs_are_preserved(self):
        run = self.portable.parent / 'person/runs/existing'
        run.mkdir(parents=True)
        marker = run / 'log.txt'
        marker.write_text('existing run', encoding='utf-8')
        report = self.package()
        self.assertEqual(report['status'], 'blocked')
        self.assertIn('already exists', report['issues'][0])
        self.assertEqual(marker.read_text(encoding='utf-8'), 'existing run')

    def test_missing_class_warning_is_in_plan_and_built_package_report(self):
        self.spec['names'].append('noncommunity')
        (self.input / 'data.yaml').write_text(yaml.safe_dump(self.spec), encoding='utf-8')
        expected = [{'split': 'train', 'missing_classes': ['noncommunity'],
                     'message': 'Selected split has no positive examples for: noncommunity'}]
        plan = self.package(plan_only=True)
        self.assertEqual(plan['status'], 'ready')
        self.assertEqual(plan['class_coverage_warnings'], expected)
        built = self.package(output_name='missing_class_fixture')
        self.assertEqual(built['status'], 'built')
        self.assertEqual(built['class_coverage_warnings'], expected)

    def test_unknown_source_group_is_in_plan_and_built_package_report(self):
        self.config['training_policy']['mode'] = 'fixed_validation'
        self.write_config()
        self.rows[0]['source_id'] = None
        self.write_rows()
        plan = self.package(plan_only=True)
        self.assertEqual(plan['status'], 'ready')
        self.assertEqual(plan['source_group_unknown_counts'], {'train': 1, 'val': 0})
        self.assertEqual(plan['evaluation_scope'], 'independence_not_established')
        built = self.package(output_name='unknown_source_fixture')
        self.assertEqual(built['status'], 'built')
        self.assertEqual(built['source_group_unknown_counts'], {'train': 1, 'val': 0})
        self.assertEqual(built['evaluation_scope'], 'independence_not_established')

    def test_blocks_unknown_labels_and_legacy_v4_policy(self):
        self.rows[0]['review_state'] = 'unknown'
        self.write_rows()
        report = self.package(plan_only=True)
        self.assertEqual(report['status'], 'blocked')
        self.assertIn('complete reviewed supervision', report['issues'][0])
        del self.config['training_policy']
        self.config['epochs'] = 50
        self.write_config()
        report = self.package(plan_only=True)
        self.assertEqual(report['status'], 'blocked')
        self.assertIn('declare a training policy', report['issues'][0])

    def test_fixed_package_runs_read_only_check_without_workspace_assets(self):
        self.assertEqual(self.package(plan_only=True)['dataset_counts'], {'train': 1})
        result = self.package(output_name='person_V4_fixture')
        package = self.root / result['output']
        self.assertEqual(result['status'], 'built')
        self.assertEqual(result['portable_environment'], '../portable_env')
        self.assertFalse(result['python_included'])
        self.assertFalse((package / 'portable_env').exists())
        self.assertEqual(result['dataset_counts'], {'train': 1})
        self.assertFalse((package / 'runs').exists())
        for path in builder.SCRIPTS.values():
            self.assertTrue((package / path).is_file())
        self.assertTrue((package / 'train.ps1').is_file())
        self.assertTrue((package / 'tensorboard.ps1').is_file())
        defaults = yaml.safe_load((package / 'config/launcher.yaml').read_text(encoding='utf-8'))
        self.assertEqual(defaults['wall_seconds'], 30)
        self.assertNotIn('tasks', defaults)
        self.assertNotIn('train_config', defaults)
        for name in ('source_review.json', 'label_review.json'):
            (self.input / name).unlink()
        (self.root / 'initial.pt').unlink()
        config = yaml.safe_load((package / 'config/train.yaml').read_text(encoding='utf-8'))
        self.assertNotIn('data_overrides', config)
        completed = subprocess.run([sys.executable, '-B', str(package / 'scripts/train.py'), 'check'],
                                   capture_output=True, text=True, encoding='utf-8', cwd=self.root, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)['available_images'], {'train': 1})

    def test_distribution_moves_without_original_workspace(self):
        result = self.package(output_name='portable_fixture')
        package = self.root / result['output']
        with tempfile.TemporaryDirectory(prefix='v4-relocated-') as temporary:
            isolated = Path(temporary) / 'isolated'
            shutil.copytree(package, isolated)
            shutil.copytree(self.portable, isolated.parent / 'portable_env')
            self.assertTrue((isolated.parent / 'portable_env/run.ps1').is_file())
            self.assertTrue((isolated.parent / 'portable_env/site-packages/fixture.py').is_file())
            self.assertTrue((isolated.parent / 'portable_env/cp314/site-packages/fixture.py').is_file())
            completed = subprocess.run([sys.executable, '-I', '-B', str(isolated / 'scripts/train.py'), 'check'],
                                       capture_output=True, text=True, encoding='utf-8', cwd=isolated.parent, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)['available_images'], {'train': 1})

    def test_package_fingerprint_is_used_inside_unrelated_git_repository(self):
        result = self.package(output_name='portable_fixture')
        package = self.root / result['output']
        with tempfile.TemporaryDirectory(prefix='v4-other-git-') as temporary:
            outer = Path(temporary)
            subprocess.run(['git', 'init', '-q', str(outer)], check=True, capture_output=True)
            isolated = outer / 'training'
            shutil.copytree(package, isolated)
            query = ("import json,sys; from pathlib import Path; "
                     "root=Path(sys.argv[1]); sys.path.insert(0,str(root/'scripts/common')); "
                     "from engine.runtime import environment; print(json.dumps(environment(root)))")
            def inspect():
                run = subprocess.run([sys.executable, '-I', '-c', query, str(isolated)],
                                     capture_output=True, text=True, encoding='utf-8', cwd=outer, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr)
                return json.loads(run.stdout)
            first = inspect()
            self.assertIsNone(first['git_commit'])
            self.assertEqual(first['package_code_status'], 'verified')
            self.assertEqual(first['package_code_fingerprint'], result['code_fingerprint']['digest'])
            with (isolated / 'scripts/train.py').open('a', encoding='utf-8') as stream:
                stream.write('\n# modified after packing\n')
            self.assertEqual(inspect()['package_code_status'], 'modified')

    def test_environment_metadata_change_during_package_is_rejected(self):
        original_copy = builder._copy
        def changing_copy(source, destination, root):
            result = original_copy(source, destination, root)
            if destination.name == 'pretrained.pt':
                (self.portable / 'manifest.json').write_text('changed\n', encoding='utf-8')
            return result
        with patch.object(builder, '_copy', side_effect=changing_copy):
            with self.assertRaisesRegex(RuntimeError, 'Training source or configuration changed'):
                self.package()
        self.assertFalse((self.portable.parent / 'person').exists())

    def test_runtime_reports_selected_manifest_identity_without_blocking(self):
        from engine.runtime import environment
        cp314 = self.portable / 'cp314/manifest.json'
        cp314.write_text('3.14 manifest', encoding='utf-8')
        result = self.package(output_name='env_identity')
        package = self.root / result['output']
        expected = hashlib.sha256(cp314.read_bytes()).hexdigest()
        self.assertEqual(result['portable_environment_manifests_sha256']['cp314'], expected)
        with patch.dict(os.environ, {'AEC_PORTABLE_ENV': str(cp314.parent)}):
            identity = environment(package)['portable_environment_identity']
            self.assertEqual(identity['bundle'], 'cp314')
            self.assertEqual(identity['status'], 'matched')
            self.assertEqual(identity['manifest_sha256'], expected)
            cp314.write_text('replaced', encoding='utf-8')
            self.assertEqual(environment(package)['portable_environment_identity']['status'], 'changed')
            cp314.unlink()
            self.assertEqual(environment(package)['portable_environment_identity']['status'], 'unavailable')
        with patch.dict(os.environ, {'AEC_PORTABLE_ENV': str(self.portable)}):
            self.assertEqual(environment(package)['portable_environment_identity']['status'], 'matched')
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(environment(package)['portable_environment_identity']['status'], 'not_portable')

    def test_old_manifest_and_unknown_bundle_remain_nonblocking(self):
        from engine.runtime import environment
        result = self.package(output_name='legacy_environment')
        package = self.root / result['output']
        manifest_path = package / 'package_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        del manifest['portable_environment_manifests_sha256']
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        with patch.dict(os.environ, {'AEC_PORTABLE_ENV': str(self.portable)}):
            self.assertEqual(environment(package)['portable_environment_identity']['status'], 'matched')
        with patch.dict(os.environ, {'AEC_PORTABLE_ENV': str(self.portable / 'cp314')}):
            self.assertEqual(environment(package)['portable_environment_identity']['status'], 'unavailable')

    def test_optional_environment_manifest_appearing_during_package_is_rejected(self):
        original_copy = builder._copy
        def changing_copy(source, destination, root):
            result = original_copy(source, destination, root)
            if destination.name == 'pretrained.pt':
                (self.portable / 'cp314/manifest.json').write_text('appeared', encoding='utf-8')
            return result
        with patch.object(builder, '_copy', side_effect=changing_copy):
            with self.assertRaisesRegex(RuntimeError, 'Training source or configuration changed'):
                self.package()

    def test_early_stopping_preserves_train_and_val_with_packaged_evidence_map(self):
        source_records = self.input / 'source_records'
        source_records.mkdir()
        evidence_refs = ('review_evidence', 'source_review_evidence')
        mapping = {}
        for key in evidence_refs:
            source = self.root / self.rows[0][key]
            copied = source_records / source.name
            shutil.copy2(source, copied)
            mapping['original/' + source.name] = copied.relative_to(self.root).as_posix()
        for row in self.rows:
            row['parent_video'] = 'same_capture.mp4'
            for key in evidence_refs:
                row[key] = 'original/' + Path(row[key]).name
        self.rows[1]['development_contact'] = ['shared_diagnostic_video']
        (source_records / 'evidence_map.json').write_text(json.dumps(mapping), encoding='utf-8')
        self.write_rows()
        self.config['training_policy'] = {'mode': 'early_stopping', 'save_period': 1, 'min_epochs': 1,
                                          'patience': 2, 'min_delta': 0.001}
        self.write_config()
        report = self.package()
        self.assertEqual(report['status'], 'built')
        self.assertEqual(report['dataset_counts'], {'train': 1, 'val': 1})
        self.assertEqual(report['evaluation_scope'], 'same_source_diagnostic')
        self.assertEqual(report['development_contact_counts'], {'train': 0, 'val': 1})
        self.assertFalse(report['source_reuse'][0]['declared_in_manifest'])
        package = self.root / report['output']
        copied_map = json.loads((package / 'dataset/source_records/evidence_map.json').read_text(encoding='utf-8'))
        self.assertEqual(set(copied_map), set(mapping))
        self.assertTrue(all((package / value).is_file() for value in copied_map.values()))

    def test_failed_copy_does_not_publish_a_package(self):
        original_copy = builder._copy
        label = self.root / self.rows[0]['label']

        def changed_source(source, destination, root):
            if source == label:
                source.write_text('0 0.5 0.5 0.3 0.3\n', encoding='utf-8')
            return original_copy(source, destination, root)

        with patch.object(builder, '_copy', side_effect=changed_source):
            with self.assertRaisesRegex(RuntimeError, 'Audited label changed'):
                self.package()
        self.assertEqual(list(self.portable.parent.iterdir()), [self.portable])

    def test_optional_test_is_copied_but_excluded_from_training_selection(self):
        self.config['training_policy'] = {'mode': 'fixed_validation', 'save_period': 1}
        self.config['epochs'] = 2
        self.write_config()
        images, labels = self.input / 'images/test', self.input / 'labels/test'
        images.mkdir()
        labels.mkdir()
        Image.new('RGB', (20, 20), 'green').save(images / 'test.png')
        (labels / 'test.txt').write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf-8')
        self.rows.append({**self.rows[0], 'split': 'test', 'source_id': 'test-session',
                          'image': 'finished_dataset/images/test/test.png',
                          'label': 'finished_dataset/labels/test/test.txt'})
        self.write_rows()
        self.spec['test'] = 'images/test'
        (self.input / 'data.yaml').write_text(yaml.safe_dump(self.spec), encoding='utf-8')
        report = self.package(include_test=True, wall_seconds=7200)
        self.assertEqual(report['dataset_counts'], {'train': 1, 'val': 1, 'test': 1})
        self.assertEqual(report['training_dataset_counts'], {'train': 1, 'val': 1})
        package = self.root / report['output']
        self.assertEqual(len(list((package / 'dataset/images/test').glob('*.png'))), 1)
        launcher = yaml.safe_load((package / 'config/launcher.yaml').read_text(encoding='utf-8'))
        self.assertEqual(launcher['wall_seconds'], 7200)
        self.assertEqual(len(json.loads((package / 'dataset/workspace_sources.json').read_text('utf-8'))['records']), 3)

    def test_paper_entry_metadata_and_diagnostics_work_after_relocation(self):
        self.config.update(task_id='fire',model_key='building_name_paper',output_root='runs/building_name_paper',epochs=150,
            training_policy={'mode':'early_stopping','save_period':5,'min_epochs':120,'patience':15,'min_delta':.001})
        self.write_config();self.spec.update(names=['paper'],test='images/test')
        (self.input/'data.yaml').write_text(yaml.safe_dump(self.spec),encoding='utf8')
        for part in ('images','labels'):(self.input/part/'test').mkdir()
        Image.new('RGB',(20,20),'green').save(self.input/'images/test/test.png')
        (self.input/'labels/test/test.txt').write_text('0 0.5 0.5 0.2 0.2\n',encoding='utf8')
        self.rows.append({**self.rows[0],'split':'test','source_id':'test-session',
            'image':'finished_dataset/images/test/test.png','label':'finished_dataset/labels/test/test.txt'})
        for row in self.rows:
            row['paper']={'schema_version':1,'image_size':[20,20],
                'instances':[{'xyxy':[8,8,12,12],'building_id':'A'}],
                'scene':{'branch':'blender_photo_composite','buildings':'AB','papers':'A',
                         'yaw_degrees':-50,'camera_height_m':.14,'view_basis':'render_control_with_estimated_camera'}}
        (self.input/'workspace_sources.json').write_text(json.dumps({'model_key':'building_name_paper','records':self.rows}),encoding='utf8')
        scripts=self.root/'lab/features/fire/training/scripts';scripts.mkdir(parents=True)
        for name in ('train_building_name_paper_v4.py','paper_report.py','evaluate_building_name_paper.ps1'):
            shutil.copy2(WORKSPACE/'lab/features/fire/training/scripts'/name,scripts/name)
        report=builder.package('building_name_paper','finished_dataset/data.yaml','train.yaml','initial.pt',
            root=self.root,output_name='paper',include_test=True,wall_seconds=36000)
        self.assertEqual(report['dataset_counts'],{'train':1,'val':1,'test':1})
        self.assertEqual(report['training_dataset_counts'],{'train':1,'val':1})
        package=self.root/'training_packages/paper';isolated=self.root/'relocated/paper';shutil.copytree(package,isolated)
        self.assertTrue((isolated/'evaluate.ps1').is_file())
        self.assertEqual(yaml.safe_load((isolated/'config/launcher.yaml').read_text())['wall_seconds'],36000)
        shutil.rmtree(self.input)
        source=json.loads((isolated/'dataset/workspace_sources.json').read_text(encoding='utf8'))
        record=next(r for r in source['records'] if r['split']=='test')
        evaluation=isolated/'runs/building_name_paper/unit/evaluation/test';(evaluation/'predictions').mkdir(parents=True)
        (evaluation/'summary.json').write_text(json.dumps({'model_key':'building_name_paper','split':'test','images':1,
            'evaluation_config':{'imgsz':640},'diagnostics':{'match_iou':.5}}),encoding='utf8')
        prediction={'source_image':record['image'],'shape_hw':[20,20],
            'ground_truth':[{'class_id':0,'xyxy':[8,8,12,12]}],'predictions':[],
            'matches':[],'false_prediction_indices':[]}
        (evaluation/'predictions/predictions.jsonl').write_text(json.dumps(prediction)+'\n',encoding='utf8')
        for command in ('paper-report','paper-sizes'):
            args=[sys.executable,'-I','-B',str(isolated/'scripts/train.py'),command,
                  '--evaluation','runs/building_name_paper/unit/evaluation/test']
            if command=='paper-report':args+=['--dataset-manifest','dataset/workspace_sources.json']
            result=subprocess.run(args,cwd=self.root,capture_output=True,text=True,encoding='utf8',timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
        diagnostic=json.loads((evaluation/'paper_report.json').read_text(encoding='utf8'))
        self.assertEqual(diagnostic['image_groups']['paper_state']['partial']['images'],1)
        self.assertEqual(diagnostic['instance_groups']['yaw_control']['left_large_yaw']['missed'],1)

    def test_sign_entry_and_machine_metadata_work_after_relocation(self):
        self.config.update(task_id='sign',model_key='sign',output_root='runs/sign',epochs=2,
                           training_policy={'mode':'fixed_validation','save_period':1})
        self.write_config();self.spec.update(names=['sign_fixture'],test='images/test')
        (self.input/'data.yaml').write_text(yaml.safe_dump(self.spec),encoding='utf8')
        for part in ('images','labels'):(self.input/part/'test').mkdir()
        Image.new('RGB',(20,20),'green').save(self.input/'images/test/test.png')
        (self.input/'labels/test/test.txt').write_text('0 0.5 0.5 0.2 0.2\n',encoding='utf8')
        self.rows.append({**self.rows[0],'split':'test','source_id':'test-session',
            'image':'finished_dataset/images/test/test.png','label':'finished_dataset/labels/test/test.txt'})
        for row in self.rows:
            row['sign']={'schema_version':1,'image_size':[20,20],
                'instances':[{'xyxy':[8,8,12,12],'view':'oblique','visibility':'partial','layout':'partial','origin':'composite'}],
                'scene':{'quota_group':'partial','tone':'normal'}}
        (self.input/'workspace_sources.json').write_text(json.dumps({'model_key':'sign','records':self.rows}),encoding='utf8')
        task=self.root/'lab/features/sign/training/scripts';task.mkdir(parents=True)
        for name in ('train_sign_v4.py','sign_report.py','evaluate_sign.ps1'):
            shutil.copy2(WORKSPACE/'lab/features/sign/training/scripts'/name,task/name)
        report=builder.package('sign','finished_dataset/data.yaml','train.yaml','initial.pt',
            root=self.root,output_name='sign',include_test=True,wall_seconds=120)
        self.assertEqual(report['dataset_counts'],{'train':1,'val':1,'test':1})
        package=self.root/'training_packages/sign'
        self.assertTrue((package/'scripts/training/sign_report.py').is_file())
        self.assertTrue((package/'evaluate.ps1').is_file())
        isolated=self.root/'relocated/sign';shutil.copytree(package,isolated)
        shutil.rmtree(self.input)
        packed=json.loads((isolated/'dataset/workspace_sources.json').read_text(encoding='utf8'))
        record=next(row for row in packed['records'] if row['split']=='test')
        evaluation=isolated/'runs/sign/fixture/evaluation/test';(evaluation/'predictions').mkdir(parents=True)
        (evaluation/'summary.json').write_text(json.dumps({'model_key':'sign','split':'test','diagnostics':{'match_iou':.5}}),encoding='utf8')
        prediction={'source_image':record['image'],'shape_hw':[20,20],
            'ground_truth':[{'class_id':0,'xyxy':[8,8,12,12]}],'predictions':[]}
        (evaluation/'predictions/predictions.jsonl').write_text(json.dumps(prediction)+'\n',encoding='utf8')
        result=subprocess.run([sys.executable,'-I','-B',str(isolated/'scripts/train.py'),'sign-report',
            '--evaluation','runs/sign/fixture/evaluation/test','--dataset-manifest','dataset/workspace_sources.json'],
            cwd=self.root,capture_output=True,text=True,encoding='utf8',timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        result=json.loads((evaluation/'sign_report.json').read_text(encoding='utf8'))
        self.assertEqual(result['instance_groups']['view']['oblique']['missed'],1)

    def test_bin_state_task_entry_and_report_metadata_move_with_the_package(self):
        self.config.update(task_id='waste_bin', model_key='waste_bin_state',
                           output_root='runs/waste_bin_state', epochs=2,
                           training_policy={'mode': 'fixed_validation', 'save_period': 1})
        self.write_config()
        self.spec['names'] = ['open', 'closed']
        self.spec['test'] = 'images/test'
        (self.input / 'data.yaml').write_text(yaml.safe_dump(self.spec), encoding='utf-8')
        (self.input / 'images/test').mkdir()
        (self.input / 'labels/test').mkdir()
        Image.new('RGB', (20, 20), 'green').save(self.input / 'images/test/test.png')
        (self.input / 'labels/test/test.txt').write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf-8')
        self.rows.append({**self.rows[0], 'split': 'test', 'source_id': 'test-session',
                          'image': 'finished_dataset/images/test/test.png',
                          'label': 'finished_dataset/labels/test/test.txt'})
        metadata = {'image_size': [20, 20], 'bins': [{'color': 'green', 'state': 'open',
                                                     'bbox_xyxy': [8, 8, 12, 12]}],
                    'allocation_axes': {'view': 'frontal', 'quantity': 1},
                    'augmentation': {'value_gain': 1.0, 'saturation_gain': 1.0}}
        for row in self.rows:
            row.update(metadata, physical_parent_group=row['source_id'])
        (self.input / 'workspace_sources.json').write_text(
            json.dumps({'model_key': 'waste_bin_state', 'records': self.rows}), encoding='utf-8')
        scripts = self.root / 'lab/features/waste_bin/training/scripts'
        scripts.mkdir(parents=True)
        for name in ('train_waste_bin_state_v4.py', 'evaluate_waste_bin_state.ps1'):
            shutil.copy2(WORKSPACE / 'lab/features/waste_bin/training/scripts' / name, scripts / name)
        result = builder.package('waste_bin_state', 'finished_dataset/data.yaml', 'train.yaml',
                                 'initial.pt', root=self.root, include_test=True, wall_seconds=120)
        self.assertEqual(result['status'], 'built')
        self.assertEqual(result['dataset_counts'], {'train': 1, 'val': 1, 'test': 1})
        self.assertEqual(result['training_dataset_counts'], {'train': 1, 'val': 1})
        package = self.root / result['output']
        self.assertTrue((package / 'scripts/training/train_waste_bin_state_v4.py').is_file())
        self.assertTrue((package / 'scripts/common/engine/bin_state_report.py').is_file())
        self.assertTrue((package / 'evaluate.ps1').is_file())
        packed = json.loads((package / 'dataset/workspace_sources.json').read_text(encoding='utf-8'))
        self.assertTrue(all({key: row[key] for key in metadata} == metadata for row in packed['records']))
        self.assertEqual({row['physical_parent_group'] for row in packed['records']},
                         {'train-session', 'val-session', 'test-session'})
        with tempfile.TemporaryDirectory(prefix='v4-bin-state-relocated-', dir=WORKSPACE / 'tmp') as temporary:
            isolated = Path(temporary) / 'distribution/waste_bin_state'
            shutil.copytree(package, isolated)
            shutil.rmtree(self.input)
            for command in ('bin-state-report', 'color-state-report'):
                check = subprocess.run([sys.executable, '-I', '-B', str(isolated / 'scripts/train.py'),
                                        command, '--help'], cwd=isolated.parent, capture_output=True,
                                       text=True, encoding='utf-8', timeout=30)
                self.assertEqual(check.returncode, 0, check.stderr)
                self.assertIn('--evaluation', check.stdout)
                self.assertIn('--dataset-manifest' if command == 'bin-state-report' else '--review', check.stdout)
            check = subprocess.run([sys.executable, '-I', '-B', str(isolated / 'scripts/train.py'), 'check'],
                                   cwd=isolated.parent, capture_output=True, text=True,
                                   encoding='utf-8', timeout=30)
            self.assertEqual(check.returncode, 0, check.stderr)
            self.assertEqual(json.loads(check.stdout)['available_images'], {'train': 1, 'val': 1})
            evaluation = isolated / 'runs/fixture/evaluation/test'
            (evaluation / 'predictions').mkdir(parents=True)
            (evaluation / 'summary.json').write_text(json.dumps({
                'model_key': 'waste_bin_state', 'split': 'test',
                'diagnostics': {'match_iou': .5}, 'evaluation_config': {'imgsz': 640},
            }), encoding='utf-8')
            item = next(row for row in packed['records'] if row['split'] == 'test')
            prediction = {'source_image': item['image'], 'shape_hw': [20, 20],
                          'ground_truth': [{'class_id': 0, 'xyxy': [8, 8, 12, 12]}],
                          'predictions': [{'class_id': 1, 'xyxy': [8, 8, 12, 12], 'confidence': .9}]}
            (evaluation / 'predictions/predictions.jsonl').write_text(json.dumps(prediction)+'\n', encoding='utf-8')
            check = subprocess.run([sys.executable, '-I', '-B', str(isolated / 'scripts/train.py'),
                                    'bin-state-report', '--evaluation', 'runs/fixture/evaluation/test',
                                    '--dataset-manifest', 'dataset/workspace_sources.json'], cwd=isolated.parent,
                                   capture_output=True, text=True, encoding='utf-8', timeout=30)
            self.assertEqual(check.returncode, 0, check.stderr)
            report = json.loads((evaluation / 'bin_state_report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['totals']['instances'], 1)
            self.assertEqual(report['totals']['opposite_state'], 1)
            self.assertEqual(report['totals']['localization_miss'], 0)
            self.assertEqual(report['by_dimension']['color']['green']['opposite_state'], 1)


if __name__ == '__main__':
    unittest.main()
