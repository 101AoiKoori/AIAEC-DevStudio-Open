import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('package_builder', ROOT / 'tools/build_training_packages.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class PackageBuilderTest(unittest.TestCase):
    def test_same_length_different_image_is_not_the_same_source(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp)/'source.bmp', Path(temp)/'mirror.bmp'
            left.write_bytes(b'first image')
            right.write_bytes(b'other image')
            self.assertEqual(left.stat().st_size, right.stat().st_size)
            self.assertFalse(builder.same_file_bytes(left, right))
            right.write_bytes(left.read_bytes())
            self.assertTrue(builder.same_file_bytes(left, right))

    def test_refresh_preserves_dataset_weights_and_existing_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            template = root / 'lab/training_common'
            (template / 'engine').mkdir(parents=True)
            for name in ('cli.py', 'runner.py', '__init__.py'):
                (template / 'engine' / name).write_text('# engine\n')
            for name in ('train.py', 'launch_v3.py', 'start_v3.ps1', 'tensorboard_v3.py'):
                (template / name).write_text('# source\n')
            config = root / 'lab/features/person/training/config/person_v3.yaml'
            config.parent.mkdir(parents=True)
            config.write_text(yaml.safe_dump({'schema_version': 3, 'model_key': 'person',
                'epochs': 150, 'data': 'original/data.yaml', 'model': 'models/yolov8n.pt',
                'output_root': 'runs/person', 'data_overrides': {'test': []}}))
            package = root / 'packages/person'
            (package / 'dataset').mkdir(parents=True)
            (package / 'dataset/data.yaml').write_text(yaml.safe_dump({'names': ['person']}))
            for name in ('dataset/one.jpg', 'weights/yolov8n.pt', 'runs/previous/result.json'):
                file = package / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b'preserve this')
            with mock.patch.object(builder, 'ROOT', root), mock.patch.object(builder, 'DEST', root/'packages'), \
                 mock.patch.object(builder, 'TEMPLATE', template), \
                 mock.patch.object(builder, 'verify_mirror_data', return_value=({'train': 1}, [])):
                builder.refresh_code('person')
            for name in ('dataset/one.jpg', 'weights/yolov8n.pt', 'runs/previous/result.json'):
                self.assertEqual((package/name).read_bytes(), b'preserve this')
            result = yaml.safe_load((package/'config/train.yaml').read_text())
            self.assertEqual(result['data'], 'dataset/data.yaml')
            self.assertEqual(result['output_root'], 'runs/person')
            self.assertTrue((package/'runs').is_dir())
            self.assertNotIn('data_overrides', result)
            self.assertTrue((package/'scripts/common/engine/runner.py').is_file())
            self.assertIn('engine.cli', (package/'scripts/train.py').read_text())
            self.assertIn('ROOT = Path(__file__).resolve().parents[1]', (package/'scripts/train.py').read_text())
            self.assertTrue((package/'scripts/view_tensorboard.py').is_file())
            self.assertFalse((package/'scripts/tensorboard.py').exists())
            self.assertNotIn('\r', (package/'README.md').read_text())

    def test_refresh_rejects_missing_v3_source_before_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root/'packages/person'
            (package/'dataset').mkdir(parents=True)
            (package/'dataset/data.yaml').write_text('names: [person]\n')
            (package/'train.py').write_text('existing code')
            config = root/'lab/features/person/training/config/person_v3.yaml'
            config.parent.mkdir(parents=True)
            config.write_text('schema_version: 3\n')
            with mock.patch.object(builder, 'ROOT', root), mock.patch.object(builder, 'DEST', root/'packages'), \
                 mock.patch.object(builder, 'TEMPLATE', root/'missing'):
                with self.assertRaises(FileNotFoundError):
                    builder.refresh_code('person')
            self.assertEqual((package/'train.py').read_text(), 'existing code')

    def test_source_label_maps_to_canonical_competition_name(self):
        result = builder.waste_class_mapping({8: '苹果'}, {'苹果': {'name': '果核', 'category': 'food'}})
        self.assertEqual(result['names'][8], '苹果')
        self.assertEqual(result['competition_names'][8], '果核')
        self.assertEqual(result['categories'][8], 'food')

    def test_images_and_labels_are_copied_into_each_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            template = root / 'templates'
            template.mkdir()
            for filename in ('train.py', 'tensorboard_log.py', 'start_training.ps1', 'requirements.txt'):
                (template / filename).write_text('template', encoding='utf-8')
            weights = root / 'models/weights/yolo_v8'
            weights.mkdir(parents=True)
            (weights / 'yolov8n.pt').write_bytes(b'model')
            (weights / 'LICENSE.txt').write_text('license', encoding='utf-8')
            source = root / 'input'
            data = {'names': {0: 'object'}}
            for split in ('train', 'val', 'test'):
                for kind in ('images', 'labels'):
                    (source / kind / split).mkdir(parents=True)
                (source / 'images' / split / 'one.jpg').write_bytes(b'image')
                (source / 'labels' / split / 'one.txt').write_text('0 0.5 0.5 0.1 0.1\n')
                data[split] = 'images/' + split
            (source / 'data.yaml').write_text(yaml.safe_dump(data), encoding='utf-8')
            config = root / 'lab/features/test/training/config/test_v3.yaml'
            config.parent.mkdir(parents=True)
            config.write_text(yaml.safe_dump({'data': 'input/data.yaml', 'task': 'detect',
                'epochs': 10, 'model': 'models/weights/yolo_v8/yolov8n.pt'}))
            with mock.patch.object(builder, 'ROOT', root), mock.patch.object(builder, 'DEST', root / 'packages'), \
                 mock.patch.object(builder, 'TEMPLATE', template), \
                 mock.patch.object(builder, 'TASKS', {'test': ('input/data.yaml', 'detect', 10)}), \
                 mock.patch.object(builder, 'NOTES', {'test': '来源待核实'}), \
                 mock.patch.object(builder, 'configured_sources', return_value=[
                     (split, source/'images'/split, source/'labels'/split)
                     for split in ('train', 'val', 'test')]):
                builder.build('test')
                package = root / 'packages/test'
                for split in ('train', 'val', 'test'):
                    self.assertTrue((package / 'dataset/images' / split / 'one.jpg').is_file())
                    self.assertTrue((package / 'dataset/labels' / split / 'one.txt').is_file())
                self.assertTrue((package / 'weights/yolov8n.pt').is_file())
                with self.assertRaises(FileExistsError):
                    builder.build('test')

    def test_combines_two_synthetic_sources_and_test_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            primary, extra, evaluation = root / 'primary', root / 'extra', root / 'evaluation'
            for directory, split, stem in ((primary, 'train', 'a'), (primary, 'val', 'b'),
                                            (extra, 'train', 'c'), (evaluation, 'test', 'd')):
                for kind in ('images', 'labels'):
                    (directory / kind / split).mkdir(parents=True)
                (directory / 'images' / split / (stem + '.jpg')).write_bytes(b'image')
                (directory / 'labels' / split / (stem + '.txt')).write_text('0 0.5 0.5 0.1 0.1\n')
            (primary / 'data.yaml').write_text(yaml.safe_dump({'train': 'images/train', 'val': 'images/val'}))
            package = root / 'package'
            for split in ('train', 'val', 'test'):
                for kind in ('images', 'labels'):
                    (package / 'dataset' / kind / split).mkdir(parents=True)
            with mock.patch.object(builder, 'ROOT', root):
                result = builder.copy_candidates(package, primary / 'data.yaml',
                    {'test': 'evaluation/images/test'}, {'train': 'extra/images/train'})
            self.assertEqual(sum(x['images'] for x in result['train']), 2)
            self.assertTrue((package / 'dataset/images/train/c.jpg').is_file())
            self.assertTrue((package / 'dataset/labels/test/d.txt').is_file())


if __name__ == '__main__':
    unittest.main()
