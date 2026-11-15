import importlib.util
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sync_layout_docs
import workspace_layout


def contract():
    return {
        'version': 1,
        'generated_docs': {
            'readme': 'README.md',
            'dataset_design': 'docs/DATASET_DESIGN.md',
        },
        'root_entries': [
            {'path': 'docs', 'kind': 'dir', 'purpose': '文档', 'required': True},
            {'path': 'runs', 'kind': 'dir', 'purpose': '本机运行记录', 'required': True, 'local': True},
            {'path': 'datasets/assets/blender', 'kind': 'dir', 'purpose': '本地 Blender 资产',
             'required': True, 'local': True},
        ],
        'datasets': {
            'materialized_if_any': ['datasets/raw', 'datasets/annotated'],
            'shared': [
                {'path': 'datasets/annotated/common/review', 'purpose': '审查观测', 'required': True},
            ],
            'tasks': {
                'demo': {
                    'scripts': 'datasets/scripts/demo',
                    'config': {'required': ['dataset.yaml'], 'conditional': []},
                    'annotation_roots': ['datasets/annotated/demo'],
                    'annotation_note': '',
                    'exports': 'datasets/exports/demo',
                    'exports_splits': ['train'],
                    'samples': 'datasets/samples/demo',
                },
            },
        },
        'features': {
            'perception_tasks': 'config/perception_tasks.json',
            'root': 'lab/features',
            'required_files': ['README.md'],
            'branches': ['training', 'inference'],
            'branch_entries': ['scripts', 'config', 'tests'],
            'branch_local_entries': {'training': ['runs'], 'inference': ['outputs']},
            'require_branches': True,
            'extra_dirs': {},
            'feature_aliases': {},
            'fixed_roots': ['lab/navigation'],
        },
        'packages': {
            'root': 'ws/src',
            'name_pattern': '^aec_[a-z]+(?:_[a-z]+)*$',
            'required_files': ['package.xml'],
            'python_package': 'src/{name}/__init__.py',
        },
    }


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.contract_path = self.root / 'config/workspace_layout.yaml'
        self.contract_path.parent.mkdir(parents=True)
        self.contract_path.write_text(
            yaml.safe_dump(contract(), allow_unicode=True, sort_keys=False), encoding='utf-8')
        self.data = workspace_layout.load_contract(self.contract_path)
        for relative in (
                'docs', 'runs', 'datasets/assets/blender', 'datasets/annotated/common/review',
                'datasets/annotated/demo',
                'datasets/scripts/demo/config', 'datasets/exports/demo/train',
                'datasets/samples/demo', 'lab/features/demo/training/scripts',
                'lab/features/demo/inference/scripts',
                'lab/navigation', 'ws/src/aec_demo/src/aec_demo'):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        (self.root / 'datasets/scripts/demo/config/dataset.yaml').write_text('task: demo\n', encoding='utf-8')
        (self.root / 'config/perception_tasks.json').write_text(
            json.dumps({'tasks': [{'id': 'demo'}]}), encoding='utf-8')
        (self.root / 'lab/features/demo/README.md').write_text('# demo\n', encoding='utf-8')
        (self.root / 'ws/src/aec_demo/package.xml').write_text('<package/>\n', encoding='utf-8')
        (self.root / 'ws/src/aec_demo/src/aec_demo/__init__.py').write_text('', encoding='utf-8')
        (self.root / 'README.md').write_text(
            '# demo\n\n<!-- WORKSPACE_LAYOUT:START -->\nold\n<!-- WORKSPACE_LAYOUT:END -->\n', encoding='utf-8')
        (self.root / 'docs/DATASET_DESIGN.md').write_text(
            '# demo\n\n<!-- DATASET_LAYOUT:START -->\nold\n<!-- DATASET_LAYOUT:END -->\n', encoding='utf-8')

    def tearDown(self):
        self.temporary.cleanup()

    def test_check_passes_on_complete_tree(self):
        self.assertEqual(workspace_layout.check(self.root, self.data), [])

    def test_check_skips_dataset_tree_when_not_materialized(self):
        shutil.rmtree(self.root / 'datasets/annotated')
        (self.root / 'runs').rmdir()
        items = workspace_layout.check(self.root, self.data)
        self.assertTrue(any(item['severity'] == 'info'
                            and item['path'] == 'datasets' for item in items))
        self.assertFalse(any(item['severity'] == 'error' for item in items))

    def test_missing_shared_directory_is_error(self):
        (self.root / 'datasets/annotated/common/review').rmdir()
        items = workspace_layout.check(self.root, self.data)
        self.assertTrue(any(item['severity'] == 'error'
                            and item['path'] == 'datasets/annotated/common/review' for item in items))

    def test_missing_task_config_is_error(self):
        (self.root / 'datasets/scripts/demo/config/dataset.yaml').unlink()
        items = workspace_layout.check(self.root, self.data)
        self.assertTrue(any(item['severity'] == 'error'
                            and item['path'].endswith('config/dataset.yaml') for item in items))

    def test_conditional_config_is_info_not_error(self):
        spec = self.data['datasets']['tasks']['demo']['config']
        spec['conditional'] = [{'path': 'production_plan.json', 'reason': '等待渲染'}]
        items = workspace_layout.check(self.root, self.data)
        self.assertTrue(any(item['severity'] == 'info'
                            and item['path'].endswith('production_plan.json') for item in items))
        self.assertFalse(any(item['severity'] == 'error' for item in items))

    def test_scaffold_creates_declared_directories(self):
        (self.root / 'datasets/samples/demo').rmdir()
        (self.root / 'datasets/annotated/demo').rmdir()
        created, missing = workspace_layout.scaffold(self.root, self.data, ['demo'])
        self.assertIn('datasets/samples/demo', created)
        self.assertIn('datasets/annotated/demo', created)
        self.assertEqual(missing, [])
        self.assertTrue((self.root / 'datasets/samples/demo').is_dir())

    def test_scaffold_reports_missing_config_files(self):
        (self.root / 'datasets/scripts/demo/config/dataset.yaml').unlink()
        _, missing = workspace_layout.scaffold(self.root, self.data, ['demo'])
        self.assertEqual(missing, ['datasets/scripts/demo/config/dataset.yaml'])

    def test_scaffold_shared_creates_local_dataset_roots(self):
        (self.root / 'datasets/assets/blender').rmdir()
        created, _ = workspace_layout.scaffold(self.root, self.data, ['demo'], include_shared=True)
        self.assertIn('datasets/assets/blender', created)
        self.assertTrue((self.root / 'datasets/assets/blender').is_dir())

    def test_scaffold_skips_shared_input_roots(self):
        self.data['datasets']['shared'][0]['create'] = False
        (self.root / 'datasets/annotated/common/review').rmdir()
        created, _ = workspace_layout.scaffold(self.root, self.data, ['demo'], include_shared=True)
        self.assertNotIn('datasets/annotated/common/review', created)
        self.assertFalse((self.root / 'datasets/annotated/common/review').exists())

    def test_scaffold_skips_unowned_annotation_roots(self):
        self.data['datasets']['tasks']['demo']['annotation_roots_owned'] = False
        (self.root / 'datasets/annotated/demo').rmdir()
        created, _ = workspace_layout.scaffold(self.root, self.data, ['demo'])
        self.assertNotIn('datasets/annotated/demo', created)
        self.assertFalse((self.root / 'datasets/annotated/demo').exists())

    def test_missing_feature_branch_is_error(self):
        shutil.rmtree(self.root / 'lab/features/demo/inference')
        items = workspace_layout.check(self.root, self.data)
        self.assertTrue(any(item['severity'] == 'error'
                            and item['path'].endswith('demo/inference') for item in items))

    def test_unregistered_feature_directory_warns(self):
        (self.root / 'lab/features/other').mkdir()
        items = workspace_layout.check(self.root, self.data)
        self.assertTrue(any(item['severity'] == 'warning'
                            and item['path'] == 'lab/features/other' for item in items))

    def test_suggest_feature_alias(self):
        self.data['features']['feature_aliases'] = {'alias': 'demo'}
        self.assertEqual(
            workspace_layout.suggest(self.data, 'feature-root', task='alias', root=self.root),
            ['lab/features/demo'])

    def test_suggest_dataset_export_returns_split_paths(self):
        self.assertEqual(
            workspace_layout.suggest(self.data, 'dataset-export', task='demo'),
            ['datasets/exports/demo/train'])

    def test_suggest_returns_declared_annotation_roots(self):
        self.assertEqual(workspace_layout.suggest(self.data, 'dataset-annotation', task='demo'),
                         ['datasets/annotated/demo'])

    def test_suggest_shared_roles_do_not_need_task(self):
        self.assertEqual(workspace_layout.suggest(self.data, 'dataset-shared-script'),
                         ['datasets/scripts/common'])
        self.assertEqual(workspace_layout.suggest(self.data, 'dataset-shared-annotation'),
                         ['datasets/annotated/common/review',
                          'datasets/annotated/common/perspective'])

    def test_suggest_rejects_unknown_task(self):
        with self.assertRaises(ValueError):
            workspace_layout.suggest(self.data, 'dataset-script', task='missing')

    def test_package_name_rule_reports_error(self):
        (self.root / 'ws/src/aec_Bad').mkdir()
        items = workspace_layout.check(self.root, self.data)
        self.assertTrue(any(item['severity'] == 'error'
                            and item['path'] == 'ws/src/aec_Bad' for item in items))

    def test_unknown_feature_entry_is_warning(self):
        (self.root / 'lab/features/demo/training/weird').mkdir()
        items = workspace_layout.check(self.root, self.data)
        self.assertTrue(any(item['severity'] == 'warning'
                            and item['path'].endswith('training/weird') for item in items))

    def test_render_replaces_marked_sections(self):
        outputs = sync_layout_docs.render(self.root, self.contract_path)
        readme = outputs[self.root / 'README.md']
        design = outputs[self.root / 'docs/DATASET_DESIGN.md']
        self.assertIn('`docs/`', readme)
        self.assertIn('| demo |', design)
        self.assertIn('`datasets/annotated/demo/`', design)
        self.assertNotIn('\nold\n', readme)
        self.assertNotIn('\nold\n', design)

    def test_render_requires_one_marker_pair(self):
        path = self.root / 'README.md'
        path.write_text('# demo\n\n<!-- WORKSPACE_LAYOUT:START -->\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            sync_layout_docs.render(self.root, self.contract_path)

    def test_render_includes_shared_roots_and_optional_config(self):
        self.data['datasets']['tasks']['demo']['optional_config'] = ['review_plan.json']
        self.data['datasets']['shared'][0]['purpose'] = '共享审查根'
        self.contract_path.write_text(
            yaml.safe_dump(self.data, allow_unicode=True, sort_keys=False), encoding='utf-8')
        outputs = sync_layout_docs.render(self.root, self.contract_path)
        design = outputs[self.root / 'docs/DATASET_DESIGN.md']
        self.assertIn('`datasets/annotated/common/review/`', design)
        self.assertIn('共享审查根', design)
        self.assertIn('可选配置（存在时使用）', design)
        self.assertIn('`review_plan.json`', design)

    def test_render_rejects_doc_path_outside_root(self):
        data = contract()
        data['generated_docs']['readme'] = '../outside.md'
        path = self.root / 'config/outside.yaml'
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
        with self.assertRaises(ValueError):
            sync_layout_docs.render(self.root, path)

    def test_dataset_task_names_match_paths_module(self):
        root = Path(__file__).resolve().parents[4]
        data = workspace_layout.load_contract(root / 'config/workspace_layout.yaml')
        spec = importlib.util.spec_from_file_location(
            'workspace_dataset_paths', root / 'datasets/scripts/common/paths.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(set(module.TASKS), set(data['datasets']['tasks']))
        splits = set()
        for task in data['datasets']['tasks'].values():
            splits.update(task['exports_splits'])
        self.assertEqual(set(module.SPLITS), splits)

    def test_review_roots_match_contract(self):
        root = Path(__file__).resolve().parents[4]
        data = workspace_layout.load_contract(root / 'config/workspace_layout.yaml')
        shared = {entry['path'] for entry in data['datasets']['shared']}
        text = (root / 'datasets/scripts/common/review/cli.py').read_text(encoding='utf-8')
        for name in ('DEFAULT_ROOT', 'PERSPECTIVE_ROOT'):
            match = re.search(r'^{} = "([^"]+)"'.format(name), text, re.M)
            self.assertIsNotNone(match, name)
            self.assertIn(match.group(1), shared)


    def test_validate_contract_rejects_duplicate_shared_path(self):
        data = contract()
        data['datasets']['shared'].append(dict(data['datasets']['shared'][0]))
        with self.assertRaises(ValueError):
            workspace_layout.validate_contract(data)

    def test_validate_contract_rejects_missing_task_key(self):
        data = contract()
        del data['datasets']['tasks']['demo']['samples']
        with self.assertRaises(ValueError):
            workspace_layout.validate_contract(data)

    def test_validate_contract_rejects_absolute_path(self):
        data = contract()
        data['datasets']['tasks']['demo']['scripts'] = '/absolute/path'
        with self.assertRaises(ValueError):
            workspace_layout.validate_contract(data)

    def test_validate_contract_rejects_bad_package_pattern(self):
        data = contract()
        data['packages']['name_pattern'] = '('
        with self.assertRaises(ValueError):
            workspace_layout.validate_contract(data)

    def test_validate_contract_accepts_yaml_loaded_contract(self):
        self.assertEqual(
            workspace_layout.validate_contract(workspace_layout.load_contract(
                self.contract_path))['version'], 1)


if __name__ == '__main__':
    unittest.main()
