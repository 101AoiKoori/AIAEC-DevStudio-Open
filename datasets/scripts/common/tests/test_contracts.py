"""Small, file-free checks for the shared dataset script contracts."""
import ast
import unittest

import numpy as np

from datasets.scripts.common import bayesian, paths
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.perspective import warp_patch


class CommonContractsTest(unittest.TestCase):
    def test_task_code_does_not_import_other_tasks_or_legacy_generators(self):
        for task in paths.TASKS:
            for path in (paths.WORKSPACE / 'datasets/scripts' / task).glob('*.py'):
                if path.name.startswith('test_'):
                    continue
                tree = ast.parse(path.read_text(encoding='utf-8-sig'))
                modules = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        modules.append(node.module)
                    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                          and node.func.attr == 'import_module' and node.args
                          and isinstance(node.args[0], ast.Constant)
                          and isinstance(node.args[0].value, str)):
                        modules.append(node.args[0].value)
                for module in modules:
                    with self.subTest(path=path.relative_to(paths.WORKSPACE), module=module):
                        self.assertFalse(module.startswith('lab.features.'), 'task implementation must own its dependencies')
                        if module.startswith('datasets.scripts.'):
                            self.assertIn(module.split('.')[2], (task, 'common'))

    def test_task_configs_are_separate_and_resolve_declared_records(self):
        for task in paths.TASKS:
            design, source = load_task_config(task)
            self.assertEqual(design["sample_output"], f"datasets/samples/{task}")
            self.assertEqual(design["dataset_output"], f"datasets/exports/{task}")
            self.assertTrue(design["allocation"]["required"])
            self.assertAlmostEqual(sum(design["source_mix_initial"].values()), 1.0)
            for value in source["roots"] + source["records"] + list(source["databases"].values()):
                self.assertTrue((paths.WORKSPACE / value).exists(), value)

    def test_unknown_task_and_split_are_rejected(self):
        with self.assertRaises(ValueError):
            paths.sample_dir("../person")
        with self.assertRaises(ValueError):
            paths.split_dir("person", "../train")

    def test_glare_is_a_planned_state_only_for_meter(self):
        for task in paths.TASKS:
            with self.subTest(task=task):
                design, _ = load_task_config(task)
                self.assertEqual('反光' in design['states'], task == 'meter')

    def test_warp_returns_visible_mask_and_clipped_box(self):
        patch = np.full((4, 4, 3), 200, np.uint8)
        _, mask, box, _ = warp_patch(patch, [[-2, 1], [2, 1], [2, 4], [-2, 4]], (6, 6))
        self.assertEqual(box, [0, 1, 4, 5])
        self.assertTrue(np.all(mask[1:5, :3] > 0))
        self.assertFalse(np.any(mask[:, 4:]))

    def test_warp_rejects_invalid_or_invisible_quads(self):
        patch = np.ones((4, 4, 3), np.uint8)
        with self.assertRaises(ValueError):
            warp_patch(patch, [[0, 0], [1, 1], [2, 2], [3, 3]], (6, 6))
        with self.assertRaises(ValueError):
            warp_patch(patch, [[10, 10], [13, 10], [13, 13], [10, 13]], (6, 6))

    def test_bayesian_interface_is_existing_allocator(self):
        from lab.training_common import bayesian_allocation
        self.assertIs(bayesian.allocate, bayesian_allocation.allocate)

    def test_guard_requires_task_configs_and_stable_output(self):
        code = ['datasets/scripts/sign/compose_sign_v4_measured.py']
        configs = ['datasets/scripts/sign/config/dataset.yaml',
                   'datasets/scripts/sign/config/source.yaml']
        self.assertEqual(paths.validate_guarded_outputs(code, ['datasets/samples/sign'], configs), 'sign')
        self.assertEqual(paths.validate_guarded_outputs(code, ['datasets/samples/sign/val'], configs), 'sign')
        with self.assertRaises(ValueError):
            paths.validate_guarded_outputs(code, ['datasets/samples/sign/arbitrary'], configs)
        with self.assertRaises(ValueError):
            paths.validate_guarded_outputs(['datasets/scripts/person/generator.py'],
                ['datasets/samples/sign/val'], ['datasets/scripts/person/config/dataset.yaml',
                                              'datasets/scripts/person/config/source.yaml'])
        self.assertEqual(paths.validate_guarded_outputs(code, ['datasets/exports/sign/train'], configs), 'sign')
        with self.assertRaises(ValueError):
            paths.validate_guarded_outputs(code, ['runs/sign/batch01'], configs)
        with self.assertRaises(ValueError):
            paths.validate_guarded_outputs(code, ['datasets/samples/sign'], configs[:1])
        with self.assertRaises(ValueError):
            paths.validate_guarded_outputs(code + ['datasets/scripts/person/generator.py'],
                                           ['datasets/samples/sign'], configs)

    def test_sticker_orbit_review_has_its_own_sample_directory(self):
        code = ['datasets/scripts/waste_sticker/render_sticker_arena_v4.py']
        configs = ['datasets/scripts/waste_sticker/config/dataset.yaml',
                   'datasets/scripts/waste_sticker/config/source.yaml']
        output = paths.sticker_orbit_sample_dir(5).relative_to(paths.WORKSPACE).as_posix()
        self.assertEqual(paths.validate_guarded_outputs(code, [output], configs), 'waste_sticker')
        with self.assertRaises(ValueError):
            paths.validate_guarded_outputs(code, ['datasets/samples/waste_sticker/arbitrary'], configs)
        with self.assertRaises(ValueError):
            paths.validate_guarded_outputs(['datasets/scripts/waste_sticker/generate_stickers.py'],
                                           [output], configs)


if __name__ == "__main__":
    unittest.main()
