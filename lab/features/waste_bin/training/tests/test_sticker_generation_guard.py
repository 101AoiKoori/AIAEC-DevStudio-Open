"""Retired public sticker entrypoints reject before any generation or file write."""
import ast
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS = ROOT / 'lab/features/waste_bin/training/scripts'


class StickerGuardTests(unittest.TestCase):
    def test_retired_main_rejects_without_reaching_worker(self):
        for name in ('generate_sticker_closeups_v4', 'generate_sticker_multibin_v4',
                     'build_blender_waste_stickers', 'plan_waste_sticker_batch_v4'):
            with self.subTest(entry=name):
                tree = ast.parse((SCRIPTS / (name + '.py')).read_text(encoding='utf8'))
                main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
                env = {}
                exec(compile(ast.Module(body=[main], type_ignores=[]), name, 'exec'), env)
                with patch.object(Path, 'mkdir', side_effect=AssertionError('unexpected write')), \
                     patch.object(Path, 'write_text', side_effect=AssertionError('unexpected write')), \
                     self.assertRaisesRegex(RuntimeError, 'Retired'):
                    env['main']()

    def test_retired_produce_api_rejects_without_importing_images(self):
        for name in ('generate_sticker_master_bin_v4', 'generate_sticker_real_parent_v4'):
            tree = ast.parse((SCRIPTS / (name + '.py')).read_text(encoding='utf8'))
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'produce')
            self.assertEqual(len(function.body), 1)
            self.assertIsInstance(function.body[0], ast.Raise)

    def test_canonical_real_source_is_declared_to_guard(self):
        tree = ast.parse((ROOT / 'datasets/scripts/waste_sticker/package_real_stickers.py').read_text(encoding='utf8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        guards = [node for node in ast.walk(main) if isinstance(node, ast.With) and any(
            isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == 'generation_guard' for item in node.items)]
        self.assertEqual(len(guards), 1)
        calls = [node for node in ast.walk(guards[0]) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == '_write']
        self.assertEqual(len(calls), 1)
        keywords = {kw.arg for kw in guards[0].items[0].context_expr.keywords}
        self.assertTrue({'production_plan', 'procedural_inputs', 'code_paths'} <= keywords)


if __name__ == '__main__':
    unittest.main()
