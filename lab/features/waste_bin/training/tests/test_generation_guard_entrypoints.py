"""Exercise the task entry points' guard blocks without running Blender or producing data."""
import ast
import json
from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
TEMP_ROOT = SCRIPTS.parents[4] / 'tmp'


def entry_path(name):
    if name in {'generate_waste_v4_grouped.py', 'composite_bin_real_scene.py', 'render_blender_waste_bin.py', 'export_bin_candidates.py', 'render_arena_waste_v4.py'}:
        return SCRIPTS.parents[4] / 'datasets/scripts/waste_bin_state' / name
    return SCRIPTS / name


ENTRY_POINTS = (
    'render_arena_waste_v4.py',
    'render_bin_reference_views.py',
    'render_blender_waste_bin.py',
    'generate_waste_v4_grouped.py',
    'composite_bin_real_scene.py',
    'apply_photo_bin_markings.py',
    'build_blender_waste_bin.py',
    'export_bin_candidates.py',
    'isolate_waste_v4_near_duplicates.py',
    'prepare_bin_test_candidates.py',
    'prepare_v4_real_bin_development.py',
    'tune_waste_bin_pbr.py',
    'check_waste_v4_grounded.py',
    'check_waste_v4_render_overlap.py',
)


def entry_guard(path):
    """Extract the actual main guard block and check that generation is inside it."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    mains = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == 'main']
    if len(mains) != 1:
        raise AssertionError(f'{path.name}: expected one main')
    main = mains[0]
    guards = [node for node in ast.walk(main) if isinstance(node, ast.With)
              and any(isinstance(item.context_expr, ast.Call)
                      and isinstance(item.context_expr.func, ast.Name)
                      and item.context_expr.func.id == 'generation_guard' for item in node.items)]
    if len(guards) != 1:
        raise AssertionError(f'{path.name}: expected one generation_guard in main')
    guarded = guards[0]
    if guarded not in main.body:
        raise AssertionError(f'{path.name}: guard must be an unconditional main statement')
    preceding = main.body[:main.body.index(guarded)]
    for node in preceding:
        for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
            func = call.func
            if isinstance(func, ast.Name) and func.id == '_generate':
                raise AssertionError(f'{path.name}: possible write or body before guard')
            if isinstance(func, ast.Name) and func.id == 'open':
                mode = call.args[1] if len(call.args) > 1 else next(
                    (arg.value for arg in call.keywords if arg.arg == 'mode'), None)
                if mode is not None and (not isinstance(mode, ast.Constant) or
                                         not isinstance(mode.value, str) or
                                         any(flag in mode.value for flag in 'wax+')):
                    raise AssertionError(f'{path.name}: possible file write before guard')
            if isinstance(func, ast.Attribute) and func.attr in {
                    'mkdir', 'write_text', 'write_bytes', 'write', 'render',
                    'save_as_mainfile', 'copy', 'copyfile', 'copytree'}:
                raise AssertionError(f'{path.name}: possible write before guard')
    body_calls = [node for node in ast.walk(guarded) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == '_generate']
    all_calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == '_generate']
    if len(body_calls) != 1 or len(all_calls) != 1:
        raise AssertionError(f'{path.name}: generation body must be called once inside guard')
    call = next(item.context_expr for item in guarded.items
                if isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Name)
                and item.context_expr.func.id == 'generation_guard')
    if not {'code_paths', 'procedural_inputs'} <= {keyword.arg for keyword in call.keywords}:
        raise AssertionError(f'{path.name}: missing explicit code/procedural inputs')
    return guarded


def run_guard_block(guarded, fake_guard, generate, invalid_plan=False):
    """Run the actual guard body with fake entry inputs and a temporary output writer."""
    block = ast.parse(ast.unparse(guarded)).body[0]
    for item in block.items:
        if (isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name)
                and item.context_expr.func.id == 'generation_guard'):
            item.context_expr = ast.Call(func=ast.Name(id='generation_guard', ctx=ast.Load()),
                                         args=[], keywords=[])
    function = ast.FunctionDef(name='invoke', args=ast.arguments(posonlyargs=[], args=[],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=[block], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    names = {node.id for node in ast.walk(block) if isinstance(node, ast.Name)}
    env = {name: MagicMock(name=name) for name in names}
    env.update(len=len, ValueError=ValueError, RuntimeError=RuntimeError)
    # A guarded wrapper may inspect its generated manifest after the mocked
    # body. The integration status itself is exercised by the real generator.
    class CompletedStatus:
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

    if 'json' in env:
        env['json'].loads.return_value = {'status': CompletedStatus()}
    if 'args' in env:
        env['args'].count = 0
    if 'planned' in env:
        env['planned'] = {}
    if invalid_plan:
        def reject(*args, **kwargs):
            raise ValueError('unsupported planned state')
        env['arena_planned_jobs'] = reject
        env['standalone_planned_jobs'] = reject
    env.update(generation_guard=fake_guard, _generate=generate)
    exec(compile(module, '<guard-block>', 'exec'), env)
    return env['invoke']()


class GuardEntryPointTests(unittest.TestCase):
    def test_standalone_packed_images_use_scene_without_external_file(self):
        tree = ast.parse(entry_path('render_blender_waste_bin.py').read_text(encoding='utf8'))
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_guard_inputs')
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp)
            scene = root / 'source.blend'
            scene.write_bytes(b'fixture')
            packed = SimpleNamespace(source='FILE', packed_file=object(), filepath='missing.png', name='packed')
            images = [packed]
            reloads = []
            def reopen(**kwargs):
                reloads.append(kwargs)
                return {'FINISHED'}
            bpy = SimpleNamespace(app=SimpleNamespace(background=True),
                                  data=SimpleNamespace(is_dirty=True, filepath=str(scene), images=images),
                                  ops=SimpleNamespace(wm=SimpleNamespace(open_mainfile=reopen)),
                                  path=SimpleNamespace(abspath=lambda value: str(root / value)))
            env = {'bpy': bpy, 'Path': Path, 'ROOT': root, 'json': json}
            exec(compile(ast.Module(body=[helper], type_ignores=[]), 'standalone_guard', 'exec'), env)
            media, procedural = env['_guard_inputs']()
            self.assertEqual(media, [])
            self.assertIn(scene, procedural)
            self.assertEqual(reloads, [{'filepath': str(scene), 'load_ui': False}])
            images.append(SimpleNamespace(source='FILE', packed_file=None, filepath='missing.png', name='external'))
            with self.assertRaisesRegex(ValueError, 'scene image source missing'):
                env['_guard_inputs']()
            bpy.app.background = False
            with self.assertRaisesRegex(ValueError, 'isolated background'):
                env['_guard_inputs']()
            self.assertEqual(len(reloads), 2)
            bpy.app.background = True
            bpy.ops.wm.open_mainfile = lambda **kwargs: {'CANCELLED'}
            with self.assertRaisesRegex(RuntimeError, 'could not reload'):
                env['_guard_inputs']()

    def test_planned_bin_renderers_require_plan_before_generation(self):
        for name in ('render_arena_waste_v4.py', 'render_blender_waste_bin.py'):
            with self.subTest(entry=name):
                tree = ast.parse(entry_path(name).read_text(encoding='utf8'))
                main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
                options = [call for call in ast.walk(main) if isinstance(call, ast.Call)
                           and isinstance(call.func, ast.Attribute) and call.func.attr == 'add_argument'
                           and call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value == '--plan']
                self.assertEqual(len(options), 1)
                self.assertTrue(any(kw.arg == 'required' and isinstance(kw.value, ast.Constant)
                                    and kw.value.value is True for kw in options[0].keywords))
                guarded = entry_guard(entry_path(name))
                guard_call = guarded.items[0].context_expr
                self.assertIn('production_plan', {kw.arg for kw in guard_call.keywords})
                calls = []
                @contextmanager
                def planned():
                    yield SimpleNamespace(reused=False, source_refs={}, plan={})
                with self.assertRaisesRegex(ValueError, 'unsupported planned state'):
                    run_guard_block(guarded, planned, lambda *args: calls.append(args), invalid_plan=True)
                self.assertEqual(calls, [])

    def test_arena_declares_known_sources_without_extra_manifest_field(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root = Path(temp)
            source = root / 'bin.blend'
            (root / 'manifest.json').write_text(json.dumps({'source_scene': 'upstream.blend',
                                                            'photo': 'photo.jpg'}), encoding='utf8')
            tree = ast.parse(entry_path('render_arena_waste_v4.py').read_text(encoding='utf8'))
            helper = next(node for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == '_guard_inputs')
            external = root / 'external.png'
            external.write_bytes(b'image')
            images = [SimpleNamespace(source='FILE', packed_file=object(), filepath='', name='packed',
                                      as_pointer=lambda: 1)]
            asset_image = SimpleNamespace(source='FILE', packed_file=None, filepath='//external.png',
                                          name='external', library=None, as_pointer=lambda: 2)
            @contextmanager
            def load(path, link):
                self.assertEqual(path, str(source))
                self.assertFalse(link)
                src = SimpleNamespace(objects=['Measured height scale', 'other'])
                dst = SimpleNamespace(objects=[])
                yield src, dst
                images.append(asset_image)
                self.assertEqual(dst.objects, ['Measured height scale'])

            def abspath(path, *, start=None, library=None):
                return str(Path(start) / path[2:]) if path.startswith('//') else path

            env = {'bpy': SimpleNamespace(app=SimpleNamespace(background=True),
                                          data=SimpleNamespace(is_dirty=False, filepath='scene.blend', images=images,
                                                               libraries=SimpleNamespace(load=load)),
                                          path=SimpleNamespace(abspath=abspath)),
                   'Path': Path, 'ROOT': root, 'SOURCE': source, 'json': json, 'BIN_NAMES': ('blue',),
                   'source_path': lambda *args: root/'upstream.blend',
                   'load_task_config': lambda *args: ({}, {'asset_parent_manifest_path': 'upstream.blend'})}
            exec(compile(ast.Module(body=[helper], type_ignores=[]), 'arena_guard', 'exec'), env)
            media, procedural, objects = env['_guard_inputs']()
            self.assertEqual(set(media), {root / 'photo.jpg', external})
            self.assertEqual(objects, ['Measured height scale'])
            self.assertIn(source, procedural)
            self.assertIn(root / 'upstream.blend', procedural)
            from lab.training_common.production_guard import _inventory
            before = _inventory(root, ['external.png'], True)
            external.write_bytes(b'changed image')
            self.assertNotEqual(before, _inventory(root, ['external.png'], True))
            external.unlink()
            images.pop()
            with self.assertRaisesRegex(ValueError, 'scene image source missing'):
                env['_guard_inputs']()

    def test_dirty_blender_scene_is_rejected_before_guard(self):
        for name in ('render_arena_waste_v4.py', 'render_bin_reference_views.py'):
            with self.subTest(entry=name):
                tree = ast.parse(entry_path(name).read_text(encoding='utf-8'))
                helper = next(node for node in tree.body
                              if isinstance(node, ast.FunctionDef) and node.name == '_guard_inputs')
                env = {'bpy': SimpleNamespace(app=SimpleNamespace(background=True),
                                              data=SimpleNamespace(is_dirty=True))}
                exec(compile(ast.Module(body=[helper], type_ignores=[]), name, 'exec'), env)
                with self.assertRaisesRegex(ValueError, 'unsaved changes'):
                    env['_guard_inputs']()

    def test_reuse_skips_generation_and_preserves_output(self):
        for name in ENTRY_POINTS:
            with self.subTest(entry=name), tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
                guarded = entry_guard(entry_path(name))
                output = Path(temp) / 'output.txt'
                calls = []

                def generate(*args, **kwargs):
                    calls.append(1)
                    output.write_text('first', encoding='utf-8')

                @contextmanager
                def first():
                    yield type('Run', (), {'reused': False, 'source_refs': {}, 'plan': {}})()

                @contextmanager
                def reused():
                    yield type('Run', (), {'reused': True, 'source_refs': {}, 'plan': {}})()

                run_guard_block(guarded, first, generate)
                before = output.stat().st_mtime_ns
                run_guard_block(guarded, reused, generate)
                self.assertEqual(calls, [1])
                self.assertEqual(output.read_text(encoding='utf-8'), 'first')
                self.assertEqual(output.stat().st_mtime_ns, before)

    def test_guard_error_prevents_generation_and_output(self):
        for name in ENTRY_POINTS:
            with self.subTest(entry=name), tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
                guarded = entry_guard(entry_path(name))
                output = Path(temp) / 'output.txt'

                def generate(*args, **kwargs):
                    output.write_text('unexpected', encoding='utf-8')

                @contextmanager
                def failed():
                    raise ValueError('database unavailable')
                    yield  # pragma: no cover

                with self.assertRaisesRegex(ValueError, 'database unavailable'):
                    run_guard_block(guarded, failed, generate)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
