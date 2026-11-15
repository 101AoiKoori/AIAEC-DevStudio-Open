"""Static production-entry inventory; does not import or execute generators.

Run with ``python lab/scripts/audit/test_production_entry_coverage.py``.  Exit 1 means at
least one discovered entry has an unproved guard path or parse failure.  This
is a lexical call-graph audit, not a proof of runtime input closure, database
freshness, dynamic dispatch, branch feasibility, or complete side effects.
``--self-test`` exercises the analyzer entirely in memory.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SHARED = (
    'lab/training_common/bayesian_allocation.py',
    'lab/training_common/camera_tone.py',
    'lab/scripts/media/extract_capture_frames.py',
    'lab/scripts/data/export_site_adaptation.py',
    'lab/scripts/data/v4_source_gate.py',
)
# These perform model training or a verifier with no production dataset writes.
# Other audit/check scripts remain in scope because they may write state/media.
PURE_EVALUATION = {
    'lab/features/meter/training/scripts/audit_meter_ultralytics.py':
        'model-library verifier; no dataset production',
    'lab/features/waste_bin/training/scripts/audit_bin_renders.py':
        'read-only render and label audit; stdout only',
    'datasets/scripts/waste_bin_state/audit_bin_renders.py':
        'read-only render and label audit; stdout only',
    'lab/features/waste_bin/training/scripts/audit_stickers.py':
        'read-only zip audit; stdout only',
    'lab/features/waste_bin/training/scripts/check_generated_stickers.py':
        'read-only generated-data check; stdout only',
}
HELPER_ONLY = {
    'e_bike_queue_inputs.py', 'fire_blender_labels.py', 'meter_blender_action.py',
    'meter_production_guard.py', 'power_meter_room.py', 'plate_guard_inputs.py',
    'sign_production_guard.py', 'bin_render_contract.py',
    'inspect_arena_plate.py',  # read-only scene inspection and stdout
}
PUBLIC_PRODUCTION = {'generate', 'export', 'prepare', 'produce', 'render',
                     'build', 'compose', 'synthesize', 'extract', 'make_preview', 'build_queue',
                     'main', 'package'}
WRITE_METHODS = {'mkdir', 'write_text', 'write_bytes', 'tofile', 'imwrite',
                 'copy2', 'copyfile', 'copytree', 'move',
                 'rename', 'unlink', 'rmtree', 'save', 'save_as_mainfile',
                 'render', 'dump', 'dump_all', 'savez', 'savez_compressed'}
WRITE_HELPERS = {'write_image', 'write_run', 'write_json', 'save_image',
                 'write_manifest', 'write_label', 'render_id_pass'}


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return (base + '.' if base else '') + node.attr
    return ''


def is_main_test(node):
    return isinstance(node, ast.If) and any(isinstance(x, ast.Constant) and x.value == '__main__'
                                            for x in ast.walk(node.test))


def has_guard_import(tree):
    return any(isinstance(x, ast.ImportFrom) and x.module == 'lab.training_common.production_guard'
               and any(a.name == 'generation_guard' for a in x.names) for x in ast.walk(tree))


def adapter_valid(path):
    try:
        tree = ast.parse(path.read_text(encoding='utf8'))
    except (OSError, SyntaxError, UnicodeError):
        return False
    if not has_guard_import(tree):
        return False
    return any(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == 'guarded'
               and any(isinstance(y, ast.Call) and call_name(y.func) == 'generation_guard'
                       for y in ast.walk(x)) for x in tree.body)


def trusted_guards(tree, path, root):
    """Accept only known imports, including two actual task adapters."""
    guards = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == 'lab.training_common.production_guard':
                guards.update(a.asname or a.name for a in node.names if a.name == 'generation_guard')
            # The package builder puts the verified workspace training_common
            # directory on sys.path before this exact import.
            if (node.module == 'production_guard' and path.relative_to(root).as_posix() in
                    {'lab/scripts/data/v4_source_gate.py'}):
                guards.update(a.asname or a.name for a in node.names if a.name == 'generation_guard')
            if node.module in ('sign_production_guard', 'meter_production_guard',
                               'datasets.scripts.sign.sign_production_guard',
                               'datasets.scripts.meter.meter_production_guard'):
                adapter = path.parent / (node.module + '.py')
                if node.module.startswith('datasets.scripts.'):
                    adapter = root / (node.module.replace('.', '/') + '.py')
                if adapter_valid(adapter):
                    guards.update(a.asname or a.name for a in node.names if a.name == 'guarded')
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'generate_sticker_closeups_v4':
                    module = path.parent / (alias.name + '.py')
                    try:
                        imported = ast.parse(module.read_text(encoding='utf8'))
                    except (OSError, SyntaxError, UnicodeError):
                        continue
                    if has_guard_import(imported):
                        guards.add((alias.asname or alias.name) + '.generation_guard')
    # A local function can shadow an imported name. Do not trust that binding.
    local = {x.name for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    return {g for g in guards if g.split('.')[0] not in local}


def write_call(node):
    name = call_name(node.func)
    tail = name.rsplit('.', 1)[-1]
    if name in ('shutil.copy', 'os.replace', 'Path.replace'):
        return name
    if name == 'json.dump' and len(node.args) > 1 and call_name(node.args[1]) in ('sys.stdout', 'sys.stderr'):
        return ''
    if tail in WRITE_METHODS or tail in WRITE_HELPERS:
        return name
    if tail == 'open':
        position = 1 if name in ('open', 'io.open') else 0
        mode = node.args[position] if len(node.args) > position else next((x.value for x in node.keywords if x.arg == 'mode'), None)
        if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and any(x in mode.value for x in 'wax+'):
            return name
    return ''


class Analyzer:
    def __init__(self, tree, path, root):
        self.functions = {x.name: x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.guards = trusted_guards(tree, path, root)
        self.local_guards = set()
        for name, function in self.functions.items():
            if name != '_guarded':
                continue
            for node in ast.walk(function):
                if isinstance(node, (ast.With, ast.AsyncWith)) and any(
                        isinstance(x.context_expr, ast.Call) and call_name(x.context_expr.func) in self.guards
                        for x in node.items):
                    self.local_guards.add(name)
        self.findings = []
        self.guard_lines = set()
        self.visited = set()
        self.terminated = False
        self.dispatches = []

    def inspect_block(self, body, guarded=False, stack=()):
        """Stop at an unconditional exit, including an early retired CLI."""
        for statement in body:
            self.inspect(statement, guarded, stack)
            if isinstance(statement, (ast.Raise, ast.Return)):
                self.terminated = True
                break

    def inspect(self, node, guarded=False, stack=()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                self.inspect(item.context_expr, guarded, stack)
            active = any(isinstance(x.context_expr, ast.Call) and call_name(x.context_expr.func) in self.guards
                         for x in node.items)
            if active:
                self.guard_lines.add(node.lineno)
            self.inspect_block(node.body, guarded or active, stack)
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and not guarded:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                text = ast.unparse(target)
                if text.startswith(('bpy.data.', 'bpy.context.', 'scene.render.', 'scene.camera.',
                                    'scene.cycles.', 'scene.view_settings.')):
                    self.findings.append((node.lineno, 'scene mutation before guard: ' + text))
        if isinstance(node, ast.Call):
            name = call_name(node.func)
            if name in ('runpy.run_path', 'runpy.run_module', 'exec', 'eval',
                        'subprocess.run', 'subprocess.call', 'subprocess.Popen', 'os.system'):
                self.dispatches.append((node.lineno, name))
            if name in self.local_guards:
                self.guard_lines.add(node.lineno)
            effect = write_call(node)
            if effect and not guarded:
                self.findings.append((node.lineno, 'possible write before guard: ' + effect))
            if name in self.functions and name not in stack and name not in self.local_guards:
                marker = (name, guarded)
                if marker not in self.visited:
                    self.visited.add(marker)
                    self.inspect_block(self.functions[name].body, guarded, stack + (name,))
        for child in ast.iter_child_nodes(node):
            self.inspect(child, guarded, stack)


def entries_for(tree):
    functions = {x.name: x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}
    result = []
    if any(is_main_test(node) for node in tree.body):
        body = []
        for node in tree.body:
            if is_main_test(node):
                body.extend(node.body)
            elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body.append(node)
        result.append(('CLI', body))
    for name in sorted(PUBLIC_PRODUCTION & functions.keys()):
        result.append((name, functions[name].body))
    # Blender accepts --python for scripts without __main__; inspect module body.
    if not result and any(isinstance(x, ast.Import) and any(a.name == 'bpy' for a in x.names) for x in tree.body):
        result.append(('module', [x for x in tree.body if not isinstance(x, (ast.FunctionDef, ast.ClassDef))]))
    return result


def analyze_text(text, path, root=ROOT):
    tree = ast.parse(text, filename=str(path))
    entries = entries_for(tree)
    output = []
    for name, body in entries:
        analyzer = Analyzer(tree, path, root)
        analyzer.inspect_block(body)
        # Common pure helpers such as compose()/extract() are not production
        # entries unless they write or reach a guard; CLI paths remain in scope.
        if name in ('CLI', 'module') or analyzer.guard_lines or analyzer.findings:
            output.append({'entry': name, 'guard_lines': sorted(analyzer.guard_lines),
                           'unprotected': sorted(set(analyzer.findings)),
                           'dispatches': sorted(set(analyzer.dispatches)),
                           'disabled': _unconditional_exit(body, tree) and not analyzer.findings and not analyzer.guard_lines})
    return output


def _unconditional_exit(body, tree):
    """Recognize a hard exit, never a marker string or a conditional raise."""
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for node in body:
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            if not any(isinstance(arg, ast.Call) for arg in node.exc.args):
                return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = call_name(node.value.func)
            if name in functions and any(isinstance(s, ast.Raise) for s in functions[name].body):
                return True
        if isinstance(node, ast.Return):
            return False
    return False


def forwarding_target(tree, path, root):
    """Resolve the checked literal runpy shim; do not execute path expressions."""
    assignments = {n.targets[0].id: n.value for n in tree.body
                   if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    main = [n for n in tree.body if is_main_test(n)]
    if len(main) != 1 or len(main[0].body) != 1:
        return None
    statement = main[0].body[0]
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if call_name(call.func) != 'runpy.run_path' or len(call.args) != 1:
        return None
    if not any(k.arg == 'run_name' and isinstance(k.value, ast.Constant) and k.value.value == '__main__' for k in call.keywords):
        return None
    arg = call.args[0]
    if isinstance(arg, ast.Call) and call_name(arg.func) == 'str' and len(arg.args) == 1:
        arg = arg.args[0]
    if isinstance(arg, ast.Name):
        arg = assignments.get(arg.id)
    if not isinstance(arg, ast.BinOp) or not isinstance(arg.op, ast.Div) or not isinstance(arg.right, ast.Constant):
        return None
    left = arg.left
    if isinstance(left, ast.Name):
        left = assignments.get(left.id)
    expected = f'Path(__file__).resolve().parents[{len(path.relative_to(root).parts) - 1}]'
    if left is None or ast.unparse(left) != expected or not isinstance(arg.right.value, str):
        return None
    target = (root / arg.right.value).resolve()
    if not target.is_relative_to((root / 'datasets/scripts').resolve()) or not target.is_file():
        return None
    return target


def classify_entry(relative, entry, root=ROOT, seen=()):
    """Keep bypasses failing while separating disabled and checked forwarded CLIs."""
    if entry.get('disabled'):
        return 'DISABLED', 'unconditional exit before writes'
    if entry.get('dispatches') and not entry['guard_lines']:
        path = root / relative
        target = forwarding_target(ast.parse(path.read_text(encoding='utf8')), path, root)
        if target and relative not in seen and not entry['unprotected']:
            target_relative = target.relative_to(root).as_posix()
            checks = analyze_text(target.read_text(encoding='utf8'), target, root)
            cli = next((x for x in checks if x['entry'] == 'CLI'), None)
            if cli:
                status, _ = classify_entry(target_relative, cli, root, seen + (relative,))
                if status in ('GUARDED', 'FORWARDED', 'DISABLED', 'PLAN_ONLY', 'NO_STATIC_WRITES'):
                    return 'FORWARDED', f'{target_relative} ({status})'
        return 'UNPROVED', 'dynamic dispatch has no verified guarded target'
    if entry['unprotected']:
        review = reviewed_branch(relative, entry, root)
        if review:
            return 'REVIEWED_BRANCH', review
        # Planning tools write a JSON declaration, not dataset images. Still
        # report their metadata side effects and reject any media/scene write
        # or unresolved dispatch instead of excluding by filename alone.
        if (Path(relative).stem.startswith('plan_') and entry['entry'] in ('CLI', 'main')
                and not entry.get('dispatches') and not entry['guard_lines']
                and all(reason.startswith('possible write before guard: ') and
                        reason.split(': ', 1)[1].rsplit('.', 1)[-1] in ('open', 'dump', 'write_text', 'mkdir')
                        for _, reason in entry['unprotected'])):
            return 'PLAN_ONLY', 'JSON planning entry with metadata writes; no media production detected'
        return 'BYPASS', 'reachable writes outside a recognized guard'
    if entry['guard_lines']:
        return 'GUARDED', 'recognized guard on lexical write path'
    return 'NO_STATIC_WRITES', 'no write found in the local call graph; imports are not runtime proof'


def scaled_parking_delegation(tree, path, root):
    """Prove the one known cross-file CLI route rather than trust its name."""
    imported = any(isinstance(n, ast.ImportFrom) and n.module == 'check_v4_multibike_parking'
                   and any(x.name == 'run_check' and not x.asname for x in n.names) for n in tree.body)
    main = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main'), None)
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and call_name(n.func) == 'run_check'] if main else []
    if not imported or len(calls) != 1 or not any(k.arg == 'scale_check' and isinstance(k.value, ast.Constant)
                                                   and k.value.value is True for k in calls[0].keywords):
        return None
    target = path.parent / 'check_v4_multibike_parking.py'
    try:
        target_tree = ast.parse(target.read_text(encoding='utf8'))
    except (OSError, SyntaxError, UnicodeError):
        return None
    function = next((n for n in target_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_check'), None)
    if function is None:
        return None
    # The delegated _scale_check callback must occur lexically under the
    # same recognized context; an import/call alone is insufficient.
    analyzer = Analyzer(target_tree, target, root)
    for statement in function.body:
        analyzer.inspect(statement)
    guarded_callback = any(isinstance(n, (ast.With, ast.AsyncWith)) and any(
        isinstance(i.context_expr, ast.Call) and call_name(i.context_expr.func) in analyzer.guards for i in n.items)
        and any(isinstance(x, ast.Call) and call_name(x.func) == '_scale_check'
                for body in n.body for x in ast.walk(body)) for n in ast.walk(function))
    if not guarded_callback:
        return None
    return [{'entry': 'CLI', 'guard_lines': sorted(analyzer.guard_lines),
             'unprotected': sorted(set(analyzer.findings))}]


def inventory(root=ROOT):
    candidates = sorted((root / 'lab/features').glob('*/training/scripts/*.py'))
    candidates += sorted(path for path in (root / 'datasets/scripts').glob('*/*.py')
                         if path.parent.name != 'common')
    candidates += [root / x for x in SHARED]
    output, excluded, errors = [], [], []
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        if path.name.startswith(('train_', 'test_')) or relative in PURE_EVALUATION:
            excluded.append((relative, 'training or test entry' if path.name.startswith(('train_', 'test_')) else PURE_EVALUATION[relative]))
            continue
        if path.name in HELPER_ONLY:
            excluded.append((relative, 'helper or read-only scene inspection'))
            continue
        try:
            text = path.read_text(encoding='utf8')
            if relative == 'lab/features/e_bike/training/scripts/check_v4_scaled_parking.py':
                checks = scaled_parking_delegation(ast.parse(text, filename=str(path)), path, root)
                if checks is None:
                    errors.append((relative, 'cross-file guarded delegation not established'))
                    continue
            else:
                checks = analyze_text(text, path, root)
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append((relative, str(exc)))
            continue
        if checks:
            output.append((relative, checks))
        else:
            # Report helper modules explicitly. Module-level side effects are
            # checked even when there is no CLI or public production name.
            tree = ast.parse(text)
            analyzer = Analyzer(tree, path, root)
            analyzer.inspect_block([n for n in tree.body if not isinstance(
                n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))])
            for function in tree.body:
                if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and not function.name.startswith('_'):
                    analyzer.inspect_block(function.body, stack=(function.name,))
            if analyzer.findings or analyzer.dispatches:
                errors.append((relative, 'unclassified helper has module/public-function effects or dynamic dispatch'))
            else:
                excluded.append((relative, 'helper without detected module/public-function writes'))
    return output, excluded, errors


def reviewed_branch(relative, entry, root=ROOT):
    """Recognize two reviewed branch shapes; this is not runtime proof."""
    if entry['entry'] != 'CLI' or not entry['guard_lines'] or not entry['unprotected']:
        return None
    meter = 'lab/features/meter/training/scripts/audit_meter_batch.py'
    closeups = 'lab/features/waste_bin/training/scripts/generate_sticker_closeups_v4.py'
    if relative not in (meter, closeups):
        return None
    try:
        tree = ast.parse((root / relative).read_text(encoding='utf8'))
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        if relative == meter:
            check, worker = functions['check'], functions['_check']
            branch = next(n for n in check.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'previews <= 0')
            assert len(branch.body) == 1 and ast.unparse(branch.body[0]) == 'return _check(root, previews)'
            loop = next(n for n in worker.body if isinstance(n, ast.For) and ast.unparse(n.target) == '(n, s)' and ast.unparse(n.iter) == 'enumerate(samples)')
            condition = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'n < previews')
            effect_lines = {n.lineno for n in ast.walk(condition) if isinstance(n, ast.Call)}
            assert all(line in effect_lines for line, _ in entry['unprotected'])
            return 'static test: lab/scripts/audit/test_production_entry_coverage.py::test_meter_zero_preview_branch_structure; no dedicated runtime test identified'
        main, worker = functions['main'], functions['_generate']
        branch = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'args.audit_only')
        assert len(branch.body) == 1 and ast.unparse(branch.body[0]) == 'return _generate()'
        early = next(n for n in worker.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'args.audit_only')
        assert len(early.body) == 1 and isinstance(early.body[0], ast.Return) and early.body[0].value is None
        assert all(line > early.end_lineno for line, _ in entry['unprotected'])
        test_path = 'lab/features/waste_bin/training/tests/test_sticker_generation_guard.py'
        tests = ast.parse((root / test_path).read_text(encoding='utf8'))
        test = next(n for n in ast.walk(tests) if isinstance(n, ast.FunctionDef) and n.name == 'test_closeups_audit_only_does_not_write')
        literals = {n.value for n in ast.walk(test) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert {'mkdir', 'write_text', '--audit-only'} <= literals
        return f'targeted test: {test_path}::test_closeups_audit_only_does_not_write (not run by this static checker)'
    except (OSError, SyntaxError, UnicodeError, KeyError, StopIteration, AssertionError):
        return None


class AnalyzerTests(unittest.TestCase):
    def check(self, text):
        return analyze_text(text, ROOT / 'lab/features/fire/training/scripts/fixture.py')

    def test_relocated_sign_adapter_uses_task_owned_guard(self):
        tree = ast.parse('from sign_production_guard import guarded')
        path = ROOT / 'datasets/scripts/sign/compose_sign_v4_measured.py'
        with mock.patch(__name__ + '.adapter_valid', return_value=True) as validate:
            guards = trusted_guards(tree, path, ROOT)
        self.assertIn('guarded', guards)
        validate.assert_called_once_with(ROOT / 'datasets/scripts/sign/sign_production_guard.py')

    def test_guarded_call_chain(self):
        checks = self.check('from lab.training_common.production_guard import generation_guard\n'
                            'def write():\n open("x", "w")\n'
                            'def main():\n with generation_guard():\n  write()\n'
                            'if __name__ == "__main__": main()\n')
        self.assertTrue(checks[0]['guard_lines'])
        self.assertFalse(checks[0]['unprotected'])

    def test_write_before_guard_and_public_bypass(self):
        checks = self.check('from lab.training_common.production_guard import generation_guard\n'
                            'def generate():\n open("x", "w")\n'
                            'def main():\n generate()\n with generation_guard():\n  pass\n'
                            'if __name__ == "__main__": main()\n')
        self.assertTrue(all(x['unprotected'] for x in checks))

    def test_untrusted_and_shadowed_guard(self):
        for prefix in ('from other import generation_guard\n',
                       'from lab.training_common.production_guard import generation_guard\n'
                       'def generation_guard(): pass\n'):
            checks = self.check(prefix + 'def main():\n with generation_guard():\n  open("x", "w")\n'
                                'if __name__ == "__main__": main()\n')
            self.assertFalse(checks[0]['guard_lines'])
            self.assertTrue(checks[0]['unprotected'])

    def test_blender_module_scope_mutation(self):
        checks = self.check('import bpy\nbpy.context.scene.render.filepath = "x"\n')
        self.assertTrue(checks[0]['unprotected'])

    def test_new_unprotected_cli_is_reported(self):
        checks = self.check('def main():\n open("new.png", "wb")\n'
                            'if __name__ == "__main__": main()\n')
        self.assertFalse(checks[0]['guard_lines'])
        self.assertTrue(checks[0]['unprotected'])

    def test_known_cross_file_delegation(self):
        path = ROOT / 'lab/features/e_bike/training/scripts/check_v4_scaled_parking.py'
        tree = ast.parse(path.read_text(encoding='utf8'))
        checks = scaled_parking_delegation(tree, path, ROOT)
        self.assertIsNotNone(checks)
        self.assertTrue(checks[0]['guard_lines'])
        self.assertFalse(checks[0]['unprotected'])

    def test_meter_zero_preview_branch_structure(self):
        relative = 'lab/features/meter/training/scripts/audit_meter_batch.py'
        path = ROOT / relative
        cli = next(x for x in analyze_text(path.read_text(encoding='utf8'), path) if x['entry'] == 'CLI')
        self.assertIsNotNone(reviewed_branch(relative, cli))

    def test_closeups_audit_only_branch_and_fixture(self):
        relative = 'lab/features/waste_bin/training/scripts/generate_sticker_closeups_v4.py'
        path = ROOT / relative
        cli = next(x for x in analyze_text(path.read_text(encoding='utf8'), path) if x['entry'] == 'CLI')
        self.assertEqual(classify_entry(relative, cli)[0], 'DISABLED')

    def test_retired_cli_does_not_hide_importable_bypass(self):
        checks = self.check('if __name__ == "__main__": raise SystemExit("retired")\n'
                            'def generate():\n open("x", "w")\n'
                            'if __name__ == "__main__": generate()\n')
        self.assertEqual(len([x for x in checks if x['entry'] == 'CLI']), 1)
        self.assertEqual(classify_entry('fixture.py', checks[0])[0], 'DISABLED')
        self.assertEqual(classify_entry('fixture.py', checks[1])[0], 'BYPASS')

    def test_retired_cli_does_not_hide_importable_main(self):
        checks = self.check('if __name__ == "__main__": raise SystemExit("retired")\n'
                            'def main():\n open("x", "w")\n')
        imported = next(row for row in checks if row['entry'] == 'main')
        self.assertEqual(classify_entry('fixture.py', imported)[0], 'BYPASS')

    def test_marker_and_conditional_exit_do_not_disable_cli(self):
        checks = self.check('message="retired"\nif __name__ == "__main__":\n'
                            ' if flag: raise SystemExit("retired")\n open("x", "w")\n')
        self.assertEqual(classify_entry('fixture.py', checks[0])[0], 'BYPASS')

    def test_path_open_write_is_detected_but_stdout_is_not(self):
        checks = self.check('if __name__ == "__main__":\n'
                            ' json.dump(data, sys.stdout)\n path.open("w")\n')
        self.assertEqual(len(checks[0]['unprotected']), 1)
        self.assertIn('path.open', checks[0]['unprotected'][0][1])

    def test_forwarding_checks_target_and_does_not_trust_shim_name(self):
        relative = 'lab/features/sign/training/scripts/fixture.py'
        path = ROOT / relative
        shim = ('from pathlib import Path\nimport runpy\n'
                '_target=Path(__file__).resolve().parents[5]/"datasets/scripts/sign/fixture.py"\n'
                'if __name__ == "__main__": runpy.run_path(str(_target), run_name="__main__")\n')
        safe = ('from lab.training_common.production_guard import generation_guard\n'
                'if __name__ == "__main__":\n with generation_guard(): open("x", "w")\n')
        unsafe = 'if __name__ == "__main__": open("x", "w")\n'
        entry = analyze_text(shim, path)[0]
        for target_text, expected in ((safe, 'FORWARDED'), (unsafe, 'UNPROVED')):
            with mock.patch.object(Path, 'is_file', return_value=True), mock.patch.object(
                    Path, 'read_text', side_effect=lambda p=None, **kw: shim) as read:
                read.side_effect = [shim, target_text]
                self.assertEqual(classify_entry(relative, entry)[0], expected)

    def test_planner_with_media_writes_is_not_metadata_only(self):
        relative = 'datasets/scripts/sign/plan_fixture.py'
        safe = self.check('if __name__ == "__main__":\n output.write_text(json.dumps(plan))\n')[0]
        unsafe = self.check('if __name__ == "__main__":\n output.write_text(json.dumps(plan))\n image.save("x.png")\n')[0]
        self.assertEqual(classify_entry(relative, safe)[0], 'PLAN_ONLY')
        self.assertEqual(classify_entry(relative, unsafe)[0], 'BYPASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnalyzerTests)).wasSuccessful() else 1
    output, excluded, errors = inventory()
    counts = Counter()
    for path, entries in output:
        for entry in entries:
            status, reason = classify_entry(path, entry)
            counts[status] += 1
            if status in ('BYPASS', 'UNPROVED', 'FORWARDED', 'REVIEWED_BRANCH'):
                print(f"{status} {path}::{entry['entry']} {reason} effects={entry['unprotected']}")
    for path, reason in errors:
        print(f'UNCLASSIFIED {path}: {reason}')
    categories = Counter(reason for _, reason in excluded)
    print(f"SCOPE files={len(output)} entries={sum(len(x) for _, x in output)} "
          f"excluded={len(excluded)} unclassified={len(errors)} " + ' '.join(f'{k.lower()}={v}' for k, v in sorted(counts.items())))
    print('EXCLUDED ' + ', '.join(f'{reason}={count}' for reason, count in sorted(categories.items())))
    print('BOUNDARY: lexical conservative analysis; no runtime, dynamic import, branch feasibility, input closure, or database validation.')
    return 1 if counts['BYPASS'] or counts['UNPROVED'] or errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
