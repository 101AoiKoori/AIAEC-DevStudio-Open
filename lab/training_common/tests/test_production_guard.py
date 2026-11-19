"""Production receipt, provenance and failure checks using the index fixture."""
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import types
import unittest
from unittest import mock

MEDIA_TESTS = Path(__file__).resolve().parents[2] / 'scripts/media/tests'
sys.path.insert(0, str(MEDIA_TESTS))
import test_media_index_sqlite as fixtures

MODULE_PATH = Path(__file__).resolve().parents[1] / 'production_guard.py'
SPEC = importlib.util.spec_from_file_location('production_guard', MODULE_PATH)
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


class ProductionGuardTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CacheTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.image, self.parts = self.fixture.root, self.fixture.image, self.fixture.parts
        shard = self.root / self.parts['datasets/raw/camera']['shard']
        item = json.loads(shard.read_text(encoding='utf8'))
        stat = self.image.stat()
        item.update(bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
        shard.write_text(json.dumps(item) + '\n', encoding='utf8')
        self.assertEqual(self.fixture.call(['sync'])[0], 0)
        (self.root / 'fixtures').mkdir(exist_ok=True)
        self.config = self.root / 'fixtures' / 'production_config.json'
        self.config.write_text('{"seed":1}', encoding='utf8')
        self.code_file = self.root / 'fixtures' / 'generator.py'
        self.code_file.write_text('print(1)', encoding='utf8')
        self.output = self.root / 'datasets' / 'outputs' / 'batch.txt'
        self.args = {'operation': 'fixture-generation',
                     'input_paths': ['datasets/raw/camera/a.png', 'fixtures/production_config.json'],
                     'output_paths': ['datasets/outputs/batch.txt'],
                     'parameters': {'seed': 1}, 'code_paths': ['fixtures/generator.py'],
                     'procedural_inputs': ['fixtures/production_config.json'], 'root': self.root}

    def produce(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text('result', encoding='utf8')

    def history(self, operation='fixture-generation'):
        with sqlite3.connect(str(fixtures.core.INDEX / 'production_state.sqlite3')) as db:
            return db.execute('SELECT run_token,origin,receipt_json FROM production_receipt_history '
                              'WHERE operation=? ORDER BY id', (operation,)).fetchall()

    def test_receipt_history_tracks_completions_not_reuse_or_failure(self):
        with guard.generation_guard(**self.args):
            self.produce()
        first = self.history()
        self.assertEqual(len(first), 1)
        self.assertTrue(first[0][0])
        self.assertEqual(first[0][1], 'completed_run')

        with guard.generation_guard(**self.args) as reused:
            self.assertTrue(reused.reused)
        self.assertEqual(self.history(), first)

        changed = dict(self.args, parameters={'seed': 2})
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            with guard.generation_guard(**changed):
                raise RuntimeError('fixture failure')
        self.assertEqual(self.history(), first)
        with guard.generation_guard(**changed):
            self.produce()
        second = self.history()
        self.assertEqual(len(second), 2)
        self.assertNotEqual(second[0][0], second[1][0])
        self.assertEqual(second[0], first[0])
        self.assertEqual(second[1][1], 'completed_run')
        with sqlite3.connect(str(fixtures.core.INDEX / 'production_state.sqlite3')) as db:
            latest = db.execute('SELECT receipt_json,running_json FROM production_operations WHERE operation=?',
                                ('fixture-generation',)).fetchone()
        self.assertEqual(latest, (second[1][2], None))
        self.assertEqual(json.loads(second[0][2])['request']['parameters']['seed'], 1)
        self.assertEqual(json.loads(second[1][2])['request']['parameters']['seed'], 2)

    def test_existing_state_receipt_is_preserved_once_as_snapshot(self):
        state_path = fixtures.core.INDEX / 'production_state.sqlite3'
        with sqlite3.connect(str(state_path)) as db:
            db.execute('CREATE TABLE production_operations '
                       '(operation TEXT PRIMARY KEY, receipt_json TEXT, running_json TEXT)')
            db.execute('INSERT INTO production_operations VALUES(?,?,?)',
                       ('old', '{"known":true}', None))
        with guard.generation_guard(**self.args):
            self.produce()
        self.assertEqual(self.history('old'), [(None, 'existing_snapshot', '{"known":true}')])
        with guard.generation_guard(**self.args) as reused:
            self.assertTrue(reused.reused)
        self.assertEqual(self.history('old'), [(None, 'existing_snapshot', '{"known":true}')])

    def test_receipt_history_and_latest_update_rollback_together(self):
        with guard.generation_guard(**self.args):
            self.produce()
        first = self.history()
        with sqlite3.connect(str(fixtures.core.INDEX / 'production_state.sqlite3')) as db:
            db.execute('CREATE TRIGGER reject_receipt_update BEFORE UPDATE OF receipt_json '
                       'ON production_operations BEGIN '
                       "SELECT RAISE(ABORT, 'fixture update rejected'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'fixture update rejected'):
            with guard.generation_guard(**dict(self.args, parameters={'seed': 2})):
                self.produce()
        self.assertEqual(self.history(), first)
        with sqlite3.connect(str(fixtures.core.INDEX / 'production_state.sqlite3')) as db:
            latest = db.execute('SELECT receipt_json,running_json FROM production_operations WHERE operation=?',
                                ('fixture-generation',)).fetchone()
        self.assertEqual(latest, (first[0][2], None))

    def test_unplanned_replacement_requires_verifier_and_exposes_committed_token(self):
        with guard.generation_guard(**self.args):
            self.produce()
        replacement = dict(self.args, replace_outputs=True, parameters={'seed': 2})
        with self.assertRaisesRegex(ValueError, 'requires a production plan'):
            with guard.generation_guard(**replacement):
                self.fail('unverified replacement must not start')
        calls = []
        def verify(workspace, output_state, previous):
            calls.append((workspace, output_state, previous))
            self.assertEqual(workspace, self.root.resolve())
            self.assertEqual(output_state[0]['path'], 'datasets/outputs/batch.txt')
            self.assertEqual(previous['request']['parameters']['seed'], 1)
            return True
        with guard.generation_guard(**replacement, replace_verifier=verify) as run:
            token = run.token
            self.assertTrue(token)
            self.assertFalse(guard.receipt_completed(self.root, 'fixture-generation', token))
            self.output.write_text('replacement', encoding='utf8')
        self.assertEqual(len(calls), 1)
        self.assertEqual(run.token, '')
        self.assertTrue(guard.receipt_completed(self.root, 'fixture-generation', token))
        self.assertFalse(guard.receipt_completed(self.root, 'other', token))
        self.assertEqual(len(self.history()), 2)

    def test_verifier_rejects_and_must_not_change_output(self):
        with guard.generation_guard(**self.args):
            self.produce()
        before = self.output.read_text(encoding='utf8')
        replace = dict(self.args, replace_outputs=True)
        for result in (False, None, 1):
            with self.subTest(result=result), self.assertRaisesRegex(ValueError, 'did not approve'):
                with guard.generation_guard(**replace, replace_verifier=lambda *_: result):
                    self.fail('unapproved verifier must not start')
        def mutate(_workspace, _state, _receipt):
            self.output.write_text('modified by verifier', encoding='utf8')
            return True
        with self.assertRaisesRegex(ValueError, 'changed output state'):
            with guard.generation_guard(**replace, replace_verifier=mutate):
                self.fail('mutating verifier must not start')
        self.assertNotEqual(self.output.read_text(encoding='utf8'), before)
        self.assertEqual(len(self.history()), 1)

    def test_receipt_lookup_is_read_only_and_does_not_guess_missing_state(self):
        token = 'a' * 32
        state = fixtures.core.INDEX / 'production_state.sqlite3'
        with self.assertRaises(FileNotFoundError):
            guard.receipt_completed(self.root, 'fixture-generation', token)
        self.assertFalse(state.exists())
        with sqlite3.connect(str(state)) as db:
            db.execute('CREATE TABLE old_state (operation TEXT)')
        with self.assertRaisesRegex(RuntimeError, 'history is unavailable'):
            guard.receipt_completed(self.root, 'fixture-generation', token)
        with sqlite3.connect(str(state)) as db:
            self.assertIsNone(db.execute("SELECT 1 FROM sqlite_master WHERE name='production_receipt_history'").fetchone())
        with self.assertRaises(ValueError):
            guard.receipt_completed(self.root, 'fixture-generation', '../invalid')

    def test_unplanned_child_replacement_is_rejected_before_verifier(self):
        parent_args = dict(self.args, operation='parent', output_paths=['datasets/outputs/parent'])
        child_args = dict(self.args, operation='child', output_paths=['datasets/outputs/parent/child.txt'],
                          replace_outputs=True)
        verifier = mock.Mock(return_value=True)
        with self.assertRaisesRegex(RuntimeError, 'child production stage failed'):
            with guard.generation_guard(**parent_args) as parent:
                marker = self.root / 'datasets/outputs/parent/marker.txt'
                marker.parent.mkdir(parents=True)
                marker.write_text('x', encoding='utf8')
                with mock.patch.dict(os.environ, parent.child_env(), clear=True):
                    with self.assertRaisesRegex(ValueError, 'only supported by a top-level'):
                        with guard.generation_guard(**child_args, replace_verifier=verifier):
                            self.fail('child replacement must not start')
        verifier.assert_not_called()

    def test_active_operation_blocks_replacement_before_verifier(self):
        verifier = mock.Mock(return_value=True)
        with guard.generation_guard(**self.args) as first:
            self.produce()
            token = first.token
            with self.assertRaisesRegex(RuntimeError, 'already in use'):
                with guard.generation_guard(**dict(self.args, replace_outputs=True),
                                            replace_verifier=verifier):
                    self.fail('concurrent replacement must not start')
            verifier.assert_not_called()
            self.assertFalse(guard.receipt_completed(self.root, 'fixture-generation', token))
        self.assertTrue(guard.receipt_completed(self.root, 'fixture-generation', token))

    def test_replacement_exception_never_reports_completion(self):
        with guard.generation_guard(**self.args):
            self.produce()
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with guard.generation_guard(**dict(self.args, replace_outputs=True),
                                        replace_verifier=lambda *_: True) as run:
                token = run.token
                raise RuntimeError('interrupted')
        self.assertFalse(guard.receipt_completed(self.root, 'fixture-generation', token))
        self.assertEqual(len(self.history()), 1)

    def test_planned_replacement_keeps_verified_files_out_of_samples(self):
        args, state = self.planned_args()
        manual = self.root / 'datasets/outputs/batch/images/manual.png'
        note = self.root / 'datasets/outputs/batch/manual-notes.txt'
        image = self.root / 'datasets/outputs/batch/images/generated.jpg'
        manual.parent.mkdir(parents=True)
        manual.write_bytes(b'manual')
        note.write_text('human', encoding='utf8')
        def verify(workspace, _state, _previous):
            return {'preserved_paths': [str(manual.relative_to(workspace)), str(note.relative_to(workspace))]}
        with guard.generation_guard(**args, replace_outputs=True, replace_verifier=verify) as run:
            with self.assertRaisesRegex(ValueError, 'preserved output cannot be recorded'):
                run.record_sample('one', output=manual, sources=state['sources'],
                                  parameters=state['parameters'], actual={'seed': 1})
            image.write_bytes(b'generated')
            run.record_sample('one', output=image, sources=state['sources'],
                              parameters=state['parameters'], actual={'seed': 1})
        with sqlite3.connect(str(fixtures.core.INDEX / 'production_state.sqlite3')) as db:
            receipt = json.loads(db.execute('SELECT receipt_json FROM production_operations WHERE operation=?',
                                            ('fixture-generation',)).fetchone()[0])
        self.assertEqual([r['output'] for r in receipt['planned_samples']],
                         ['datasets/outputs/batch/images/generated.jpg'])
        self.assertEqual({r['path'] for r in receipt['preserved_outputs']},
                         {'datasets/outputs/batch/images/manual.png',
                          'datasets/outputs/batch/manual-notes.txt'})
        self.assertEqual(manual.read_bytes(), b'manual')
        self.assertEqual(note.read_text(encoding='utf8'), 'human')

    def test_planned_replacement_rejects_invalid_preserved_paths(self):
        args, _state = self.planned_args()
        manual = self.root / 'datasets/outputs/batch/images/manual.png'
        manual.parent.mkdir(parents=True)
        manual.write_bytes(b'manual')
        for result in (True, {'preserved_paths': ['../outside']},
                       {'preserved_paths': ['datasets/outputs/batch/images/missing.png']},
                       {'preserved_paths': ['datasets/outputs/batch/images']},
                       {'preserved_paths': ['datasets/outputs/batch/images/manual.png'] * 2},
                       {'preserved_paths': ['datasets/raw/camera/a.png']}):
            with self.subTest(result=result), self.assertRaises((ValueError, TypeError)):
                with guard.generation_guard(**args, replace_outputs=True,
                                            replace_verifier=lambda *_: result):
                    self.fail('invalid preserved output must not be admitted')
        self.assertEqual(manual.read_bytes(), b'manual')

    def test_planned_replacement_detects_preserved_content_change_even_if_stat_restored(self):
        args, state = self.planned_args()
        manual = self.root / 'datasets/outputs/batch/images/manual.png'
        image = self.root / 'datasets/outputs/batch/images/generated.jpg'
        manual.parent.mkdir(parents=True)
        manual.write_bytes(b'original')
        stat = manual.stat()
        with self.assertRaisesRegex(ValueError, 'preserved output was removed or changed'):
            with guard.generation_guard(**args, replace_outputs=True,
                                        replace_verifier=lambda *_: {'preserved_paths': [str(manual)]}) as run:
                token = run.token
                image.write_bytes(b'generated')
                run.record_sample('one', output=image, sources=state['sources'],
                                  parameters=state['parameters'], actual={'seed': 1})
                manual.write_bytes(b'overwrit')
                os.utime(manual, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertFalse(guard.receipt_completed(self.root, 'fixture-generation', token))

    def test_planned_replacement_without_verifier_remains_strict(self):
        args, state = self.planned_args()
        manual = self.root / 'datasets/outputs/batch/images/manual.png'
        image = self.root / 'datasets/outputs/batch/images/generated.jpg'
        manual.parent.mkdir(parents=True)
        manual.write_bytes(b'manual')
        with self.assertRaisesRegex(ValueError, 'output images do not match recorded samples'):
            with guard.generation_guard(**args, replace_outputs=True) as run:
                image.write_bytes(b'generated')
                run.record_sample('one', output=image, sources=state['sources'],
                                  parameters=state['parameters'], actual={'seed': 1})

    def test_relocated_task_rejects_old_output_and_missing_source_config(self):
        code = ['datasets/scripts/sign/generator.py']
        configs = ['datasets/scripts/sign/config/dataset.yaml',
                   'datasets/scripts/sign/config/source.yaml']
        args = dict(self.args, code_paths=code, procedural_inputs=configs,
                    output_paths=['runs/sign/old_batch'])
        with self.assertRaisesRegex(ValueError, 'train/test/val'):
            with guard.generation_guard(**args):
                self.fail('old output must not enter the production body')
        args.update(output_paths=['datasets/samples/sign'], procedural_inputs=configs[:1])
        with self.assertRaisesRegex(ValueError, 'dataset.yaml and source.yaml'):
            with guard.generation_guard(**args):
                self.fail('missing source configuration must not enter the production body')

    def planned_args(self):
        state = {'id': 'one', 'failure_event': 'miss', 'min_count': 1, 'max_count': 1,
                 'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
                 'conditions': ['normal'], 'background': {'id': 'camera'},
                 'target': {'class': 'person'}, 'labels': {'format': 'yolo'},
                 'augmentation': {'brightness': 'normal'}, 'sources': ['camera', 'person'],
                 'parameters': {'yaw': 0}}
        declaration = {'schema_version': 1, 'model_key': 'person', 'target_split': 'train',
                       'purpose': 'synthetic test', 'budget': 1, 'uncertainty_weight': 0,
                       'independence_basis': 'fixture', 'constraints': {'size': [640, 480]},
                       'strata': [state]}
        args = dict(self.args, output_paths=['datasets/outputs/batch'], production_plan=declaration,
                    sample_output_paths=['datasets/outputs/batch/images'],
                    auxiliary_output_paths=['datasets/outputs/batch/masks'])
        return args, state

    def assert_inactive(self, run, state):
        self.assertFalse(run.active)
        self.assertEqual(run._token, '')
        with self.assertRaisesRegex(ValueError, 'active planned production run'):
            run.record_sample(state['id'], output='datasets/outputs/batch/images/a.jpg',
                              sources=state['sources'], parameters=state['parameters'], actual={'seed': 1})
        with self.assertRaisesRegex(ValueError, 'reused or inactive'):
            run.child_env()
        with self.assertRaisesRegex(ValueError, 'active production run'):
            run.set_result({'late': True})

    def test_planned_production_records_actual_and_auxiliary_and_reuses(self):
        args, state = self.planned_args()
        context = {'task': 'person', 'coverage': 'unverified', 'executable': False,
                   'crosses': [{'id': 'category_context', 'coverage': 'unverified'}]}
        args['production_plan']['design_context'] = context
        image = self.root / 'datasets/outputs/batch/images/a.jpg'
        mask = self.root / 'datasets/outputs/batch/masks/a.png'
        with guard.generation_guard(**args) as run:
            self.assertTrue(run.active)
            self.assertEqual(run.plan['strata'][0]['planned_count'], 1)
            copy = run.plan
            copy['strata'][0]['planned_count'] = 100
            image.parent.mkdir(parents=True)
            mask.parent.mkdir(parents=True)
            image.write_bytes(b'image')
            mask.write_bytes(b'mask')
            run.record_sample('one', output=image, sources=state['sources'], parameters=state['parameters'],
                              actual={'seed': 73, 'source_file': 'datasets/raw/camera/a.png'})
        self.assert_inactive(run, state)
        self.assertEqual(run.plan['strata'][0]['planned_count'], 1)
        db = sqlite3.connect(str(fixtures.core.INDEX / 'production_state.sqlite3'))
        try:
            receipt = json.loads(db.execute('SELECT receipt_json FROM production_operations WHERE operation=?',
                                            ('fixture-generation',)).fetchone()[0])
        finally:
            db.close()
        self.assertEqual(receipt['planned_samples'][0]['actual']['seed'], 73)
        self.assertEqual(receipt['request']['production_plan']['design_context'], context)
        self.assertEqual(len(receipt['planned_samples']), 1)
        with guard.generation_guard(**args) as run:
            self.assertTrue(run.reused)
            self.assertEqual(run.plan['strata'][0]['planned_count'], 1)
            self.assert_inactive(run, state)
        self.assert_inactive(run, state)

    def test_run_is_invalidated_after_body_or_completion_failure(self):
        args, state = self.planned_args()
        with self.assertRaisesRegex(RuntimeError, 'stopped'):
            with guard.generation_guard(**args) as run:
                self.assertTrue(run.active)
                self.assertTrue(run.child_env()['AEC_PRODUCTION_PARENT_TOKEN'])
                raise RuntimeError('stopped')
        self.assert_inactive(run, state)
        with self.assertRaisesRegex(ValueError, 'required path is missing'):
            with guard.generation_guard(**args) as run:
                self.assertTrue(run.active)
        self.assert_inactive(run, state)

    def test_previously_captured_child_environment_cannot_reopen_finished_parent(self):
        parent_args = dict(self.args, operation='finished-parent', output_paths=['datasets/outputs/parent'])
        child_args = dict(self.args, operation='late-child', output_paths=['datasets/outputs/parent/child.txt'])
        with guard.generation_guard(**parent_args) as parent:
            child_env = parent.child_env()
            marker = self.root / 'datasets/outputs/parent/marker.txt'
            marker.parent.mkdir(parents=True)
            marker.write_text('completed', encoding='utf8')
        with mock.patch.dict(os.environ, child_env, clear=True):
            with self.assertRaisesRegex(ValueError, 'invalid or inactive parent'):
                with guard.generation_guard(**child_args):
                    self.fail('a finished parent cannot authorize a new child')
    def test_planned_child_is_bound_to_parent_request_and_receipt(self):
        child, state = self.planned_args()
        child['operation'] = 'planned-child'
        child['output_paths'] = ['datasets/outputs/parent/child']
        child['sample_output_paths'] = ['datasets/outputs/parent/child/images']
        child['auxiliary_output_paths'] = []
        plan_file = self.root / 'fixtures/child-plan.json'
        plan_file.write_text(json.dumps(child['production_plan']), encoding='utf8')
        planner = self.root / 'lab/training_common/bayesian_allocation.py'
        planner.parent.mkdir(parents=True)
        planner.write_text('fixture planner', encoding='utf8')
        child['procedural_inputs'] = [*child['procedural_inputs'], 'fixtures/child-plan.json']
        parent_args = dict(self.args, operation='parent', output_paths=['datasets/outputs/parent'],
                           code_paths=[*self.args['code_paths'], 'lab/training_common/bayesian_allocation.py'],
                           procedural_inputs=[*self.args['procedural_inputs'], 'fixtures/child-plan.json'],
                           parameters={'child_production_plans': {'planned-child': child['production_plan']}})
        image = self.root / 'datasets/outputs/parent/child/images/a.jpg'
        with guard.generation_guard(**parent_args) as parent:
            with mock.patch.dict(os.environ, parent.child_env(), clear=True):
                with guard.generation_guard(**child) as run:
                    image.parent.mkdir(parents=True)
                    image.write_bytes(b'image')
                    run.record_sample(state['id'], output=image, sources=state['sources'],
                                      parameters=state['parameters'], actual={'seed': 1})
            self.assert_inactive(run, state)
        self.assert_inactive(parent, state)
        db = sqlite3.connect(str(fixtures.core.INDEX / 'production_state.sqlite3'))
        try:
            receipt = json.loads(db.execute('SELECT receipt_json FROM production_operations WHERE operation=?',
                                            ('parent',)).fetchone()[0])
        finally:
            db.close()
        self.assertEqual(receipt['request']['parameters']['child_production_plans']['planned-child'],
                         child['production_plan'])
        self.assertEqual(receipt['stages'][0]['planned_samples'][0]['state_id'], state['id'])
        with guard.generation_guard(**parent_args) as reused:
            self.assertTrue(reused.reused)
            self.assertFalse(reused.active)
        changed = json.loads(json.dumps(child['production_plan']))
        changed['purpose'] = 'changed declaration'
        plan_file.write_text(json.dumps(changed), encoding='utf8')
        changed_args = dict(parent_args, parameters={'child_production_plans': {'planned-child': changed}})
        with self.assertRaisesRegex(RuntimeError, 'stop before child'):
            with guard.generation_guard(**changed_args) as fresh:
                self.assertFalse(fresh.reused)
                raise RuntimeError('stop before child')
        with sqlite3.connect(str(fixtures.core.INDEX / 'production_state.sqlite3')) as db:
            retained = json.loads(db.execute('SELECT receipt_json FROM production_operations WHERE operation=?',
                                             ('parent',)).fetchone()[0])
        self.assertEqual(retained, receipt)

    def test_parent_rejects_unbound_or_mismatched_planned_child(self):
        child, _ = self.planned_args()
        child.update(operation='planned-child', output_paths=['datasets/outputs/parent/child'],
                     sample_output_paths=['datasets/outputs/parent/child/images'], auxiliary_output_paths=[])
        planner = self.root / 'lab/training_common/bayesian_allocation.py'
        planner.parent.mkdir(parents=True)
        planner.write_text('fixture planner', encoding='utf8')
        parent_args = dict(self.args, operation='parent', output_paths=['datasets/outputs/parent'],
                           code_paths=[*self.args['code_paths'], 'lab/training_common/bayesian_allocation.py'])
        for index, (parameters, message) in enumerate([({'seed': 1}, 'parent must bind'),
                                    ({'child_production_plans': {'planned-child': {'budget': 99}}},
                                     'differs from the parent binding')]):
            parent_scope = f'datasets/outputs/parent-{index}'
            attempt_parent = dict(parent_args, parameters=parameters, output_paths=[parent_scope])
            attempt_child = dict(child, output_paths=[parent_scope + '/child'],
                                 sample_output_paths=[parent_scope + '/child/images'])
            with self.subTest(parameters=parameters), self.assertRaisesRegex(RuntimeError, 'child production stage failed'):
                with guard.generation_guard(**attempt_parent) as parent:
                    marker = self.root / parent_scope / 'started.txt'
                    marker.parent.mkdir(parents=True)
                    marker.write_text('started', encoding='utf8')
                    with mock.patch.dict(os.environ, parent.child_env(), clear=True):
                        with self.assertRaisesRegex(ValueError, message):
                            with guard.generation_guard(**attempt_child):
                                self.fail('unbound child must not start')

    def test_parent_requires_every_bound_planned_child_before_completion(self):
        child, _ = self.planned_args()
        parent_args = dict(self.args, operation='parent', output_paths=['datasets/outputs/parent'],
                           parameters={'child_production_plans': {'planned-child': child['production_plan']}})
        with self.assertRaisesRegex(ValueError, 'planned child operations differ'):
            with guard.generation_guard(**parent_args):
                output = self.root / 'datasets/outputs/parent/ordinary.txt'
                output.parent.mkdir(parents=True)
                output.write_text('not the declared child', encoding='utf8')
    def test_planned_production_rejects_mismatch_and_unrecorded_images(self):
        args, state = self.planned_args()
        image = self.root / 'datasets/outputs/batch/images/a.jpg'
        image.parent.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'no matching complete receipt'):
            image.write_bytes(b'image')
            with guard.generation_guard(**args):
                pass
        image.unlink()
        image.parent.rmdir()
        image.parent.parent.rmdir()
        with self.assertRaisesRegex(ValueError, 'differ from the declared state'):
            with guard.generation_guard(**args) as run:
                image.parent.mkdir(parents=True)
                image.write_bytes(b'image')
                run.record_sample('one', output=image, sources=['different'],
                                  parameters=state['parameters'], actual={'seed': 1})
        image.unlink()
        image.parent.rmdir()
        image.parent.parent.rmdir()
        with self.assertRaisesRegex(ValueError, 'outside declared sample and auxiliary'):
            with guard.generation_guard(**args) as run:
                image.parent.mkdir(parents=True)
                image.write_bytes(b'image')
                run.record_sample('one', output=image, sources=state['sources'],
                                  parameters=state['parameters'], actual={'seed': 1})
                (image.parent.parent / 'unknown.png').write_bytes(b'unknown')

    def test_planned_production_rejects_missing_counts_and_overlapping_scopes(self):
        args, state = self.planned_args()
        bad = dict(args, auxiliary_output_paths=['datasets/outputs/batch/images/masks'])
        with self.assertRaisesRegex(ValueError, 'must not overlap'):
            with guard.generation_guard(**bad):
                pass
        image = self.root / 'datasets/outputs/batch/images/a.jpg'
        with self.assertRaisesRegex(ValueError, 'counts do not match'):
            with guard.generation_guard(**args):
                image.parent.mkdir(parents=True)
                image.write_bytes(b'image')

    def test_planner_uses_workspace_python_when_scipy_is_unavailable(self):
        args, _ = self.planned_args()
        with mock.patch.object(guard.importlib.util, 'find_spec', return_value=None):
            result = guard.plan_production(args['production_plan'], self.root)
        self.assertEqual(result['planned_total'], 1)
        self.assertEqual(result['strata'][0]['planned_count'], 1)
        invalid = dict(args['production_plan'], budget=0)
        with mock.patch.object(guard.importlib.util, 'find_spec', return_value=None):
            with self.assertRaisesRegex(ValueError, 'workspace Python allocation failed'):
                guard.plan_production(invalid, self.root)

    def test_planned_production_rejects_mutation_and_duplicate_record(self):
        args, state = self.planned_args()
        image = self.root / 'datasets/outputs/batch/images/a.jpg'
        with self.assertRaisesRegex(ValueError, 'changed after recording'):
            with guard.generation_guard(**args) as run:
                image.parent.mkdir(parents=True)
                image.write_bytes(b'original')
                run.record_sample('one', output=image, sources=state['sources'],
                                  parameters=state['parameters'], actual={'seed': 3})
                with self.assertRaisesRegex(ValueError, 'already recorded'):
                    run.record_sample('one', output=image, sources=state['sources'],
                                      parameters=state['parameters'], actual={'seed': 3})
                image.write_bytes(b'modified and larger')

    def test_planned_production_rejects_missing_actual_and_undeclared_sample(self):
        args, state = self.planned_args()
        image = self.root / 'datasets/outputs/batch/images/a.jpg'
        with self.assertRaisesRegex(ValueError, 'actual must be a nonempty object'):
            with guard.generation_guard(**args) as run:
                image.parent.mkdir(parents=True)
                image.write_bytes(b'image')
                run.record_sample('one', output=image, sources=state['sources'],
                                  parameters=state['parameters'], actual={})

    def test_reuse_changed_input_and_real_provenance(self):
        with guard.generation_guard(**self.args) as run:
            self.assertFalse(run.reused)
            refs = run.source_refs['datasets/raw/camera/a.png']
            self.assertEqual({(x['manifest'], x['record_index'], x['relation']) for x in refs},
                             {('datasets/exports/derived/manifest.json', 0, 'describes'),
                              ('datasets/exports/derived/manifest.json', 0, 'referenced_as')})
            self.assertNotIn('fixtures/production_config.json', run.source_refs)
            self.produce()
        stat = self.output.stat()
        with guard.generation_guard(**self.args) as run:
            self.assertTrue(run.reused)
        self.assertEqual(stat.st_mtime_ns, self.output.stat().st_mtime_ns)
        self.config.write_text('{"seed":2}', encoding='utf8')
        with guard.generation_guard(**self.args) as run:
            self.assertFalse(run.reused)
            self.produce()
        with guard.generation_guard(**self.args) as run:
            self.assertTrue(run.reused)

    def test_untracked_or_modified_output_is_not_adopted(self):
        self.produce()
        with self.assertRaisesRegex(ValueError, 'no matching complete receipt'):
            with guard.generation_guard(**self.args):
                pass
        self.output.unlink()
        with guard.generation_guard(**self.args):
            self.produce()
        self.output.write_text('modified result', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'no matching complete receipt'):
            with guard.generation_guard(**self.args):
                pass

    def test_precreated_empty_output_directory_is_not_an_existing_batch(self):
        output = self.root / 'datasets/outputs/direct_batch'
        output.mkdir(parents=True)
        args = dict(self.args, operation='direct-batch', output_paths=['datasets/outputs/direct_batch'])
        with guard.generation_guard(**args) as run:
            self.assertFalse(run.reused)
            (output / 'image.txt').write_text('result', encoding='utf8')
        with guard.generation_guard(**args) as run:
            self.assertTrue(run.reused)

        untracked = self.root / 'datasets/outputs/untracked_batch'
        untracked.mkdir()
        (untracked / 'image.txt').write_text('untracked', encoding='utf8')
        args = dict(self.args, operation='untracked-batch', output_paths=['datasets/outputs/untracked_batch'])
        with self.assertRaisesRegex(ValueError, 'no matching complete receipt'):
            with guard.generation_guard(**args):
                pass

    def test_failure_missing_output_and_simultaneous_operation(self):
        with self.assertRaises(SystemExit):
            with guard.generation_guard(**self.args):
                raise SystemExit(2)
        with self.assertRaisesRegex(ValueError, 'required path is missing'):
            with guard.generation_guard(**self.args):
                pass
        with guard.generation_guard(**self.args) as run:
            self.assertFalse(run.reused)
            with self.assertRaisesRegex(RuntimeError, 'already in use'):
                with guard.generation_guard(**self.args):
                    pass
            self.produce()

    def test_unindexed_input_with_missing_cache_completes(self):
        extra = self.root / 'datasets/raw/camera/unindexed.png'
        extra.write_bytes(b'unindexed')
        args = dict(self.args, input_paths=['datasets/raw/camera/unindexed.png'], procedural_inputs=[])
        fixtures.cache.DATABASE.unlink()
        with guard.generation_guard(**args) as run:
            self.assertEqual(run.source_refs['datasets/raw/camera/unindexed.png'], [])
            self.produce()
        self.assertEqual(len(self.history()), 1)
        self.assertFalse(fixtures.cache.DATABASE.exists())

    def test_media_change_uses_actual_file_even_with_stale_cache(self):
        with guard.generation_guard(**self.args):
            self.produce()
        self.image.write_bytes(b'changed but not reindexed')
        shard = self.root / self.parts['datasets/raw/camera']['shard']
        shard.write_text(shard.read_text(encoding='utf8') + ' ', encoding='utf8')
        with guard.generation_guard(**self.args) as run:
            self.assertFalse(run.reused)
            self.assertEqual(run.source_refs['datasets/raw/camera/a.png'], [])
            self.produce()
        self.assertEqual(len(self.history()), 2)

    def test_empty_directory_is_not_complete(self):
        args = dict(self.args, output_paths=['datasets/outputs/empty'])
        with self.assertRaisesRegex(ValueError, 'contains no files'):
            with guard.generation_guard(**args):
                (self.root / 'datasets/outputs/empty').mkdir(parents=True)

    def test_first_inplace_and_post_state_reuse(self):
        args = dict(self.args, input_paths=['datasets/raw/camera/a.png'],
                    procedural_inputs=['fixtures/production_config.json'],
                    output_paths=['datasets/raw/camera/a.png'], inplace_paths=['datasets/raw/camera/a.png'])
        with guard.generation_guard(**args) as run:
            self.assertFalse(run.reused)
            self.image.write_bytes(b'in-place output')
        with guard.generation_guard(**args) as run:
            self.assertTrue(run.reused)
            self.assertEqual(run.media_observations['datasets/raw/camera/a.png']['bytes'], len(b'in-place output'))

    def test_indexed_media_without_claims_retains_observation(self):
        shard = self.root / self.parts['datasets/exports/derived']['evidence_shard']
        shard.write_text('', encoding='utf8')
        self.assertEqual(self.fixture.call(['sync'])[0], 0)
        with guard.generation_guard(**self.args) as run:
            self.assertEqual(run.source_refs['datasets/raw/camera/a.png'], [])
            self.assertEqual(run.media_observations['datasets/raw/camera/a.png']['path'],
                             'datasets/raw/camera/a.png')
            self.produce()

    def test_python_dependency_assets_are_scoped_and_tracked(self):
        bundle = 'lab/training_common/portable_env/bundles/win-fixture'
        deps = bundle + '/site-packages'
        directory = self.root / deps
        directory.mkdir(parents=True)
        (directory / 'icon.png').write_bytes(b'png')
        (directory / 'examples.npz').write_bytes(b'npz')
        args = dict(self.args, procedural_inputs=['fixtures/production_config.json', bundle],
                    dependency_paths=[deps])
        with guard.generation_guard(**args) as run:
            self.assertFalse(run.reused)
            self.assertNotIn(deps + '/icon.png', run.source_refs)
            self.produce()
        with guard.generation_guard(**args) as run:
            self.assertTrue(run.reused)
        (directory / 'icon.png').write_bytes(b'changed')
        with guard.generation_guard(**args) as run:
            self.assertFalse(run.reused)
            self.produce()
        with self.assertRaisesRegex(ValueError, 'portable_env bundle site-packages'):
            with guard.generation_guard(**dict(self.args, dependency_paths=['datasets/raw/camera'])):
                pass

    def test_outputless_result_and_adoption_are_explicit(self):
        args = dict(self.args, operation='queue', output_paths=[])
        with guard.generation_guard(**args) as run:
            run.set_result({'items': [1, 2]})
        with guard.generation_guard(**args) as run:
            self.assertTrue(run.reused)
            self.assertEqual(run.value, {'items': [1, 2]})
        self.produce()
        args = dict(self.args, adopt_outputs=['datasets/outputs/batch.txt'])
        with guard.generation_guard(**args) as run:
            self.assertFalse(run.reused)
            self.produce()
        with guard.generation_guard(**args) as run:
            self.assertTrue(run.reused)
        db = sqlite3.connect(str(fixtures.core.INDEX / 'production_state.sqlite3'))
        try:
            receipt = json.loads(db.execute('select receipt_json from production_operations '
                                            'where operation=?', ('fixture-generation',)).fetchone()[0])
            self.assertTrue(receipt['adopted_output_state'])
            self.assertIn('guard_code', receipt['request'])
        finally:
            db.close()

    def test_receipt_survives_index_partition_refresh(self):
        with guard.generation_guard(**self.args):
            self.produce()
        shard = self.root / self.parts['datasets/exports/derived']['shard']
        stat = shard.stat()
        os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assertEqual(self.fixture.call(['sync'])[0], 0)
        with guard.generation_guard(**self.args) as run:
            self.assertTrue(run.reused)

    def test_receipt_survives_query_cache_rebuild(self):
        with guard.generation_guard(**self.args):
            self.produce()
        state = fixtures.core.INDEX / 'production_state.sqlite3'
        self.assertTrue(state.is_file())
        fixtures.cache.DATABASE.unlink()
        self.assertEqual(self.fixture.call(['sync'])[0], 0)
        with guard.generation_guard(**self.args) as run:
            self.assertTrue(run.reused)
        self.assertEqual(self.fixture.call(['status'])[0], 0)

    def test_legacy_receipts_migrate_without_overwrite(self):
        db = fixtures.cache.connect()
        try:
            db.execute('create table production_operations '
                       '(operation text primary key, receipt_json text, running_json text)')
            db.execute('insert into production_operations values(?,?,?)', ('old', '{"saved":true}', None))
            db.commit()
        finally:
            db.close()
        with guard.generation_guard(**self.args):
            self.produce()
        state = sqlite3.connect(str(fixtures.core.INDEX / 'production_state.sqlite3'))
        try:
            self.assertEqual(state.execute('select receipt_json from production_operations where operation=?',
                                           ('old',)).fetchone(), ('{"saved":true}',))
        finally:
            state.close()
        self.assertEqual(self.history('old'), [(None, 'existing_snapshot', '{"saved":true}')])
        db = fixtures.cache.connect(readonly=True)
        try:
            self.assertEqual(db.execute('select receipt_json from production_operations where operation=?',
                                        ('old',)).fetchone(), ('{"saved":true}',))
        finally:
            db.close()

    def test_legacy_import_after_history_creation_preserves_snapshot(self):
        with guard.generation_guard(**self.args):
            self.produce()
        db = fixtures.cache.connect()
        try:
            db.execute('create table production_operations '
                       '(operation text primary key, receipt_json text, running_json text)')
            db.execute('insert into production_operations values(?,?,?)', ('old', '{"saved":true}', None))
            db.commit()
        finally:
            db.close()
        with guard.generation_guard(**self.args) as reused:
            self.assertTrue(reused.reused)
        self.assertEqual(self.history('old'), [(None, 'existing_snapshot', '{"saved":true}')])

    def test_legacy_conflict_is_not_overwritten(self):
        with guard.generation_guard(**self.args):
            self.produce()
        db = fixtures.cache.connect()
        try:
            db.execute('create table production_operations '
                       '(operation text primary key, receipt_json text, running_json text)')
            db.execute('insert into production_operations values(?,?,?)',
                       ('fixture-generation', '{"conflict":true}', None))
            db.commit()
        finally:
            db.close()
        with guard.generation_guard(**self.args) as run:
            self.assertTrue(run.reused)
        db = sqlite3.connect(str(fixtures.core.INDEX / 'production_state.sqlite3'))
        try:
            receipt = db.execute('select receipt_json from production_operations where operation=?',
                                 ('fixture-generation',)).fetchone()[0]
            self.assertNotEqual(receipt, '{"conflict":true}')
        finally:
            db.close()

    def test_index_query_code_changes_do_not_prevent_receipt_reuse(self):
        with guard.generation_guard(**self.args):
            self.produce()
        with guard.generation_guard(**self.args) as run:
            self.assertTrue(run.reused)
        original_stat = Path.stat
        for name in ('media_index.py', 'media_index_sqlite.py'):
            changed = guard._MEDIA / name
            def adjusted(path, *args, **kwargs):
                stat = original_stat(path, *args, **kwargs)
                if path == changed:
                    return types.SimpleNamespace(st_size=stat.st_size, st_mtime_ns=stat.st_mtime_ns + 1)
                return stat
            with mock.patch.object(Path, 'stat', autospec=True, side_effect=adjusted):
                with guard.generation_guard(**self.args) as run:
                    self.assertTrue(run.reused, name)
                code, output, _ = self.fixture.call(['status'])
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(output)['state'], 'ready')

    def test_nested_child_staged_input_and_parent_receipt(self):
        args = dict(self.args, output_paths=['datasets/outputs/scene'])
        stage = self.root / 'datasets/outputs/scene/face.png'
        result = self.root / 'datasets/outputs/scene/render.txt'
        code = ('import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); '
                'from production_guard import generation_guard; '
                'root=Path(sys.argv[2]); '
                'g=generation_guard("child",["datasets/outputs/scene/face.png"],'
                '["datasets/outputs/scene/render.txt"],{},code_paths=["fixtures/generator.py"],root=root); '
                'r=g.__enter__(); assert "datasets/outputs/scene/face.png" in r.staged_inputs; '
                '(root/"datasets/outputs/scene/render.txt").write_text("render"); g.__exit__(None,None,None)')
        with guard.generation_guard(**args) as run:
            stage.parent.mkdir(parents=True)
            stage.write_bytes(b'staged')
            proc = subprocess.run([sys.executable, '-c', code, str(MODULE_PATH.parent), str(self.root)],
                                  env=run.child_env(), capture_output=True, text=True, encoding='utf8', timeout=20)
            self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(result.is_file())
        with guard.generation_guard(**args) as run:
            self.assertTrue(run.reused)
        db = sqlite3.connect(str(fixtures.core.INDEX / 'production_state.sqlite3'))
        try:
            receipt = json.loads(db.execute('select receipt_json from production_operations '
                                            'where operation=?', ('fixture-generation',)).fetchone()[0])
            stage_record = receipt['stages'][0]
            self.assertEqual(stage_record['operation'], 'child')
            self.assertEqual(stage_record['staged_inputs']['datasets/outputs/scene/face.png']['parent_operation'],
                             'fixture-generation')
        finally:
            db.close()

    def test_failed_child_cannot_be_hidden_by_parent(self):
        args = dict(self.args, output_paths=['datasets/outputs/scene'])
        stage = self.root / 'datasets/outputs/scene/face.png'
        with self.assertRaisesRegex(RuntimeError, 'child production stage failed'):
            with guard.generation_guard(**args) as run:
                stage.parent.mkdir(parents=True)
                stage.write_bytes(b'staged')
                child = dict(self.args, operation='child', input_paths=['datasets/outputs/scene/face.png'],
                             procedural_inputs=[], output_paths=['datasets/outputs/scene/render.txt'])
                with mock.patch.dict(os.environ, run.child_env(), clear=True):
                    with self.assertRaisesRegex(RuntimeError, 'child failed'):
                        with guard.generation_guard(**child):
                            raise RuntimeError('child failed')

    def test_child_nonstaged_inputs_must_be_in_parent_closure(self):
        child_config = self.root / 'fixtures/child_config.json'
        child_config.write_text('tone1', encoding='utf8')
        parent_args = dict(self.args, output_paths=['datasets/outputs/parent'])
        child_args = dict(self.args, operation='child', input_paths=['datasets/raw/camera/a.png'],
                          procedural_inputs=['fixtures/child_config.json'],
                          output_paths=['datasets/outputs/parent/result.txt'])
        with self.assertRaisesRegex(ValueError, 'parent input closure'):
            with guard.generation_guard(**parent_args) as parent:
                with mock.patch.dict(os.environ, parent.child_env(), clear=True):
                    with guard.generation_guard(**child_args):
                        self.fail('undeclared child input must never reach the operation')

        parent_args['procedural_inputs'] = ['fixtures/production_config.json', 'fixtures/child_config.json']
        result = self.root / 'datasets/outputs/parent/result.txt'
        with guard.generation_guard(**parent_args) as parent:
            with mock.patch.dict(os.environ, parent.child_env(), clear=True):
                with guard.generation_guard(**child_args):
                    result.parent.mkdir(parents=True)
                    result.write_text(child_config.read_text(encoding='utf8'), encoding='utf8')
        with guard.generation_guard(**parent_args) as parent:
            self.assertTrue(parent.reused)
        child_config.write_text('tone2', encoding='utf8')
        with guard.generation_guard(**parent_args) as parent:
            self.assertFalse(parent.reused)
            with mock.patch.dict(os.environ, parent.child_env(), clear=True):
                with guard.generation_guard(**child_args):
                    result.write_text(child_config.read_text(encoding='utf8'), encoding='utf8')
        self.assertEqual(result.read_text(encoding='utf8'), 'tone2')

    def test_child_admission_failure_cannot_be_hidden_by_parent(self):
        undeclared = self.root / 'fixtures/undeclared_child.json'
        undeclared.write_text('{"tone":1}', encoding='utf8')
        parent_args = dict(self.args, output_paths=['datasets/outputs/parent'])
        child_args = dict(self.args, operation='child', input_paths=['datasets/raw/camera/a.png'],
                          procedural_inputs=['fixtures/undeclared_child.json'],
                          output_paths=['datasets/outputs/parent/render.txt'])
        with self.assertRaisesRegex(RuntimeError, 'child production stage failed'):
            with guard.generation_guard(**parent_args) as parent:
                started = self.root / 'datasets/outputs/parent/started.txt'
                started.parent.mkdir(parents=True)
                started.write_text('started', encoding='utf8')
                with mock.patch.dict(os.environ, parent.child_env(), clear=True):
                    with self.assertRaisesRegex(ValueError, 'parent input closure'):
                        with guard.generation_guard(**child_args):
                            self.fail('child admission should have failed')
        db = sqlite3.connect(str(fixtures.core.INDEX / 'production_state.sqlite3'))
        try:
            receipt = db.execute('select receipt_json from production_operations where operation=?',
                                 ('fixture-generation',)).fetchone()
            self.assertEqual(receipt, (None,))
        finally:
            db.close()


if __name__ == '__main__':
    unittest.main()
