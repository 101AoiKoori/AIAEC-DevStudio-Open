"""Exercise the actual planned guard with an isolated person production entry."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'lab/scripts/media/tests'))
import test_media_index_sqlite as fixtures

from datasets.scripts.person import compose_planned_ground_person as person
from datasets.scripts.person import package_person_dataset as direct_package
from datasets.scripts.common import recoverable_publish
from lab.training_common import production_guard


class PersonSampleGuardIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CacheTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.image = self.fixture.root, self.fixture.image
        shard = self.root / self.fixture.parts['datasets/raw/camera']['shard']
        row = json.loads(shard.read_text(encoding='utf8'))
        row.update(bytes=self.image.stat().st_size, modified_ns=self.image.stat().st_mtime_ns)
        shard.write_text(json.dumps(row) + '\n', encoding='utf8')
        self.assertEqual(self.fixture.call(['sync'])[0], 0)
        self.output = self.root / 'datasets/samples/person'
        self.output.mkdir(parents=True)
        (self.output / '.gitkeep').touch()
        self.code = self.root / 'fixtures/generator.py'
        self.code.parent.mkdir(parents=True, exist_ok=True)
        self.code.write_text('fixture', encoding='utf8')
        self.plan_path = self.root / 'fixtures/plan.json'
        state = {'id': 'one', 'failure_event': 'miss', 'min_count': 1, 'max_count': 1,
                 'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
                 'conditions': [], 'background': {'image': self.image.relative_to(self.root).as_posix()},
                 'target': {'class': 'person'}, 'labels': {'format': 'yolo'},
                 'augmentation': {'tone': 'normal'}, 'sources': ['camera', 'person'],
                 'parameters': {'yaw': 0}}
        self.plan = {'schema_version': 1, 'model_key': 'person', 'target_split': 'train',
                     'purpose': 'fixture', 'budget': 1, 'uncertainty_weight': 0,
                     'independence_basis': 'fixture', 'constraints': {'size': [640, 480]}, 'strata': [state]}
        self.plan_path.write_text(json.dumps(self.plan), encoding='utf8')
        actual_guard = production_guard.generation_guard

        def guard(operation, _inputs, outputs, parameters, **kwargs):
            return actual_guard(operation, [self.image, self.plan_path], outputs, parameters,
                                code_paths=[self.code], procedural_inputs=[self.plan_path], root=self.root,
                                production_plan=kwargs['production_plan'],
                                sample_output_paths=kwargs['sample_output_paths'],
                                auxiliary_output_paths=kwargs['auxiliary_output_paths'],
                                replace_outputs=kwargs['replace_outputs'],
                                replace_verifier=kwargs['replace_verifier'])

        working = types.SimpleNamespace(media=set(), evidence=set(), aliases={}, actual=lambda path: str(path))

        def generate(_output, number, _state, _item, _role, _reason, **kwargs):
            image = kwargs['image_dir'] / f'person_{number:06d}.png'
            label = kwargs['label_dir'] / f'person_{number:06d}.txt'
            image.write_bytes(b'person')
            label.write_text('label', encoding='utf8')
            return {'image': image.relative_to(self.root).as_posix(),
                    'label': label.relative_to(self.root).as_posix()}

        patches = [mock.patch.object(person, 'ROOT', self.root),
                   mock.patch.object(person, 'workspace_file', return_value=self.plan_path),
                   mock.patch.object(person, 'sample_dir', return_value=self.output),
                   mock.patch.object(person, 'split_dir', return_value=self.root / 'datasets/exports/person/train'),
                   mock.patch.object(person, 'generation_guard', side_effect=guard),
                   mock.patch.object(person, 'verify_current_backgrounds'),
                   mock.patch.object(person, 'scene_sources', return_value=([], [])),
                   mock.patch.object(person, 'WorkingInputs', return_value=working),
                   mock.patch.object(person, 'queue_inputs', return_value=[]),
                   mock.patch.object(person, 'build_queue', return_value={}),
                   mock.patch.object(person, 'source_role_inputs', return_value=(set(), set(), set())),
                   mock.patch.object(person, 'source_role', return_value=('fixture', 'fixture')),
                   mock.patch.object(person, 'prepare', return_value={'image': self.image}),
                   mock.patch.object(person, '_produce_one', side_effect=generate)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def run_person(self, replace=False):
        return person.produce('fixtures/plan.json', 'datasets/samples/person', replace=replace)

    def test_replace_preserves_manual_image_and_receipt(self):
        self.run_person()
        manual = self.output / 'images/manual.png'
        manual.write_bytes(b'human')
        self.run_person(replace=True)
        self.assertEqual(manual.read_bytes(), b'human')
        self.assertTrue((self.output / '.gitkeep').exists())
        with sqlite3.connect(fixtures.core.INDEX / 'production_state.sqlite3') as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM production_receipt_history').fetchone()[0], 2)

    def test_unowned_name_collision_preserves_old_output(self):
        self.run_person()
        original = recoverable_publish.signature(self.output)
        with mock.patch.object(person.sample_replace, 'owned_files', return_value={'manifest.json'}):
            with self.assertRaisesRegex(ValueError, 'overwrite an unowned'):
                self.run_person(replace=True)
        self.assertEqual(recoverable_publish.signature(self.output), original)

    def test_unified_split_rewrites_direct_record_paths(self):
        output = self.root / 'datasets/exports/person/train'
        self.plan.update(direct_capture_ids=['one'])
        self.plan['constraints']['background'] = person.TASK_COMPLETE
        self.plan_path.write_text(json.dumps(self.plan), encoding='utf8')
        selection = [(None, {'id': 'one'}, None, self.image, None, None, '')]
        def write_direct(stage, split, _selection):
            image = stage / 'images/direct_real/person_real_000001.png'
            label = stage / 'labels/direct_real/person_real_000001.txt'
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b'direct')
            label.write_text('label', encoding='utf8')
            return [{'image': image.relative_to(self.root).as_posix(),
                     'label': label.relative_to(self.root).as_posix(), 'type': 'direct_real'}]
        with mock.patch.object(direct_package, 'prepare_direct', return_value=selection), \
             mock.patch.object(direct_package, 'direct_dependencies', return_value=(set(), set(), ())), \
             mock.patch.object(direct_package, 'write_direct_records', side_effect=write_direct), \
             mock.patch.object(direct_package, 'split_coverage', return_value={'count': 2}):
            first = person.produce('fixtures/plan.json', 'datasets/exports/person/train')
            self.assertEqual(first['records'][1]['image'],
                             'datasets/exports/person/train/images/direct_real/person_real_000001.png')
            self.assertTrue((output / 'images/direct_real/person_real_000001.png').is_file())
            manual = output / 'images/synthetic/manual.png'
            manual.write_bytes(b'human')
            second = person.produce('fixtures/plan.json', 'datasets/exports/person/train', replace=True)
        self.assertEqual(second['coverage'], {'count': 2})
        self.assertEqual(manual.read_bytes(), b'human')
        self.assertFalse(any('.stage.' in record['image'] for record in second['records']))

    def test_failure_after_publish_restores_previous_output(self):
        self.run_person()
        original = recoverable_publish.signature(self.output)
        publish = recoverable_publish.publish
        def fail_after(*args):
            publish(*args)
            raise RuntimeError('fixture interrupted')
        with mock.patch.object(recoverable_publish, 'publish', side_effect=fail_after):
            with self.assertRaisesRegex(RuntimeError, 'fixture interrupted'):
                self.run_person(replace=True)
        self.assertEqual(recoverable_publish.signature(self.output), original)

    def test_killed_publisher_recovers_with_real_receipt(self):
        self.run_person()
        manual = self.output / 'images/manual.png'
        manual.write_bytes(b'human')
        script = r'''
import os, pathlib, sys, types
from unittest import mock
from datasets.scripts.person import compose_planned_ground_person as p
from datasets.scripts.common import recoverable_publish as publisher
from lab.training_common import production_guard as guard
r = pathlib.Path(sys.argv[1]); image = r / 'datasets/raw/camera/a.png'
out = r / 'datasets/samples/person'; plan = r / 'fixtures/plan.json'; code = r / 'fixtures/generator.py'
guard._cache.core.ROOT = r
guard._cache.core.INDEX = r / 'datasets/annotated/media_index'
guard._cache.DATABASE = guard._cache.core.INDEX / 'query_cache.sqlite3'
guard._cache.core.load_lifecycle_rules()
actual_guard = guard.generation_guard
def wrapped(operation, _inputs, outputs, parameters, **kw):
    return actual_guard(operation, [image, plan], outputs, parameters, code_paths=[code],
        procedural_inputs=[plan], root=r, production_plan=kw['production_plan'],
        sample_output_paths=kw['sample_output_paths'], auxiliary_output_paths=kw['auxiliary_output_paths'],
        replace_outputs=kw['replace_outputs'], replace_verifier=kw['replace_verifier'])
def generate(_output, number, _state, _item, _role, _reason, **kw):
    img = kw['image_dir'] / f'person_{number:06d}.png'
    label = kw['label_dir'] / f'person_{number:06d}.txt'
    img.write_bytes(b'person'); label.write_text('label', encoding='utf8')
    return {'image': img.relative_to(r).as_posix(), 'label': label.relative_to(r).as_posix()}
working = types.SimpleNamespace(media=set(), evidence=set(), aliases={}, actual=lambda value: str(value))
publish = publisher.publish
def crash(*args):
    publish(*args)
    os._exit(43)
with mock.patch.object(p, 'ROOT', r), mock.patch.object(p, 'workspace_file', return_value=plan), \
     mock.patch.object(p, 'sample_dir', return_value=out), \
     mock.patch.object(p, 'split_dir', return_value=r / 'datasets/exports/person/train'), \
     mock.patch.object(p, 'generation_guard', side_effect=wrapped), \
     mock.patch.object(p, 'verify_current_backgrounds'), mock.patch.object(p, 'scene_sources', return_value=([], [])), \
     mock.patch.object(p, 'WorkingInputs', return_value=working), mock.patch.object(p, 'queue_inputs', return_value=[]), \
     mock.patch.object(p, 'build_queue', return_value={}), \
     mock.patch.object(p, 'source_role_inputs', return_value=(set(), set(), set())), \
     mock.patch.object(p, 'source_role', return_value=('fixture', 'fixture')), \
     mock.patch.object(p, 'prepare', return_value={'image': image}), \
     mock.patch.object(p, '_produce_one', side_effect=generate), \
     mock.patch.object(publisher, 'publish', side_effect=crash):
    p.produce('fixtures/plan.json', 'datasets/samples/person', replace=True)
'''
        child = subprocess.run([sys.executable, '-c', script, str(self.root)], cwd=str(ROOT),
                               capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 43, child.stderr)
        self.assertTrue(recoverable_publish._metadata(self.output)[1].is_file())
        self.run_person(replace=True)
        self.assertEqual(manual.read_bytes(), b'human')
        self.assertFalse(recoverable_publish._metadata(self.output)[1].exists())
        with sqlite3.connect(fixtures.core.INDEX / 'production_state.sqlite3') as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM production_receipt_history').fetchone()[0], 2)


if __name__ == '__main__':
    unittest.main()
