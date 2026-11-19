"""Exercise a person production entry with an isolated indexed workspace."""
from contextlib import redirect_stderr, redirect_stdout
import io
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/scripts/media/tests'), str(ROOT / 'lab/features/person/training/scripts')]
from lab.training_common import production_guard
import test_media_index_sqlite as fixtures
import synthesize_real_ground as entry


class PersonProductionGuardTest(unittest.TestCase):
    def test_material_node_is_created_and_connected_when_defaults_are_missing(self):
        fake_bpy, fake_math = types.ModuleType('bpy'), types.ModuleType('mathutils')
        fake_math.Vector = object
        path = ROOT / 'datasets/scripts/person/render_person_arena_v4.py'
        spec = importlib.util.spec_from_file_location('person_blender_node_fixture', path)
        renderer = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {'bpy': fake_bpy, 'mathutils': fake_math}):
            spec.loader.exec_module(renderer)
        nodes, links = mock.MagicMock(), mock.Mock()
        shader = types.SimpleNamespace(outputs={'BSDF': 'shader-output'})
        output = types.SimpleNamespace(inputs={'Surface': 'material-surface'})
        nodes.__iter__.return_value = iter(())
        nodes.new.side_effect = [shader, output]
        material = types.SimpleNamespace(node_tree=types.SimpleNamespace(nodes=nodes, links=links))
        self.assertIs(renderer.principled_node(material), shader)
        self.assertEqual(nodes.new.call_args_list,
                         [mock.call('ShaderNodeBsdfPrincipled'), mock.call('ShaderNodeOutputMaterial')])
        links.new.assert_called_once_with('shader-output', 'material-surface')
        localized_shader = types.SimpleNamespace(bl_idname='ShaderNodeBsdfPrincipled', name='原理化 BSDF')
        nodes.reset_mock()
        nodes.__iter__.return_value = iter([localized_shader])
        self.assertIs(renderer.principled_node(material), localized_shader)
        nodes.new.assert_not_called()

    def setUp(self):
        self.fixture = fixtures.CacheTests('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        shard = self.root / 'datasets/annotated/media_index/shards/raw.jsonl'
        item = json.loads(shard.read_text(encoding='utf8'))
        item['bytes'] = self.fixture.image.stat().st_size
        item['modified_ns'] = self.fixture.image.stat().st_mtime_ns
        shard.write_text(json.dumps(item, ensure_ascii=False) + '\n', encoding='utf8')
        (self.root / 'fixtures').mkdir(exist_ok=True)
        self.code = self.root / 'fixtures/person_generator.py'
        self.code.write_text('# fixture source\n', encoding='utf8')
        self.output = self.root / 'datasets/exports/person/fixture'
        patch = mock.patch.object(production_guard, '_index', return_value=fixtures.cache)
        patch.start()
        self.addCleanup(patch.stop)

    def guard(self, operation, inputs, outputs, parameters, **kwargs):
        return production_guard.generation_guard(operation, inputs, outputs, parameters,
                                                 code_paths=[self.code], root=self.root)

    def invoke(self):
        return entry.generate_v4_pilot(self.output, 1, 42, self.fixture.image,
                                       self.fixture.image, None, False)

    def test_legacy_v4_public_entry_is_disabled_before_guard_or_generation(self):
        with mock.patch.object(entry, 'ROOT', self.root), \
             mock.patch.object(entry, 'production_inputs') as inputs, \
             mock.patch.object(entry, 'generation_guard') as guard, \
             mock.patch.object(entry, '_generate_v4_pilot') as produce:
            with self.assertRaisesRegex(ValueError, 'not connected to a production plan'):
                self.invoke()
        inputs.assert_not_called()
        guard.assert_not_called()
        produce.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_indexed_media_without_claims_stays_unclaimed(self):
        evidence = self.root / 'datasets/annotated/media_index/shards/derived.evidence.jsonl'
        evidence.write_text('', encoding='utf8')
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(self.fixture.call(['sync'])[0], 0)
        with production_guard.generation_guard('person.unclaimed.fixture', [self.fixture.image],
                                               [self.output], {}, code_paths=[self.code], root=self.root) as run:
            path = 'datasets/raw/camera/a.png'
            self.assertEqual(run.source_refs[path], [])
            self.assertEqual(run.media_observations[path]['path'], path)
            self.output.mkdir(parents=True)
            (self.output / 'done').write_text('fixture', encoding='utf8')

    def test_blender_entry_guard_precedes_scene_changes(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(self.fixture.call(['sync'])[0], 0)
        textures = self.root / 'datasets/annotated/person/textures'
        textures.mkdir(parents=True)
        relative = self.fixture.image.relative_to(self.root).as_posix()
        (textures / 'manifest.json').write_text(json.dumps({'records': [
            {'artwork': relative, 'image': relative}]}), encoding='utf8')
        fake_bpy = types.ModuleType('bpy')
        fake_bpy.data = types.SimpleNamespace(images=[], filepath=str(self.fixture.image))
        fake_math = types.ModuleType('mathutils')
        fake_math.Vector = object
        path = ROOT / 'datasets/scripts/person/render_person_arena_v4.py'
        spec = importlib.util.spec_from_file_location('person_blender_guard_fixture', path)
        renderer = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {'bpy': fake_bpy, 'mathutils': fake_math}):
            spec.loader.exec_module(renderer)
        plan = {'schema_version': 1, 'model_key': 'person', 'target_split': 'train',
                'purpose': 'contract test', 'budget': 1, 'uncertainty_weight': 0,
                'independence_basis': 'one fixture',
                'constraints': {'white_back': 'no_person_boxes', 'frontal_visibility': 'minimum_0.8',
                                'geometry': 'visually_fitted'},
                'strata': [{'id': 'scene', 'failure_event': 'miss', 'min_count': 1, 'max_count': 1,
                            'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
                            'conditions': [], 'background': {'scene': 'original', 'camera_preset': 'person-near'},
                            'target': {'front_count': 3, 'noncommunity_front': 1,
                                       'white_back_class': 'community', 'occlusion': 'none'},
                            'labels': {'person': 'visible_yolo', 'white_back': 'negative_no_boxes'},
                            'augmentation': {'exposure_ev': 0},
                            'parameters': {'resolution': [640, 480], 'camera_jitter_m': 0.0,
                                           'position_jitter_m': 0.0, 'render_device': 'optix'},
                            'sources': ['person.whiteborder_textures', 'arena.original']}]}
        plan_path = self.root / 'render_plan.json'
        plan_path.write_text(json.dumps(plan), encoding='utf8')
        parsed = types.SimpleNamespace(scene='original', camera_preset='person-near',
                                       camera_jitter=0.0, position_jitter=0.0, render_device='optix')
        renderer.validate_production_plan(plan, parsed)
        state = plan['strata'][0]
        with self.assertRaises(ValueError):
            renderer.validate_production_plan({**plan, 'strata': [{**state,
                'target': {**state['target'], 'white_back_class': 'noncommunity', 'noncommunity_front': 2}}]}, parsed)
        with self.assertRaises(ValueError):
            renderer.validate_production_plan({**plan, 'strata': [{**state,
                'augmentation': {'exposure_ev': -1}}]}, parsed)
        with self.assertRaises(ValueError):
            renderer.validate_production_plan({**plan, 'strata': [{**state, 'conditions': ['rain']}]}, parsed)
        render_output = self.root / 'datasets/samples/person'
        def produce(args, run):
            (args.output/'images').mkdir(parents=True)
            image = args.output/'images/sample.png'
            image.write_bytes(b'fixture-image')
            state = run.plan['strata'][0]
            run.record_sample(state['id'], output=image, sources=state['sources'],
                              parameters=state['parameters'], actual={'fixture': True})
            (args.output / 'done').write_text('fixture', encoding='utf8')
        def render_guard(operation, inputs, outputs, parameters, **kwargs):
            self.assertIn(self.fixture.image, inputs)
            return production_guard.generation_guard(operation, [self.fixture.image, plan_path], outputs, parameters,
                code_paths=[self.code], procedural_inputs=[plan_path], root=self.root,
                production_plan=kwargs['production_plan'], sample_output_paths=kwargs['sample_output_paths'],
                auxiliary_output_paths=kwargs['auxiliary_output_paths'])
        argv = ['blender', '--', '--output', str(render_output), '--production-plan', str(plan_path),
                '--camera-preset', 'person-near']
        with mock.patch.object(renderer, 'ROOT', self.root), \
             mock.patch.object(renderer, 'TEXTURES', textures), \
             mock.patch.object(renderer, 'sample_dir', return_value=render_output), \
             mock.patch.object(renderer, 'split_dir', return_value=self.root/'datasets/exports/person/train'), \
             mock.patch.object(renderer, 'SCENE_SOURCE', self.fixture.image), \
             mock.patch.object(renderer, 'generation_guard', side_effect=render_guard), \
             mock.patch.object(renderer, 'activate_scene') as activate, \
             mock.patch.object(renderer, '_produce', side_effect=produce) as wrapped, \
             mock.patch.object(sys, 'argv', argv):
            renderer.main()
            before = (render_output / 'done').stat().st_mtime_ns
            renderer.main()
        self.assertEqual(wrapped.call_count, 1)
        activate.assert_called_once_with(renderer.DETAILED)
        self.assertEqual((render_output / 'done').stat().st_mtime_ns, before)


if __name__ == '__main__':
    unittest.main()
