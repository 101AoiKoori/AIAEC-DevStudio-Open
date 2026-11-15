import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'render_sign_arena_backgrounds.py'
spec = importlib.util.spec_from_file_location('render_sign_arena_backgrounds', SCRIPT)
module = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {'bpy': mock.Mock()}):
    spec.loader.exec_module(module)


class ArenaPlanTests(unittest.TestCase):
    def setUp(self):
        self.sources = [module.SCENE]
        self.state = {'id': 'view1', 'background': {'scene': module.SCENE},
                      'target': {'kind': 'background', 'sign_absence': 'unknown'},
                      'labels': {'kind': 'none'}, 'augmentation': {'kind': 'none'},
                      'conditions': ['background_only'], 'max_count': 1,
                      'parameters': {'view': list(module.VIEWS[0])}, 'sources': self.sources}
        self.plan = {'model_key': 'sign', 'constraints': {'image_size': [640, 480],
                     'cycles_samples': 16, 'device': 'OPTIX'}, 'strata': [self.state]}

    def test_supported_view_and_explicit_unknown_are_accepted(self):
        module.validate_declaration(self.plan, self.sources)

    def test_unsupported_state_and_duplicate_budget_fail(self):
        invalid = [
            {'target': {'kind': 'background', 'sign_absence': 'confirmed'}},
            {'augmentation': {'kind': 'brightness'}},
            {'parameters': {'view': [0, 0, 0]}},
            {'max_count': 2},
            {'sources': ['different']},
        ]
        for replacement in invalid:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                module.validate_declaration({**self.plan, 'strata': [{**self.state, **replacement}]}, self.sources)
        with self.assertRaisesRegex(ValueError, 'repeated'):
            module.validate_declaration({**self.plan, 'strata': [self.state, {**self.state, 'id': 'other'}]}, self.sources)

    def test_renderer_uses_allocated_view_and_records_output(self):
        class Vec:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z
            def __iter__(self):
                return iter((self.x, self.y, self.z))

        camera = SimpleNamespace(location=Vec(0, 0, 1), rotation_euler=Vec(0, 0, 0),
                                 data=SimpleNamespace(lens=10.0))
        scene = SimpleNamespace(camera=None, cycles=SimpleNamespace(),
                                render=SimpleNamespace(image_settings=SimpleNamespace()))
        device = SimpleNamespace(type='OPTIX', use=False)
        prefs = SimpleNamespace(compute_device_type=None, devices=[device], get_devices=lambda: None)
        fake = SimpleNamespace(data=SimpleNamespace(objects={'Vehicle camera': camera}),
                               context=SimpleNamespace(scene=scene, preferences=SimpleNamespace(
                                   addons={'cycles': SimpleNamespace(preferences=prefs)})),
                               ops=SimpleNamespace(render=SimpleNamespace(render=mock.Mock())))
        run = mock.Mock()
        planned = {**self.plan, 'target_split': 'train',
                   'strata': [{**self.state, 'planned_count': 1}]}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(module, 'OUTPUT_ROOT', Path(tmp)), \
                mock.patch.object(module, 'bpy', fake):
            module._main(Path(tmp) / 'batch', planned, run)
            self.assertEqual([camera.location.x, camera.location.y, camera.rotation_euler.z], list(module.VIEWS[0]))
            self.assertEqual(run.record_sample.call_count, 1)
            self.assertEqual(run.record_sample.call_args.kwargs['actual']['view'], list(module.VIEWS[0]))


if __name__ == '__main__':
    unittest.main()
