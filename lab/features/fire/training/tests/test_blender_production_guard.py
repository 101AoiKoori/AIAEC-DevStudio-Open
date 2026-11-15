"""Verify Blender entrypoints defer scene mutation and writes to the guard."""
import ast
from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
WORKSPACE_TMP = Path(__file__).resolve().parents[5] / 'tmp'


def entry(name, bindings, function='main'):
    path = (SCRIPTS.parents[4] / 'datasets/scripts/fire' / name
            if name == 'render_blender_fire_dataset.py' else SCRIPTS / name)
    tree = ast.parse(path.read_text(encoding='utf8'), filename=str(path))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
    module = ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[]))
    env = {'Path': Path, '__file__': str(path), **bindings}
    exec(compile(module, str(path), 'exec'), env)
    return env[function]


class BlenderProductionGuardTests(unittest.TestCase):
    def setUp(self):
        WORKSPACE_TMP.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=WORKSPACE_TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / 'runs/fire_sticker/out'
        self.calls = []
        self.inputs = []
        self.completed = False
        self.fail = False

    @contextmanager
    def guard(self, operation, input_paths, output_paths, parameters, **kwargs):
        self.inputs.append((operation, input_paths, output_paths, parameters, kwargs))
        if self.fail:
            raise ValueError('SQLite cache stale')
        yield SimpleNamespace(reused=self.completed, source_refs={'fixture': []},
                              plan={'planned_total': 2, 'strata': []})
        self.completed = True

    def produce(self, *args):
        self.calls.append(args)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'result.txt').write_text('fixture', encoding='utf8')

    def assert_protected(self, main):
        main()
        stamp = (self.output / 'result.txt').stat().st_mtime_ns
        main()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual((self.output / 'result.txt').stat().st_mtime_ns, stamp)
        self.output.joinpath('result.txt').unlink()
        self.output.rmdir()
        self.fail = True
        with self.assertRaisesRegex(ValueError, 'stale'):
            main()
        self.assertFalse(self.output.exists())
        self.assertEqual(len(self.calls), 1)

    def test_build_guard_precedes_production(self):
        args = SimpleNamespace(output=self.output, font=self.root / 'font.ttf', seed=1, samples=4, rect_depth=.125)
        main = entry('build_blender_buildings.py', {'ROOT': self.root, 'parse_args': lambda: args,
                     'generation_guard': self.guard, '_produce': self.produce})
        self.assert_protected(main)
        self.assertIn(self.root / 'datasets/raw/fire/team_assets/image1.png', self.inputs[0][1])

    def test_renderer_guard_precedes_production(self):
        scene = self.root / 'source.blend'
        args = SimpleNamespace(output=self.output, vehicle_background_manifest=None, vehicle_pilot=False,
                               count=2, seed=1, save_preview_scene=False, production_plan='plan.json')
        main = entry('render_blender_fire_dataset.py', {'ROOT': self.root, 'SCRIPTS': self.root / 'scripts',
                     'parse_args': lambda: (args, scene), 'background_guard_inputs': lambda *_: (set(), set()),
                     'read_production_declaration': lambda *_: (self.root / 'plan.json', {'strata': []}),
                     'plan_production': lambda *_: {'planned_total': 2, 'strata': []},
                     'output_partition': lambda *_: 'train',
                     'configured_fire_artwork': lambda: self.root/'datasets/raw/fire/team_assets/image1.png',
                     'scene_external_media': lambda: set(), 'scene_external_libraries': lambda: set(),
                     'generation_guard': self.guard, '_produce': self.produce})
        self.assert_protected(main)
        self.assertIn(scene, self.inputs[0][4]['procedural_inputs'])

    def test_arena_guard_precedes_production(self):
        args = SimpleNamespace(count=2, seed=1, samples=4)
        master, buildings, font = (self.root / name for name in ('arena.blend', 'buildings.blend', 'font.ttf'))
        fire = SimpleNamespace(scene_external_media=lambda: set(), scene_external_libraries=lambda: set(),
                               __file__=str(self.root / 'fire.py'))
        main = entry('render_fire_arena_v4.py', {'ROOT': self.root, 'MASTER': master, 'BUILDINGS': buildings,
                     'FONT': font, 'fire': fire, 'parse_args': lambda: (args, self.output),
                     'generation_guard': self.guard, '_produce': self.produce})
        self.assert_protected(main)
        self.assertEqual(set(self.inputs[0][4]['procedural_inputs']), {master, buildings, font})

    def test_background_manifest_media_ancestry_is_declared(self):
        parse = entry('render_blender_fire_dataset.py', {'ROOT': self.root}, 'background_guard_inputs')
        manifest = self.root / 'backgrounds.json'
        backgrounds = [{'image': 'datasets/frame.png', 'parent_image': 'datasets/parent.jpg',
                        'source_video': 'datasets/video.mp4', 'review_evidence': 'datasets/review.json',
                        'placement': {'review_evidence': 'datasets/review.json'}}]
        media, procedural = parse(manifest, backgrounds)
        self.assertEqual(media, {self.root / 'datasets/frame.png', self.root / 'datasets/parent.jpg',
                                 self.root / 'datasets/video.mp4'})
        self.assertEqual(procedural, {manifest, self.root / 'datasets/review.json'})


if __name__ == '__main__':
    unittest.main()
