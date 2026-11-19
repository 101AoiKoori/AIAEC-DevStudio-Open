"""Meter source bindings and missing-input behavior; no production is invoked."""
import ast
from contextlib import ExitStack, redirect_stdout
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.meter import compose_meter_real_background_v4 as module


class MeterSourceContracts(unittest.TestCase):
    def test_station_endpoints_and_face_match_selected_model(self):
        import json
        endpoint = json.loads(source_path('meter', 'endpoint_config').read_text(encoding='utf8'))
        self.assertEqual((module.ROOT/endpoint['source_blend']).resolve(), source_path('meter', 'station_source'))
        self.assertEqual((module.ROOT/endpoint['source_face']).resolve(), source_path('meter', 'station_face'))

    def test_existing_backgrounds_resolve_to_one_recorded_parent_video(self):
        _, config = load_task_config('meter')
        values = module.configured_backgrounds(config['real_composite'])
        self.assertEqual(len(values), 3)
        self.assertEqual(len({row[1] for row in values}), 1)
        rows = [module.background_spec(row, candidate_reuse=True) for row in values]
        module.validate_lineages(rows, [])
        self.assertEqual({r['row']['id'] for r in rows}, {'frame_003601', 'frame_009901', 'frame_010501'})

    def test_missing_foreground_batch_fails_before_guard(self):
        with tempfile.TemporaryDirectory(dir=module.ROOT / 'tmp') as temp:
            root = Path(temp).resolve()
            batch = root / 'datasets' / 'foreground'
            batch.mkdir(parents=True)
            output = root / 'datasets' / 'exports' / 'meter' / 'train'
            background = ['image', 'parent', 'manifest', '0', 'relation']
            defaults = {'foreground_batch': 'datasets/foreground', 'count': 1,
                        'production_plan': 'missing.json', 'background_review': ''}
            spec = {'image': root / 'image.png', 'parent': root / 'source.mp4'}
            args = ['entry', '--output', str(output), '--count', '1', '--production-plan', 'missing.json']
            with patch('sys.argv', args), patch.object(module, 'ROOT', root), \
                 patch.object(module, 'load_task_config', return_value=({}, {'real_composite': defaults})), \
                 patch.object(module, 'configured_backgrounds', return_value=[background]), \
                 patch.object(module, 'background_spec', return_value=spec), \
                 patch.object(module, 'validate_lineages'), \
                 patch.object(module, 'stored_backgrounds', return_value=(root / 'fixture.sqlite3', {})) as stored, \
                 patch.object(module, 'sample_dir', return_value=root / 'datasets/samples/meter'), \
                 patch.object(module, 'split_dir', return_value=output), \
                 patch.object(module, 'generation_guard') as guard:
                with self.assertRaises(FileNotFoundError) as error:
                    module.main()
                self.assertEqual(Path(error.exception.filename), batch / 'manifest.json')
                stored.assert_called_once()
                guard.assert_not_called()

    def test_real_guard_recovers_and_replaces_only_its_batch(self):
        with tempfile.TemporaryDirectory(dir=module.ROOT / 'tmp') as temp, ExitStack() as patches:
            root = Path(temp).resolve()
            output = root / 'datasets/exports/meter/train'
            output.mkdir(parents=True)
            (output / '.gitkeep').touch()
            fixtures = ('datasets/scripts/meter/compose_meter_real_background_v4.py',
                        'datasets/scripts/meter/config/dataset.yaml', 'datasets/scripts/meter/config/source.yaml',
                        'datasets/scripts/common/paths.py', 'datasets/scripts/common/frame_environment.py',
                        'datasets/scripts/common/recoverable_publish.py', 'datasets/scripts/common/working_inputs.py',
                        'datasets/foreground/manifest.json', 'datasets/foreground/samples.jsonl',
                        'datasets/foreground/annotation.json', 'datasets/foreground/pose.txt', 'fixture.sqlite3',
                        'review.json', 'source.mp4')
            for name in fixtures:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}', encoding='utf8')
            batch = root / 'datasets/foreground'
            Image.new('RGB', (640, 480), 'blue').save(root / 'background.png')
            overlay = Image.new('RGBA', (640, 480))
            overlay.paste((255, 0, 0, 255), (200, 200, 300, 300))
            overlay.save(batch / 'rgba.png')
            stem = 'arena_meter_00000'
            values = [0., .5, .5, .2, .2] + [.5, .5, 1.] * 8
            selected = [(0, {'id': stem}, batch / 'rgba.png', batch / 'rgba.png',
                         batch / 'pose.txt', batch / 'annotation.json', values)]
            prototype = json.loads(source_path('meter', 'legacy_production_plan').read_text(encoding='utf8'))
            state = prototype['strata'][0]
            state.update(sources=['background.png', 'source.mp4', 'datasets/foreground/rgba.png'],
                         parameters={'background_image': 'background.png', 'foreground_id': stem})
            plan = root / 'plan.json'
            plan.write_text(json.dumps(dict(prototype, budget=1, strata=[state])), encoding='utf8')
            background = ['background.png', 'source.mp4', 'review.json', '0', 'review_record']
            spec = {'image': root / 'background.png', 'parent': root / 'source.mp4', 'manifest': root / 'review.json',
                    'record_index': 0, 'relation': 'review_record', 'row': {}}
            defaults = {'foreground_batch': 'datasets/foreground', 'count': 1,
                        'production_plan': 'plan.json', 'background_review': ''}
            for key, value in {'ROOT': root, '__file__': str(root / fixtures[0]), 'WORKING_INPUTS_CODE':
                               root / 'datasets/scripts/common/working_inputs.py'}.items():
                patches.enter_context(patch.object(module, key, value))
            mocks = {
                'load_task_config': ({}, {'real_composite': defaults}), 'configured_backgrounds': [background],
                'background_spec': spec, 'validate_lineages': None, 'record_at': {}, 'split_dir': output,
                'stored_backgrounds': (root / 'fixture.sqlite3', {}),
                'source_rows': (batch / 'manifest.json', batch / 'samples.jsonl', {}, selected),
            }
            for key, result in mocks.items():
                patches.enter_context(patch.object(module, key, return_value=result))
            patches.enter_context(patch.object(module.recoverable_publish, '__file__',
                                               str(root / 'datasets/scripts/common/recoverable_publish.py')))
            working = patches.enter_context(patch.object(module, 'WorkingInputs')).return_value
            working.media, working.evidence = set(), set()
            working.path.side_effect = lambda path: path
            working.actual.side_effect = module.rel
            cli = ['entry', '--output', str(output)]
            with patch('sys.argv', cli), redirect_stdout(io.StringIO()) as captured:
                module.main()
                module.main()
                self.assertIn('METER_REAL_COMPOSITE_REUSED', captured.getvalue())
            (output / 'images/operator.png').write_bytes(b'operator-owned content')
            before = module.recoverable_publish.signature(output)
            publish = module.recoverable_publish.publish

            def fail_after_install(*args):
                publish(*args)
                raise RuntimeError('interrupted')

            with patch('sys.argv', cli + ['--replace']), redirect_stdout(io.StringIO()):
                with patch.object(module.recoverable_publish, 'publish', side_effect=fail_after_install):
                    with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                        module.main()
                self.assertEqual(module.recoverable_publish.signature(output), before)
                module.main()
            self.assertEqual((output / 'images/operator.png').read_bytes(), b'operator-owned content')
            self.assertTrue((output / '.gitkeep').exists())
            modified_label = output / 'labels/meter_real_composite_00000.txt'
            modified_label.write_text('operator-edited label', encoding='utf8')
            with patch('sys.argv', cli + ['--replace']), self.assertRaisesRegex(ValueError, 'differs from its production receipt'):
                module.main()
            self.assertEqual(modified_label.read_text(encoding='utf8'), 'operator-edited label')

    def test_blender_entries_declare_own_configuration_and_no_run_face_literal(self):
        for filename in ('render_meter_candidates.py', 'render_meter_arena_v4.py'):
            text = (module.ROOT/'datasets/scripts/meter'/filename).read_text(encoding='utf8')
            ast.parse(text)
            self.assertIn('datasets/scripts/meter/config/source.yaml', text)
            self.assertNotIn('runs/meter/v4_tick_face', text)


if __name__ == '__main__':
    unittest.main()
