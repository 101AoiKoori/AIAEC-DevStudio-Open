"""Meter source bindings and missing-input behavior; no production is invoked."""
import ast
import unittest
from unittest.mock import patch

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
        args = ['entry', '--output', 'datasets/exports/meter/train', '--count', '1', '--production-plan', 'missing.json']
        with patch('sys.argv', args), patch.object(module, 'generation_guard') as guard:
            with self.assertRaises(FileNotFoundError):
                module.main()
            guard.assert_not_called()

    def test_blender_entries_declare_own_configuration_and_no_run_face_literal(self):
        for filename in ('render_meter_candidates.py', 'render_meter_arena_v4.py'):
            text = (module.ROOT/'datasets/scripts/meter'/filename).read_text(encoding='utf8')
            ast.parse(text)
            self.assertIn('datasets/scripts/meter/config/source.yaml', text)
            self.assertNotIn('runs/meter/v4_tick_face', text)


if __name__ == '__main__':
    unittest.main()
