"""Current input-to-output contracts, without bpy, image output or receipts."""
import copy
import json
import unittest

from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common.bayesian import plan_production
from datasets.scripts.meter import meter_station_plan as render
from datasets.scripts.meter import compose_meter_real_background_v4 as composite


class CurrentMeterPlans(unittest.TestCase):
    def setUp(self):
        self.source = load_task_config('meter')[1]
        self.render = json.loads(source_path('meter', 'render_plan').read_text(encoding='utf8'))
        self.composite = json.loads(source_path('meter', 'production_plan').read_text(encoding='utf8'))

    def test_render_plan_matches_arena_contract_and_twelve_unique_states(self):
        render.validate_arena_declaration(self.render, count=12,
            sources=[self.source['arena_source'], self.source['station_source']], foreground=True)
        allocated = render.allocated_arena_states(plan_production(self.render), count=12)
        self.assertEqual(len(allocated), 12)
        self.assertEqual(self.render['execution']['device'], 'OPTIX')
        self.assertEqual(self.render['execution']['endpoints'], self.source['endpoint_config'])
        self.assertEqual(self.render['execution']['output'], self.source['real_composite']['foreground_batch'])

    def test_composite_plan_uses_three_reviewed_backgrounds_and_declared_future_rgba(self):
        specs = composite.configured_backgrounds(self.source['real_composite'])
        backgrounds = [composite.background_spec(row, candidate_reuse=True) for row in specs]
        composite.validate_lineages(backgrounds, [])
        batch = composite.ROOT / self.source['real_composite']['foreground_batch']
        future = [(i, {'id': f'arena_meter_{i:05d}'}, batch/'images'/f'arena_meter_{i:05d}.png',
                   batch/'foreground'/f'arena_meter_{i:05d}.png', None, None, None) for i in range(12)]
        composite.validate_plan(self.composite, 36, backgrounds, future)
        self.assertEqual(plan_production(self.composite)['planned_total'], 36)
        self.assertNotEqual(self.composite['execution']['output'], self.composite['execution']['foreground_batch'])
        self.assertEqual(len({s['parameters']['background_image'] for s in self.composite['strata']}), 3)
        self.assertEqual(len({s['parameters']['foreground_id'] for s in self.composite['strata']}), 12)
        invalid = copy.deepcopy(self.composite)
        invalid['strata'][0]['parameters']['foreground_id'] = 'not_rendered_by_current_plan'
        with self.assertRaises(ValueError):
            composite.validate_plan(invalid, 36, backgrounds, future)


if __name__ == '__main__':
    unittest.main()
