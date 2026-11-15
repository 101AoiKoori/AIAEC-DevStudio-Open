"""Check that prior assumptions affect actual bounded scene and view quotas."""
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(SCRIPTS)]
from lab.training_common.bayesian_allocation import allocate, plan_generation
from plan_meter_elastic_v4 import make_plan, make_arena_plan, parent_request, ARENA, STATION
from meter_station_plan import validate_arena_declaration, validate_station_declaration

class MeterElasticTests(unittest.TestCase):
    def test_parent_prior_changes_quotas(self):
        def counts(request):
            return {s['id']: s['planned_count'] for s in allocate(request)['strata']}
        self.assertEqual(counts(parent_request()), {'station': 7, 'arena': 5})
        self.assertEqual(counts(parent_request((2, 3), (3, 2))), {'station': 5, 'arena': 7})

    def test_view_prior_changes_selected_states_and_keeps_both_sides(self):
        def counts(plan):
            result = plan_generation(plan)
            return [sum(s['planned_count'] for s in result['strata'] if s['id'].startswith(side))
                    for side in ('left', 'right')]
        self.assertEqual(counts(make_plan()), [5, 1])
        self.assertEqual(counts(make_plan((2, 4), (4, 2))), [1, 5])

    def test_both_entry_contracts_accept_flexible_states(self):
        validate_station_declaration(make_plan(budget=7), count=7, source=STATION, allow_occlusion=False)
        validate_arena_declaration(make_arena_plan(budget=5), count=5, sources=[ARENA, STATION], foreground=False)

    def test_arena_lens_is_bounded_and_legacy_plan_is_accepted(self):
        legacy = make_arena_plan(budget=5)
        for state in legacy['strata']:
            del state['parameters']['camera_lens_mm']
        validate_arena_declaration(legacy, count=5, sources=[ARENA, STATION], foreground=False)
        invalid = make_arena_plan(budget=5)
        invalid['strata'][0]['parameters']['camera_lens_mm'] = 55
        with self.assertRaises(ValueError):
            validate_arena_declaration(invalid, count=5, sources=[ARENA, STATION], foreground=False)

if __name__ == '__main__':
    unittest.main()
