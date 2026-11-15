"""Current complete-detection and replacement contracts without sample output."""
import copy
import json
import unittest

from datasets.scripts.common.config import source_path
from datasets.scripts.common.bayesian import plan_production
from datasets.scripts.building_name_paper import compose_rooftop_papers as real
from datasets.scripts.building_name_paper import render_blender_names as blender


class CurrentNamePlans(unittest.TestCase):
    def setUp(self):
        self.request = json.loads(source_path('building_name_paper', 'production_plan').read_text(encoding='utf8'))

    def test_complete_detection_sources_and_current_budget(self):
        real.validate_request(self.request)
        real.current_detection_sources(self.request)
        plan = plan_production(self.request)
        self.assertEqual(plan['planned_total'], 45)
        direct = [s for s in plan['strata'] if not s['target']['papers']]
        self.assertEqual(len(direct), 5)
        positive = [s for s in direct if s['background']['candidates'][0]['existing_papers']]
        self.assertEqual(len(positive), 2)
        self.assertEqual(len(direct) - len(positive), 3)
        self.assertTrue(all(c['coverage'] == 'complete_candidate' for s in plan['strata']
                            for c in s['background']['candidates']))

    def test_replacement_cannot_cover_an_unlabelled_region(self):
        request = copy.deepcopy(self.request)
        state = next(s for s in request['strata'] if s['target']['papers'])
        state['background']['candidates'][0]['regions'][0]['quad_xy'][0][0] -= 1
        with self.assertRaises(ValueError):
            real.validate_request(request)

    def test_unknown_source_boxes_do_not_become_training_labels(self):
        request = copy.deepcopy(self.request)
        request['strata'][0]['background']['candidates'][0]['coverage'] = 'unknown'
        with self.assertRaises(ValueError):
            real.validate_request(request)

    def test_blender_plan_is_separate_and_keeps_three_name_branches(self):
        request = json.loads(source_path('building_name_paper', 'blender_plan').read_text(encoding='utf8'))
        blender.validate_plan(request)
        self.assertEqual(plan_production(request)['planned_total'], 900)
        self.assertEqual({len(s['target']['texts']) for s in request['strata']}, {0, 1, 2})
        self.assertNotEqual(request['execution']['output'], self.request['execution']['output'])


if __name__ == '__main__':
    unittest.main()
