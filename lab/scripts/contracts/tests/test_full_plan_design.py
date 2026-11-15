import copy
import unittest
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import full_plan_design

ROOT = Path(__file__).resolve().parents[4]


class FullPlanDesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.allocation = full_plan_design.load_allocation()
        cls.targets = full_plan_design.load_targets()
        cls.validated = full_plan_design.validate(cls.allocation, cls.targets, ROOT)
        cls.body = full_plan_design.render_design(cls.allocation, cls.validated, cls.targets, ROOT)

    def test_all_tasks_validated(self):
        self.assertEqual(set(self.validated), set(full_plan_design.TASK_ORDER))
        for task in full_plan_design.TASK_ORDER:
            self.assertEqual(self.validated[task]['total'],
                             self.validated[task]['total'])

    def test_person_groups_sum_to_total(self):
        person = self.allocation['tasks']['person']
        self.assertEqual(sum(group['quota'] for group in person['groups']), 3000)
        occlusion = sum(group['quota'] for group in person['groups']
                        if group['axes'].get('vehicle_occlusion') in {'upright', 'fallen'})
        self.assertEqual(occlusion, 750)

    def test_sticker_grid_product(self):
        grid = self.allocation['tasks']['waste_sticker']['grid']
        self.assertEqual(grid['class_count'] * len(grid['colors']) * grid['per_class_color'], 7200)

    def test_meter_needle_groups_sum(self):
        meter = self.allocation['tasks']['meter']
        self.assertEqual(sum(group['quota'] for group in meter['groups']), 3500)
        self.assertEqual(len(meter['groups']), 12)

    def test_bad_source_mix_is_rejected(self):
        data = copy.deepcopy(self.allocation)
        data['tasks']['person']['source_mix']['blender'] = 0.60
        with self.assertRaises(ValueError):
            full_plan_design.validate(data, self.targets, ROOT)

    def test_bad_total_is_rejected(self):
        data = copy.deepcopy(self.allocation)
        data['tasks']['person']['total'] = 2999
        with self.assertRaises(ValueError):
            full_plan_design.validate(data, self.targets, ROOT)

    def test_target_outside_task_targets_is_rejected(self):
        data = copy.deepcopy(self.allocation)
        data['tasks']['person']['groups'][0]['target'] = 'vehicle'
        with self.assertRaises(ValueError):
            full_plan_design.validate(data, self.targets, ROOT)

    def test_render_marks_draft_and_separates_plan_classes(self):
        self.assertIn('design_draft', self.body)
        self.assertIn('PlanDraft', self.body)
        self.assertIn('ProductionPlan', self.body)
        self.assertIn('### person', self.body)
        self.assertIn('### meter', self.body)

    def test_current_capacity_and_new_capacity(self):
        stats = full_plan_design._plan_stats(ROOT, 'person', self.targets)
        self.assertEqual(stats['capacity'], 54)
        self.assertEqual(3000 - stats['capacity'], 2946)

    def test_generated_doc_is_current(self):
        doc = ROOT / 'docs/FULL_TRAIN_PLAN_DESIGN.md'
        self.assertTrue(doc.is_file())
        self.assertEqual(doc.read_text(encoding='utf-8'), self.body)


if __name__ == '__main__':
    unittest.main()
