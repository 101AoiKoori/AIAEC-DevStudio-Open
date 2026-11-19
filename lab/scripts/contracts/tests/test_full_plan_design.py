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

    def test_sticker_quota_counts_images_and_visible_instances(self):
        groups = self.validated['waste_sticker']['groups']
        quotas = {group['axes']['visible_sticker_count']: group['quota'] for group in groups}
        self.assertEqual(quotas, {0: 180, 1: 405, 2: 405, 3: 405, 4: 405})
        self.assertEqual(sum(quotas.values()), 1800)
        self.assertEqual(sum(count * images for count, images in quotas.items()), 4050)
        self.assertEqual([group['target'] for group in groups], ['absent'] + ['present'] * 4)

    def test_meter_source_groups_and_split_design_sum(self):
        meter = self.allocation['tasks']['meter']
        self.assertEqual(sum(group['quota'] for group in meter['groups']), 6480)
        self.assertEqual(len(meter['groups']), 4)
        design=meter['split_design']
        self.assertEqual(sum(row['total'] for row in design['splits'].values()),8640)
        self.assertEqual({key:row['direct_real'] for key,row in design['splits'].items()},
                         {'train':45,'val':7,'test':2})
        bad=copy.deepcopy(self.allocation)
        bad['tasks']['meter']['split_design']['splits']['test']['empty_negative']+=1
        with self.assertRaises(ValueError):full_plan_design.validate(bad,self.targets,ROOT)

    def test_bad_source_mix_is_rejected(self):
        data = copy.deepcopy(self.allocation)
        data['tasks']['person']['source_mix']['blender'] = 0.60
        with self.assertRaises(ValueError):
            full_plan_design.validate(data, self.targets, ROOT)

    def test_source_mix_may_exceed_advisory_blender_share(self):
        data = copy.deepcopy(self.allocation)
        data['tasks']['person']['source_mix'] = {'direct_real': .2, 'real_composite': .2, 'blender': .6}
        validated = full_plan_design.validate(data, self.targets, ROOT)
        self.assertEqual(validated['person']['source_mix']['blender'], .6)

    def test_historical_mix_is_not_presented_as_current_fixed_allocation(self):
        item = self.validated['waste_sticker']
        self.assertEqual(item['source_mix_status'], 'historical_reference')
        row = next(line for line in self.body.splitlines() if line.startswith('| waste_sticker |'))
        self.assertEqual(row.count('按实际编批'), 3)
        section = self.body.split('### waste_sticker', 1)[1].split('### meter', 1)[0]
        self.assertIn('仅供历史参考', section)
        self.assertIn('直接实拍目标和合成', section)

    def test_invalid_source_mix_status_is_rejected(self):
        data = copy.deepcopy(self.allocation)
        data['tasks']['waste_sticker']['source_mix_status'] = 'fixed'
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

    def test_current_capacity_excludes_source_render_and_samples(self):
        for task, budget in (('meter', 6480), ('waste_bin_state', 0), ('waste_sticker', 1800)):
            with self.subTest(task=task):
                stats = full_plan_design._plan_stats(ROOT, task, self.targets)
                self.assertEqual(stats['budget'], budget)
                self.assertEqual(stats['capacity'], budget)

    def test_generated_doc_is_current(self):
        doc = ROOT / 'docs/FULL_TRAIN_PLAN_DESIGN.md'
        self.assertTrue(doc.is_file())
        self.assertEqual(doc.read_text(encoding='utf-8'), self.body)


if __name__ == '__main__':
    unittest.main()
