import json
import unittest
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import state_coverage

ROOT = Path(__file__).resolve().parents[4]


def plan(task, name='production_plan.json'):
    path = ROOT / 'datasets/scripts' / task / 'config' / name
    return json.loads(path.read_text(encoding='utf-8'))


class ExtractTest(unittest.TestCase):
    def test_person_target_and_axis_are_separate(self):
        stratum = next(item for item in plan('person')['strata']
                       if '_community_none_' in item['id'])
        self.assertEqual(state_coverage._person_targets(stratum)['classes'], ['community'])
        self.assertEqual(state_coverage._person_targets(stratum)['count'], ['single'])
        axes = state_coverage._person_axes(stratum)
        self.assertEqual(axes['vehicle_occlusion'], ['none'])
        self.assertEqual(axes['view'], ['front'])

    def test_e_bike_pose_is_target_not_axis(self):
        stratum = next(item for item in plan('e_bike')['strata']
                       if (item.get('target') or {}).get('vehicles'))
        targets = state_coverage._e_bike_targets(stratum)
        self.assertTrue(targets['classes'])
        self.assertTrue(set(targets['classes']) <= {'upright', 'fallen'})
        self.assertNotIn('pose', state_coverage._e_bike_axes(stratum))

    def test_fire_count_and_facade(self):
        stratum = plan('fire')['strata'][0]
        self.assertEqual(state_coverage._fire_targets(stratum)['classes'], ['fire_sticker'])
        self.assertIn('facade', state_coverage._fire_axes(stratum))

    def test_building_name_paper_target_and_building_axis(self):
        strata = plan('building_name_paper')['strata']
        yes = next(item for item in strata if (item.get('target') or {}).get('papers'))
        no = next(item for item in strata if 'papers' in (item.get('target') or {})
                  and not item['target']['papers'])
        self.assertEqual(state_coverage._building_name_paper_targets(yes)['classes'], ['paper'])
        self.assertEqual(state_coverage._building_name_paper_targets(no)['classes'], ['no_paper'])
        blender = plan('building_name_paper', 'blender_plan.json')['strata'][0]
        self.assertIn('building_count', state_coverage._building_name_paper_axes(blender))

    def test_sign_truncation_is_axis(self):
        stratum = next(item for item in plan('sign')['strata']
                       if (item.get('parameters') or {}).get('visible_kind') == 'identifiable_pole')
        self.assertEqual(state_coverage._sign_targets(stratum)['classes'], ['sign_fixture'])
        self.assertEqual(state_coverage._sign_axes(stratum)['truncation'], ['pole_only'])

    def test_waste_bin_state_is_target_and_color_is_axis(self):
        stratum = next(item for item in plan('waste_bin_state', 'render_plan.json')['strata']
                       if (item.get('target') or {}).get('bins'))
        targets = state_coverage._waste_bin_state_targets(stratum)
        self.assertTrue(set(targets['classes']) <= {'open', 'closed'})
        axes = state_coverage._waste_bin_state_axes(stratum)
        self.assertTrue(set(axes['color']) <= {'blue', 'red', 'green', 'gray'})

    def test_waste_sticker_class_is_target_and_bin_color_is_axis(self):
        stratum = plan('waste_sticker')['strata'][0]
        targets = state_coverage._waste_sticker_targets(stratum)
        self.assertTrue(targets['classes'])
        self.assertNotIn('class', state_coverage._waste_sticker_axes(stratum))
        self.assertEqual(state_coverage._waste_sticker_axes(stratum)['bin_color'], ['red'])

    def test_meter_needle_is_target_and_distance_is_axis(self):
        stratum = plan('meter', 'render_plan.json')['strata'][0]
        targets = state_coverage._meter_targets(stratum)
        self.assertEqual(targets['presence'], ['single'])
        self.assertTrue(targets['needle_position'])
        self.assertIn('distance', state_coverage._meter_axes(stratum))


class SummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = state_coverage.summarize(ROOT)
        cls.body = state_coverage.render_markdown(cls.summary)

    def test_all_tasks_present(self):
        self.assertEqual(set(self.summary['tasks']), set(state_coverage.TASK_ORDER))

    def test_person_targets_exclude_vehicle_occlusion(self):
        person = self.summary['tasks']['person']
        self.assertIn('community', person['targets']['classes'])
        self.assertIn('noncommunity', person['targets']['classes'])
        self.assertIn('vehicle_occlusion', person['axes'])
        self.assertEqual(person['train_capacity'], 54)
        self.assertEqual(person['capacity_gap'], 2946)

    def test_ebike_pose_is_target(self):
        bike = self.summary['tasks']['e_bike']
        self.assertIn('fallen', bike['targets']['classes'])
        self.assertIn('upright', bike['targets']['classes'])
        self.assertNotIn('pose', bike['axes'])

    def test_sticker_classes_are_target(self):
        sticker = self.summary['tasks']['waste_sticker']
        self.assertEqual(set(sticker['targets']['classes']), {str(i) for i in range(24)})
        self.assertNotIn('class', sticker['axes'])

    def test_bin_state_is_target_color_is_axis(self):
        bins = self.summary['tasks']['waste_bin_state']
        self.assertIn('open', bins['targets']['classes'])
        self.assertIn('closed', bins['targets']['classes'])
        self.assertIn('color', bins['axes'])
        self.assertNotIn('open', bins['axes']['color'])

    def test_sign_view_and_truncation_are_axes(self):
        sign = self.summary['tasks']['sign']
        self.assertEqual(sign['targets']['classes'], {'sign_fixture': 400})
        self.assertIn('front', sign['axes']['view'])
        self.assertIn('pole_only', sign['axes']['truncation'])

    def test_gap_reports_capacity_shortfall(self):
        sign = self.summary['tasks']['sign']
        self.assertEqual(sign['target_images'], 2500)
        self.assertEqual(sign['train_capacity'], 400)
        self.assertEqual(sign['capacity_gap'], 2100)

    def test_markdown_separates_targets_and_axes(self):
        self.assertIn('**训练目标覆盖', self.body)
        self.assertIn('**组合轴覆盖', self.body)
        target_part, axes_part = self.body.split('**组合轴覆盖', 1)
        self.assertIn('community', target_part)
        self.assertNotIn('community', axes_part)

    def test_target_and_axis_fields_do_not_overlap(self):
        targets = state_coverage.load_targets()
        axes = state_coverage.load_axes()
        target_fields = {'classes', 'counts', 'presence', 'ocr', 'keypoints',
                         'needle_axes', 'negatives', 'notes', 'plans'}
        for task in state_coverage.TASK_ORDER:
            self.assertFalse(
                target_fields & set(axes['tasks'][task]['axes']),
                '{} 的目标字段与组合轴重名'.format(task))
        self.assertIn('community', targets['tasks']['person']['classes'])
        self.assertNotIn('pose', axes['tasks']['e_bike']['axes'])
        self.assertNotIn('class', axes['tasks']['waste_sticker']['axes'])
        self.assertNotIn('state', axes['tasks']['waste_bin_state']['axes'])

    def test_generated_block_is_current(self):
        text = (ROOT / 'docs/DATASET_DESIGN.md').read_text(encoding='utf-8')
        start = text.index(state_coverage.START) + len(state_coverage.START)
        end = text.index(state_coverage.END)
        self.assertEqual(text[start:end].strip('\n'), self.body)


if __name__ == '__main__':
    unittest.main()
