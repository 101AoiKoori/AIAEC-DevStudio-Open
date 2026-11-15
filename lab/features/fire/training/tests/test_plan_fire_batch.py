"""Checks for fire batch selection and existing local plan inputs."""
import copy
import json
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import plan_fire_batch as planner


class UseRelationTests(unittest.TestCase):
    def test_all_dimensions_must_match_the_same_relation(self):
        records = [
            {'path': 'crossed', 'task_use_status': [
                {'task': 'fire_sticker', 'split': 'regression', 'role': 'fixed_pattern'},
                {'task': 'other', 'split': 'train', 'role': 'fixed_pattern'},
                {'task': 'fire_sticker', 'split': 'train', 'role': 'background'}]},
            {'path': 'matched', 'task_use_status': [
                {'task': 'fire_sticker', 'split': 'train', 'role': 'fixed_pattern', 'status': 'ready'},
                {'task': 'other', 'split': 'reserved', 'role': 'fixed_pattern', 'status': 'reserved'}]},
        ]
        matches = planner.matching_relations(records, 'fire_sticker', 'train', 'fixed_pattern')
        self.assertEqual([record['path'] for record, _ in matches], ['matched'])
        self.assertEqual(matches[0][1]['status'], 'ready')

    def test_invalid_relation_list_fails(self):
        with self.assertRaises(ValueError):
            planner.matching_relations([{'task_use_status': {}}], 'fire_sticker', 'train', 'background')


class WorkspacePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = planner.ROOT / planner.BASE
        paths = [base / name for name in ('arena_source_and_generation_plan.json',
                 'fire_image_use_manifest_20261114.json',
                 'single_building_C_single_sticker_train_20261114.plan.json')]
        assets = [planner.ROOT / planner.ARTWORK, planner.ROOT / planner.SCENE,
                  planner.ROOT / 'datasets/blender_source_registry.sqlite3']
        if not all(p.is_file() for p in paths + assets):
            raise unittest.SkipTest('local fire production sources are unavailable')
        cls.inputs = [json.loads(p.read_text(encoding='utf8')) for p in paths]

    def test_quota_gap_and_seed_determinism(self):
        plan, report = planner.build_batch(*self.inputs)
        plan_again, report_again = planner.build_batch(*self.inputs)
        self.assertEqual(plan, plan_again)
        self.assertEqual(report, report_again)
        self.assertEqual(len(plan['strata']), 183)
        self.assertEqual((report['target_total'], report['arranged_total'], report['unfilled_total']),
                         (3000, 900, 2100))
        self.assertEqual(len(report['leaves']), 30)
        self.assertTrue(all(leaf['target_images'] == leaf['arranged_attempts'] + leaf['gap']
                            for leaf in report['leaves']))
        changed, changed_report = planner.build_batch(*self.inputs, seed=432)
        self.assertNotEqual([s['id'] for s in plan['strata']], [s['id'] for s in changed['strata']])
        self.assertEqual(changed_report['arranged_total'], 900)

    def test_reserved_relation_does_not_turn_pattern_into_ready(self):
        inputs = copy.deepcopy(self.inputs)
        for record in inputs[1]['records']:
            for relation in record['task_use_status']:
                if (relation.get('task'), relation.get('split'), relation.get('role')) == (
                        'fire_sticker', 'train', 'fixed_pattern'):
                    relation['status'] = 'reserved'
        with self.assertRaises(ValueError):
            planner.build_batch(*inputs)


if __name__ == '__main__':
    unittest.main()
