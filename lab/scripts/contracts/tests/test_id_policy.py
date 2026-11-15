import copy
import unittest
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import id_policy

ROOT = Path(__file__).resolve().parents[4]


class IdPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = id_policy.load_policy()
        self.actual = id_policy.load_actual(ROOT, self.policy)

    def test_actual_policy_and_doc_are_synced(self):
        self.assertEqual(id_policy.validate_scopes(self.policy, self.actual), [])
        counts, rows, errors = id_policy.validate_requirements(self.policy, self.actual)
        self.assertEqual(errors, [])
        self.assertEqual(counts, {
            'canonical': 35,
            'semantic': 22,
            'legacy_numeric': 15,
        })
        self.assertEqual(sum(counts.values()), 72)
        body = id_policy.render_doc(self.policy, self.actual, counts, rows)
        self.assertEqual(
            (ROOT / 'docs/ID_AND_COUNT_POLICY.md').read_text(encoding='utf-8'), body)
        self.assertIn('`building_name_paper`', body)
        self.assertIn('`waste_sticker`', body)
        self.assertIn('`PAIR-VNN`', body)

    def test_unknown_requirement_id_is_reported(self):
        policy = copy.deepcopy(self.policy)
        policy['requirement_ids']['frozen_aliases']['semantic'] = []
        policy['requirement_ids']['frozen_aliases']['legacy_numeric'] = []
        actual = {
            'perception': [{
                'id': 'person',
                'requirements_markdown': '- `[待解决]` PERSON-NEW：新事项\n',
            }],
            'perception_ids': ['person'],
        }
        _, _, errors = id_policy.validate_requirements(policy, actual)
        self.assertTrue(any('未登记的需求 ID' in error for error in errors))

    def test_duplicate_alias_is_rejected(self):
        data = copy.deepcopy(self.policy)
        data['requirement_ids']['frozen_aliases']['legacy_numeric'].append('PERSON-COUNT')
        with self.assertRaises(ValueError):
            id_policy.validate_policy(data)

    def test_mapping_must_cover_every_dataset_task(self):
        data = copy.deepcopy(self.policy)
        del data['task_scopes']['mapping']['person']
        errors = id_policy.validate_scopes(data, self.actual)
        self.assertTrue(any('缺少识别任务: person' in error for error in errors))

    def test_mapping_must_not_drop_a_dataset_task(self):
        data = copy.deepcopy(self.policy)
        data['task_scopes']['mapping']['person'] = []
        errors = id_policy.validate_scopes(data, self.actual)
        self.assertTrue(any('未覆盖数据任务: person' in error for error in errors))

    def test_model_packages_must_be_dataset_plus_extra(self):
        actual = copy.deepcopy(self.actual)
        actual['model_package_ids'] = [
            value for value in actual['model_package_ids'] if value != 'plate']
        errors = id_policy.validate_scopes(self.policy, actual)
        self.assertTrue(any('模型包缺少: plate' in error for error in errors))


if __name__ == '__main__':
    unittest.main()
