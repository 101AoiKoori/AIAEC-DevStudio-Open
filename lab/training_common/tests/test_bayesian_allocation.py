"""Checks for bounded, evidence-scoped group-level Bayesian quotas."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lab.training_common import bayesian_allocation as allocation


class BayesianAllocationTest(unittest.TestCase):
    def setUp(self):
        workspace_tmp = Path(__file__).resolve().parents[3] / 'tmp'
        workspace_tmp.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=workspace_tmp)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / 'runs').mkdir()
        self.evidence = self.root / 'runs/val_groups.json'
        self.report = {'schema_version': 1, 'model_key': 'fire_sticker', 'state_id': 'dark_small',
                       'split': 'val', 'review_status': 'confirmed', 'failure_event': 'missed visible fire',
                       'grouping_basis': 'distinct recordings and physical materials',
                       'groups': [{'group_id': 'group_a', 'review_status': 'confirmed', 'failed': True,
                                   'source_ids': ['video_a', 'prop_a']},
                                  {'group_id': 'group_b', 'review_status': 'confirmed', 'failed': False,
                                   'source_ids': ['video_b', 'prop_b']}]}
        self.write_report()
        self.request = {'schema_version': 1, 'model_key': 'fire_sticker', 'target_split': 'train',
                        'purpose': 'synthetic fixture, no production data', 'budget': 17, 'uncertainty_weight': 0.5,
                        'independence_basis': 'distinct recordings and physical materials',
                        'strata': [self.state('dark_small', 2, 10), self.state('normal', 1, 20)]}

    def state(self, name, minimum, maximum):
        return {'id': name, 'failure_event': 'missed visible fire', 'min_count': minimum,
                'max_count': maximum, 'prior': {'alpha': 2, 'beta': 8},
                'exposure_weight': 1, 'failure_cost': 1,
                'conditions': ['low_light', 'low_saturation'] if name == 'dark_small' else [],
                'observations': []}

    def write_report(self):
        self.evidence.write_text(json.dumps(self.report), encoding='utf-8')

    def observe(self, *groups):
        self.request['strata'][0]['observations'] = [
            {'evidence': 'runs/val_groups.json', 'group_id': group, 'split': 'val',
             'review_status': 'confirmed'} for group in groups]

    def test_group_posterior_and_prior_only_are_not_performance_claims(self):
        self.observe('group_a', 'group_b')
        result = allocation.allocate(self.request, self.root)
        dark, normal = result['strata']
        self.assertEqual((dark['observed_groups'], dark['failed_groups']), (2, 1))
        self.assertEqual((dark['posterior']['alpha'], dark['posterior']['beta']), (3, 9))
        self.assertAlmostEqual(dark['posterior']['mean'], .25)
        self.assertAlmostEqual(dark['risk_weight'], .25)
        self.assertAlmostEqual(dark['allocation_weight'], dark['risk_weight'] + dark['uncertainty_component'])
        self.assertLess(dark['posterior']['equal_tail_95'][0], .25)
        self.assertGreater(dark['posterior']['equal_tail_95'][1], .25)
        self.assertEqual(normal['observation_status'], 'prior_only')
        self.assertEqual(normal['posterior']['interpretation'], 'prior_only_not_observed_or_test_performance')
        self.assertEqual(result['planned_total'], 17)
        self.assertAlmostEqual(sum(row['planned_fraction'] for row in result['strata']), 1.0)
        self.assertIn('not state occurrence probabilities', result['scope'])
        self.assertEqual(result['status'], 'planned_not_produced')

    def test_beta_interval_matches_documented_group_example(self):
        self.report['groups'] = [{'group_id': f'g{i}', 'review_status': 'confirmed', 'failed': i < 3,
                                  'source_ids': [f'source{i}']} for i in range(10)]
        self.write_report()
        self.observe(*(f'g{i}' for i in range(10)))
        posterior = allocation.allocate(self.request, self.root)['strata'][0]['posterior']
        self.assertEqual((posterior['alpha'], posterior['beta']), (5, 15))
        self.assertAlmostEqual(posterior['mean'], .25)
        self.assertAlmostEqual(posterior['equal_tail_95'][0], .0915, places=3)
        self.assertAlmostEqual(posterior['equal_tail_95'][1], .4557, places=3)

    def test_rejects_test_unreviewed_and_validation_target_observations(self):
        self.observe('group_a')
        self.report['split'] = 'test'
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'validation evidence'):
            allocation.allocate(self.request, self.root)
        self.report['split'] = 'val'
        self.report['groups'][0]['review_status'] = 'pending'
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'binary result'):
            allocation.allocate(self.request, self.root)
        self.report['groups'][0]['review_status'] = 'confirmed'
        self.write_report()
        (self.root / 'runs/test').mkdir()
        (self.root / 'runs/test/groups.json').write_text(json.dumps(self.report), encoding='utf-8')
        self.request['strata'][0]['observations'][0]['evidence'] = 'runs/test/groups.json'
        with self.assertRaisesRegex(ValueError, 'existing JSON'):
            allocation.allocate(self.request, self.root)
        self.request['strata'][0]['observations'][0]['evidence'] = 'runs/val_groups.json'
        self.request['strata'][0]['observations'][0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'validation observations'):
            allocation.allocate(self.request, self.root)
        self.request['strata'][0]['observations'][0]['split'] = 'val'
        for split in ('val', 'test', 'unassigned'):
            self.request['target_split'] = split
            with self.assertRaisesRegex(ValueError, 'prior-only'):
                allocation.allocate(self.request, self.root)

    def test_duplicate_groups_and_shared_sources_do_not_inflate_sample_size(self):
        self.observe('group_a', 'group_a')
        with self.assertRaisesRegex(ValueError, 'Duplicate group'):
            allocation.allocate(self.request, self.root)
        self.observe('group_a', 'group_b')
        self.report['groups'][1]['source_ids'].append('prop_a')
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'Source is reused'):
            allocation.allocate(self.request, self.root)

    def test_mismatch_missing_evidence_and_minimum_coverage_are_rejected(self):
        self.observe('group_a')
        self.report['failure_event'] = 'different event'
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'failure definition'):
            allocation.allocate(self.request, self.root)
        self.request['strata'][0]['observations'][0]['evidence'] = 'runs/missing.json'
        with self.assertRaisesRegex(ValueError, 'existing JSON'):
            allocation.allocate(self.request, self.root)
        self.request['strata'][0]['observations'] = []
        for budget in (2, 31):
            self.request['budget'] = budget
            with self.assertRaisesRegex(ValueError, 'outside minimum/maximum'):
                allocation.allocate(self.request, self.root)

    def test_large_sparse_state_list_conserves_budget_and_caps(self):
        self.request['strata'] = [self.state(f'state_{i:05}', i % 2, 7) for i in range(5000)]
        self.request['budget'] = 23_001
        first = allocation.allocate(self.request, self.root)
        second = allocation.allocate(self.request, self.root)
        self.assertEqual(first, second)
        self.assertEqual(sum(row['planned_count'] for row in first['strata']), 23_001)
        self.assertTrue(all(row['min_count'] <= row['planned_count'] <= row['max_count']
                            for row in first['strata']))

    def test_extreme_weight_ratios_still_allocate_exact_budget(self):
        result = allocation._weighted_extras([10, 10, 10], [1e-20, 1, 1e15], 15, ['a', 'b', 'c'])
        self.assertEqual(result, [0, 5, 10])

    def test_prior_only_test_does_not_infer_test_performance(self):
        self.request['target_split'] = 'test'
        result = allocation.allocate(self.request, self.root)
        self.assertTrue(all(row['observation_status'] == 'prior_only' for row in result['strata']))
        self.assertIn('Test performance is never inferred', result['scope'])

    def test_generation_plan_carries_concrete_states_and_rejects_feedback(self):
        self.request['constraints'] = {'camera': 'fixed'}
        for state in self.request['strata']:
            state.update(background={'name': 'scene'}, target={'name': 'fire'},
                         labels={'format': 'yolo'}, augmentation={'brightness': 'low'},
                         sources=['scene-a'], parameters={'seed': 2})
        result = allocation.plan_generation(self.request, self.root)
        self.assertEqual(result['strata'][0]['parameters'], {'seed': 2})
        self.assertEqual(result['constraints'], {'camera': 'fixed'})
        self.assertEqual(result['planned_total'], 17)
        self.observe('group_a')
        with self.assertRaisesRegex(ValueError, 'does not accept model feedback'):
            allocation.plan_generation(self.request, self.root)
        self.request['strata'][0]['observations'] = []
        del self.request['strata'][0]['background']
        with self.assertRaisesRegex(ValueError, 'background'):
            allocation.plan_generation(self.request, self.root)

    def test_single_sample_does_not_claim_all_states(self):
        self.request.update(budget=1, constraints={'visibility': 'declared'})
        for state in self.request['strata']:
            state.update(min_count=0, max_count=1, background={'id': 'b'},
                         target={'id': 't'}, labels={'format': 'yolo'},
                         augmentation={'light': 'normal'}, sources=['b'], parameters={'seed': 1})
        result = allocation.plan_generation(self.request, self.root)
        self.assertEqual(sorted(x['planned_count'] for x in result['strata']), [0, 1])
        for state in self.request['strata']:
            state['min_count'] = 1
        with self.assertRaisesRegex(ValueError, 'outside minimum/maximum'):
            allocation.plan_generation(self.request, self.root)

    def test_cli_requires_database_before_any_plan_write(self):
        request = self.root / 'request.json'
        request.write_text(json.dumps(self.request), encoding='utf-8')
        code = self.root / 'allocation.py'
        shutil.copy2(allocation.__file__, code)
        with mock.patch.object(allocation, 'ROOT', self.root), mock.patch.object(allocation, '__file__', str(code)):
            self.assertEqual(allocation.main(['--input', 'request.json', '--output', 'runs/result.json']), 1)
            self.assertEqual(allocation.main(['--input', 'missing.json', '--output', 'runs/fail.json']), 1)
            self.assertEqual(allocation.main(['--input', 'request.json', '--output', 'outside.json']), 1)
        self.assertFalse((self.root / 'runs/result.json').exists())
        self.assertFalse((self.root / 'runs/fail.json').exists())
        self.assertFalse((self.root / 'outside.json').exists())


if __name__ == '__main__':
    unittest.main()
