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

    def test_machine_validation_evidence_needs_no_human_confirmation_and_preserves_origin(self):
        self.report.pop('review_status')
        self.report.update(evidence_origin='machine', machine_evidence={
            'method': 'detector comparison v1', 'result_reference': 'runs/evaluation.json'})
        for group in self.report['groups']:
            group.pop('review_status')
            group['result_status'] = 'complete'
        self.write_report()
        self.observe('group_a', 'group_b')
        for item in self.request['strata'][0]['observations']:
            item.pop('review_status')
            item['evidence_origin'] = 'machine'
        row = allocation.allocate(self.request, self.root)['strata'][0]
        self.assertEqual((row['observed_groups'], row['failed_groups']), (2, 1))
        self.assertEqual(row['observation_status'], 'validation_groups')
        self.assertEqual(row['group_observations'][0]['evidence_origin'], 'machine')
        self.assertEqual(row['group_observations'][0]['machine_evidence']['result_reference'],
                         'runs/evaluation.json')
        self.assertNotIn('review_status', row['group_observations'][0])

    def test_machine_evidence_rejects_missing_trace_incomplete_result_and_origin_mismatch(self):
        self.report.pop('review_status')
        self.report.update(evidence_origin='machine', machine_evidence={
            'method': 'detector comparison v1', 'result_reference': 'runs/evaluation.json'})
        for group in self.report['groups']:
            group.pop('review_status')
            group['result_status'] = 'complete'
        self.observe('group_a')
        observation = self.request['strata'][0]['observations'][0]
        observation.pop('review_status')
        observation['evidence_origin'] = 'machine'
        self.report['machine_evidence']['result_reference'] = ''
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'result_reference'):
            allocation.allocate(self.request, self.root)
        self.report['machine_evidence']['result_reference'] = 'runs/evaluation.json'
        self.report['groups'][0]['result_status'] = 'pending'
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'binary result'):
            allocation.allocate(self.request, self.root)
        self.report['groups'][0]['result_status'] = 'complete'
        self.write_report()
        observation.update(evidence_origin='human', review_status='confirmed')
        with self.assertRaisesRegex(ValueError, 'origin differ'):
            allocation.allocate(self.request, self.root)

    def test_unselected_pending_groups_do_not_block_selected_group(self):
        self.observe('group_a')
        self.report['groups'][1]['review_status'] = 'pending'
        self.report['groups'][1].pop('failed')
        self.write_report()
        row = allocation.allocate(self.request, self.root)['strata'][0]
        self.assertEqual((row['observed_groups'], row['failed_groups']), (1, 1))
        self.report['groups'][1]['group_id'] = 'group_a'
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'unique ID'):
            allocation.allocate(self.request, self.root)

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
        self.request['design_context'] = {'task': 'person', 'coverage': 'unverified', 'executable': False}
        result = allocation.plan_generation(self.request, self.root)
        self.assertEqual(result['design_context'], self.request['design_context'])
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

    def test_conditioned_plan_balances_people_and_respects_feasible_capacity(self):
        states = []
        for people in range(2, 7):
            for yaw in ('left', 'right'):
                state = self.state(f'{people}_{yaw}', 0, 4)
                state.update(axes={'person_count': people, 'yaw': yaw},
                             background={'id': 'scene'}, target={'count': people},
                             labels={'format': 'yolo'}, augmentation={'light': 'dark'},
                             sources=['scene'], parameters={'yaw': yaw})
                states.append(state)
        self.request.update(budget=23, constraints={'visibility': 'declared'}, strata=states,
                            conditional_balance={'axes': ['person_count', 'yaw'],
                                                 'stratify_by': 'person_count', 'seed': 17})
        result = allocation.plan_generation(self.request, self.root)
        self.assertEqual(result, allocation.plan_generation(self.request, self.root))
        counts = {item['value']: item['count']
                  for item in result['conditional_balance']['counts']['person_count']}
        self.assertEqual(sum(counts.values()), 23)
        self.assertEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertTrue(all(0 <= row['planned_count'] <= 4 for row in result['strata']))
        self.assertEqual(result['strata'][0]['axes'], states[0]['axes'])
        for cell in result['conditional_balance']['cross_counts']['yaw']:
            rows = [row for row in result['strata'] if row['axes']['person_count'] == cell['group_value']
                    and row['axes']['yaw'] == cell['value']]
            self.assertEqual(cell['count'], sum(row['planned_count'] for row in rows))
            self.assertEqual(cell['capacity'], 4 * len(rows))
            self.assertEqual(cell['group_count'], counts[cell['group_value']])

    def test_conditional_cross_counts_only_report_declared_cells(self):
        states = []
        for name, group, background, minimum, maximum in (
            ('a1', 'a', 'x', 1, 3), ('a2', 'a', 'x', 0, 2),
            ('a3', 'a', 'y', 0, 0), ('b1', 'b', 'y', 2, 2),
        ):
            state = self.state(name, minimum, maximum)
            state.update(axes={'category': group, 'background': background},
                         background={'id': background}, target={'id': group},
                         labels={'format': 'yolo'}, augmentation={'light': 'normal'},
                         sources=[background], parameters={'seed': 1})
            states.append(state)
        self.request.update(budget=3, constraints={'visibility': 'declared'}, strata=states,
                            conditional_balance={'axes': ['category', 'background'],
                                                 'stratify_by': 'category', 'seed': 4})
        result = allocation.plan_generation(self.request, self.root)
        self.assertEqual([row['planned_count'] for row in result['strata']], [1, 0, 0, 2])
        cells = result['conditional_balance']['cross_counts']['background']
        self.assertEqual(cells, [
            {'group_value': 'a', 'value': 'x', 'count': 1, 'capacity': 5, 'group_count': 1},
            {'group_value': 'a', 'value': 'y', 'count': 0, 'capacity': 0, 'group_count': 1},
            {'group_value': 'b', 'value': 'y', 'count': 2, 'capacity': 2, 'group_count': 2},
        ])
        self.assertNotIn(('b', 'x'), {(cell['group_value'], cell['value']) for cell in cells})
        self.assertEqual(result, allocation.plan_generation(self.request, self.root))
        self.assertEqual(json.loads(json.dumps(result))['conditional_balance']['cross_counts']['background'], cells)

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
