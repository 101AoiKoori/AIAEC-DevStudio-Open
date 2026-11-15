import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
planner = importlib.import_module('plan_waste_bin_v4_batches')
intake = importlib.import_module('prepare_bin_real_candidate_intake')


class PlannedInputTest(unittest.TestCase):
    def test_elastic_prior_and_cost_change_quotas(self):
        from lab.training_common.bayesian_allocation import allocate
        def plan(alpha, cost):
            rows = [{'id': name, 'failure_event': 'unknown', 'conditions': [], 'min_count': 0,
                     'max_count': 10, 'prior': {'alpha': a, 'beta': 1}, 'exposure_weight': 1,
                     'failure_cost': c, 'observations': []} for name, a, c in
                    [('a', alpha, cost), ('b', 1, 1)]]
            return allocate({'schema_version': 1, 'purpose': 'fixture', 'model_key': 'waste_bin_state',
                             'target_split': 'train', 'budget': 10, 'uncertainty_weight': 0,
                             'independence_basis': 'fixture only', 'strata': rows})['strata'][0]['planned_count']
        self.assertGreater(plan(4, 1), plan(1, 1))
        self.assertGreater(plan(1, 3), plan(1, 1))

    def test_task_split_role_and_status_must_match_one_use(self):
        row = {'task_use_status': [
            {'task': 'waste_bin_state', 'split': 'train', 'role': 'other', 'status': 'ready'},
            {'task': 'waste_sticker', 'split': 'train', 'role': 'direct_sample', 'status': 'ready'},
            {'task': 'waste_bin_state', 'split': 'train', 'role': 'direct_sample', 'status': 'reserved'}]}
        self.assertEqual(planner._matching(row, 'direct_sample'), [])
        row['task_use_status'].append({'task': 'waste_bin_state', 'split': 'train',
                                       'role': 'direct_sample', 'status': 'candidate'})
        self.assertEqual(planner._matching(row, 'direct_sample')[0][0], 3)

    def test_seeded_candidate_batches_use_only_matching_relations(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as name:
            root = Path(name)
            for rel in (planner.V2, planner.V3, 'image.png', 'label.txt'):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
            scenes = [{'asset': path, 'task_use_status': [{'task': 'waste_bin_state', 'split': 'train',
                      'role': 'blender_scene', 'status': 'candidate'}]} for path in (planner.V2, planner.V3)]
            uses = [{'task': 'waste_bin_state', 'split': 'train', 'role': 'direct_sample',
                     'status': 'candidate', 'label_status': 'candidate_complete_positive',
                     'label': 'label.txt', 'candidate_boxes': [1, 2, 3, 4]}]
            relations = {'handoff': {'asset_task_relations': scenes,
                         'image_task_relations': [{'image': 'image.png', 'source_group': 'parent',
                                                  'task_use_status': uses}]}}
            targets = {'D4': 800, 'Dfew': 150, 'Dneg': 100, 'R4': 800, 'Rfew': 400,
                       'Rocc': 200, 'B4': 600, 'Bfew': 300, 'Bocc': 150}
            full = {'combinations': [{'id': key, 'target': value} for key, value in targets.items()]}
            for entry in full['combinations']:
                if entry['id'] in ('B4', 'Bfew', 'Bocc'):
                    first = {'B4': 350, 'Bfew': 150, 'Bocc': 100}[entry['id']]
                    entry['shares'] = {'arena_master': first, 'arena_detailed': entry['target'] - first}
            with patch.object(planner, 'ROOT', root), patch.object(planner, 'RELATIONS', root / 'relations.json'), \
                 patch.object(planner, 'MAIN', root / 'source.json'):
                direct, scenes_a, report = planner.build(42, relations, full)
                again, scenes_b, report_b = planner.build(42, relations, full)
            self.assertEqual((direct, scenes_a, report), (again, scenes_b, report_b))
            self.assertEqual(sum(len(s['strata']) for s in scenes_a.values()), 1050)
            allocated = report['semantic_allocation']['strata']
            self.assertTrue(all(r['min_count'] <= r['planned_count'] <= r['max_count'] for r in allocated))
            self.assertTrue(any(r['planned_count'] != {'B4': 600, 'Bfew': 300, 'Bocc': 150}[r['id'].split(':')[0]]
                                for r in allocated))
            self.assertEqual(len(direct['records']), 1)
            self.assertEqual(direct['records'][0]['combination_id'], 'D4')
            self.assertEqual(next(x for x in report['branches'] if x['id'] == 'R4')['entry_arrangeable_attempts'], 0)
            occ = next(x for x in report['branches'] if x['id'] == 'Bocc')
            self.assertGreater(occ['entry_arrangeable_attempts'], 150)
            self.assertEqual(occ['unverified_occlusion_gap'], 150)

    def test_intake_preserves_candidate_truth_and_rejects_regression(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as name:
            root = Path(name)
            for value in ('image.png', 'label.txt'):
                (root / value).write_text('', encoding='utf8')
            original = {'images': [{'image': 'image.png', 'split': 'unassigned',
                                    'tasks': {'waste_bin_state': {'label': 'label.txt',
                                    'state': 'candidate_complete_negative', 'boxes': []}}}]}
            (root / 'review.json').write_text(json.dumps(original), encoding='utf8')
            use = {'task': 'waste_bin_state', 'split': 'train', 'role': 'direct_sample',
                   'status': 'candidate', 'label': 'label.txt', 'label_status': 'candidate_complete_negative',
                   'candidate_boxes': [], 'human_accepted': False}
            source = {'handoff': {'image_task_relations': [{'image': 'image.png',
                      'original_record': {'manifest': 'review.json', 'json_pointer': '/images/0'},
                      'task_use_status': [use]}]}}
            (root / 'relations.json').write_text(json.dumps(source), encoding='utf8')
            declaration = {'model_key': 'waste_bin_state', 'target_split': 'train',
                           'relation_source': 'relations.json', 'records': [{'id': 'one',
                           'relation_index': 0, 'use_index': 0, 'image': 'image.png', 'label': 'label.txt',
                           'combination_id': 'Dneg'}]}
            path = root / 'plan.json'
            path.write_text(json.dumps(declaration), encoding='utf8')
            with patch.object(intake, 'ROOT', root):
                rows, _ = intake.collect_planned(path)
                self.assertEqual(rows[0]['source_label_state'], 'candidate_complete_negative')
                self.assertFalse(rows[0]['human_accepted'])
                original['images'][0]['split'] = 'regression'
                (root / 'review.json').write_text(json.dumps(original), encoding='utf8')
                with self.assertRaisesRegex(ValueError, 'regression'):
                    intake.collect_planned(path)
                original['images'][0]['split'] = 'unassigned'
                (root / 'review.json').write_text(json.dumps(original), encoding='utf8')
                use['status'] = 'reserved'
                (root / 'relations.json').write_text(json.dumps(source), encoding='utf8')
                with self.assertRaisesRegex(ValueError, 'train direct candidate'):
                    intake.collect_planned(path)


if __name__ == '__main__':
    unittest.main()
